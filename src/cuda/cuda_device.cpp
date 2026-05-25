// Copyright (c) 2025 Digital Anarchy, Inc. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#if WITH_CUDA

// <delayimp.h> requires immediate-after-<windows.h> ordering, which the
// transitive <cuda.h> include downstream breaks. Forward-declare the one
// helper we use instead of including it.
#if defined(WITH_CUDA_DELAYLOAD) && defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
extern "C" HRESULT __stdcall __HrLoadAllImportsForDll(const char* szDll);
#ifndef FACILITY_VISUALCPP
#define FACILITY_VISUALCPP 0x6d
#endif
#ifndef VcppException
#define VcppException(sev, code) ((sev) | (FACILITY_VISUALCPP << 16) | (code))
#endif
#endif

#include <ghost/allocator.h>
#include <ghost/cuda/device.h>
#include <ghost/cuda/exception.h>
#include <ghost/cuda/impl_device.h>
#include <ghost/cuda/impl_function.h>

#include <atomic>
#include <cstring>
#include <new>
#include <sstream>
#include <utility>
#include <vector>

namespace ghost {
namespace {
#if defined(WITH_CUDA_DELAYLOAD) && defined(_WIN32)
// SEH-required: __HrLoadAllImportsForDll raises a Win32 exception
// (ERROR_MOD_NOT_FOUND) when nvcuda.dll is absent, not a C++ throw.
bool probeCudaDriverOnce() {
  bool ok = true;
  __try {
    if (FAILED(__HrLoadAllImportsForDll("nvcuda.dll"))) {
      ok = false;
    }
  } __except (GetExceptionCode() ==
                      VcppException(ERROR_SEVERITY_ERROR, ERROR_MOD_NOT_FOUND)
                  ? EXCEPTION_EXECUTE_HANDLER
                  : EXCEPTION_CONTINUE_SEARCH) {
    ok = false;
  }
  return ok;
}

bool isCudaDriverAvailable() {
  static const bool available = probeCudaDriverOnce();
  return available;
}
#else
inline bool isCudaDriverAvailable() { return true; }
#endif
}  // namespace

namespace implementation {
using namespace cu;

namespace {
// Shared state for a deferred CUdeviceptr release. The cu::ptr inside owns
// the device memory; when the last scheduled host-fn callback runs, the
// DeferredRelease is deleted and the cu::ptr destructor calls cuMemFree.
struct DeferredRelease {
  std::atomic<int> remaining{0};
  cu::ptr<CUdeviceptr> mem;

  DeferredRelease() : mem(0, false) {}
};

void CUDA_CB releaseDeferredCallback(void* p) {
  auto* d = static_cast<DeferredRelease*>(p);
  if (d->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    delete d;
  }
}

// Schedule cuMemFree of @p mem to fire after pending work on each stream in
// @p streams has completed. On any scheduling failure, the cu::ptr's
// destructor runs synchronously (matching pre-fix behavior — the caller has
// already violated the lifetime contract).
void scheduleDeferredRelease(cu::ptr<CUdeviceptr>&& mem,
                             const std::vector<CUstream>& streams) {
  if (!mem.get() || !mem.owned()) return;  // nothing to free
  if (streams.empty()) return;  // caller path uses synchronous release
  auto* d = new DeferredRelease{};
  d->remaining.store(static_cast<int>(streams.size()),
                     std::memory_order_relaxed);
  d->mem = std::move(mem);
  for (CUstream s : streams) {
    CUresult err = cuLaunchHostFunc(s, &releaseDeferredCallback, d);
    if (err != CUDA_SUCCESS) {
      // Decrement on this stream's behalf so the remaining callbacks (or the
      // synchronous fallback below) get the count right.
      if (d->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        delete d;
        return;
      }
    }
  }
}
}  // namespace

EventCUDA::EventCUDA(cu::ptr<CUevent> event_) : event(event_) {}

void EventCUDA::wait() {
  CUresult err = cuEventSynchronize(event);
  checkError(err);
}

bool EventCUDA::isComplete() const {
  CUresult err = cuEventQuery(event);
  if (err == CUDA_SUCCESS) return true;
  if (err == CUDA_ERROR_NOT_READY) return false;
  checkError(err);
  return false;
}

double EventCUDA::elapsed(const Event& other) const {
  auto& otherCUDA = static_cast<const EventCUDA&>(other);
  float ms = 0.0f;
  CUresult err = cuEventElapsedTime(&ms, event, otherCUDA.event);
  if (err != CUDA_SUCCESS) return 0.0;
  return static_cast<double>(ms) / 1000.0;
}

StreamCUDA::StreamCUDA(cu::ptr<CUstream> queue_) : queue(queue_) {}

StreamCUDA::StreamCUDA(CUcontext dev) {
  CUresult err;
  err = cuStreamCreate(&queue, CU_STREAM_NON_BLOCKING);
  checkError(err);
}

void StreamCUDA::sync() {
  CUresult err;
  if (queue)
    err = cuStreamSynchronize(queue);
  else
    err = cuCtxSynchronize();
  checkError(err);
}

std::shared_ptr<Event> StreamCUDA::record() {
  cu::ptr<CUevent> ev;
  CUresult err;
  err = cuEventCreate(&ev, CU_EVENT_DEFAULT);
  checkError(err);
  err = cuEventRecord(ev, queue);
  checkError(err);
  return std::make_shared<EventCUDA>(ev);
}

void StreamCUDA::waitForEvent(const std::shared_ptr<Event>& e) {
  auto eventCUDA = static_cast<EventCUDA*>(e.get());
  CUresult err = cuStreamWaitEvent(queue, eventCUDA->event, 0);
  checkError(err);
}

BufferCUDA::BufferCUDA(cu::ptr<CUdeviceptr> mem_, size_t bytes)
    : mem(mem_), _size(bytes) {}

BufferCUDA::~BufferCUDA() {
  if (_allocator) {
    _allocator->freeBuffer(
        reinterpret_cast<void*>(static_cast<uintptr_t>(mem.release())), _size);
    return;
  }
  // Defer cuMemFree until pending work on each used stream completes.
  // Non-owning mem (sub-buffers, sharedImage donors) falls through to the
  // cu::ptr destructor, which is a no-op.
  scheduleDeferredRelease(std::move(mem), _useStreams);
}

void BufferCUDA::markUsed(CUstream s) {
  for (CUstream existing : _useStreams) {
    if (existing == s) return;
  }
  _useStreams.push_back(s);
}

BufferCUDA::BufferCUDA(const DeviceCUDA& dev, size_t bytes,
                       const BufferOptions&)
    : _size(bytes) {
  // TODO: honor opts.hint (Staging → cuMemHostAlloc pinned host)
  CUresult err;
  err = cuMemAlloc(&mem, bytes);
  checkError(err);
}

size_t BufferCUDA::size() const { return _size; }

void BufferCUDA::copy(const ghost::Encoder& s, const ghost::Buffer& src,
                      size_t bytes) {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  auto src_impl = static_cast<implementation::BufferCUDA*>(src.impl().get());
  markUsed(stream_impl->queue);
  src_impl->markUsed(stream_impl->queue);
  CUresult err;
  err = cuMemcpyDtoDAsync(mem, src_impl->mem, bytes, stream_impl->queue);
  checkError(err);
}

void BufferCUDA::copy(const ghost::Encoder& s, const void* src, size_t bytes) {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  markUsed(stream_impl->queue);
  CUresult err;
  err = cuMemcpyHtoDAsync(mem, src, bytes, stream_impl->queue);
  checkError(err);
}

void BufferCUDA::copyTo(const ghost::Encoder& s, void* dst,
                        size_t bytes) const {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  const_cast<BufferCUDA*>(this)->markUsed(stream_impl->queue);
  CUresult err;
  err = cuMemcpyDtoHAsync(dst, mem, bytes, stream_impl->queue);
  checkError(err);
}

void BufferCUDA::copy(const ghost::Encoder& s, const ghost::Buffer& src,
                      size_t srcOffset, size_t dstOffset, size_t bytes) {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  auto src_impl = static_cast<implementation::BufferCUDA*>(src.impl().get());
  markUsed(stream_impl->queue);
  src_impl->markUsed(stream_impl->queue);
  CUresult err;
  err =
      cuMemcpyDtoDAsync(mem.get() + dstOffset, src_impl->mem.get() + srcOffset,
                        bytes, stream_impl->queue);
  checkError(err);
}

void BufferCUDA::copy(const ghost::Encoder& s, const void* src,
                      size_t dstOffset, size_t bytes) {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  markUsed(stream_impl->queue);
  CUresult err;
  err =
      cuMemcpyHtoDAsync(mem.get() + dstOffset, src, bytes, stream_impl->queue);
  checkError(err);
}

void BufferCUDA::copyTo(const ghost::Encoder& s, void* dst, size_t srcOffset,
                        size_t bytes) const {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  const_cast<BufferCUDA*>(this)->markUsed(stream_impl->queue);
  CUresult err;
  err =
      cuMemcpyDtoHAsync(dst, mem.get() + srcOffset, bytes, stream_impl->queue);
  checkError(err);
}

void BufferCUDA::fill(const ghost::Encoder& s, size_t offset, size_t size,
                      uint8_t value) {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  markUsed(stream_impl->queue);
  CUresult err;
  err = cuMemsetD8Async(mem.get() + offset, value, size, stream_impl->queue);
  checkError(err);
}

void BufferCUDA::fill(const ghost::Encoder& s, size_t offset, size_t size,
                      const void* pattern, size_t patternSize) {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  markUsed(stream_impl->queue);
  CUresult err;
  CUdeviceptr dst = mem.get() + offset;
  if (patternSize == 1) {
    err = cuMemsetD8Async(dst, *static_cast<const uint8_t*>(pattern), size,
                          stream_impl->queue);
  } else if (patternSize == 2) {
    unsigned short v;
    memcpy(&v, pattern, 2);
    err = cuMemsetD16Async(dst, v, size / 2, stream_impl->queue);
  } else if (patternSize == 4) {
    unsigned int v;
    memcpy(&v, pattern, 4);
    err = cuMemsetD32Async(dst, v, size / 4, stream_impl->queue);
  } else {
    // For non-standard pattern sizes, fill from host
    std::vector<uint8_t> buf(size);
    for (size_t i = 0; i < size; i += patternSize) {
      size_t n = std::min(patternSize, size - i);
      memcpy(buf.data() + i, pattern, n);
    }
    err = cuMemcpyHtoDAsync(dst, buf.data(), size, stream_impl->queue);
  }
  checkError(err);
}

std::shared_ptr<Buffer> BufferCUDA::createSubBuffer(
    const std::shared_ptr<Buffer>& self, size_t offset, size_t size) {
  cu::ptr<CUdeviceptr> subMem(mem.get() + offset, false);
  return std::make_shared<SubBufferCUDA>(self, subMem, size);
}

SubBufferCUDA::SubBufferCUDA(std::shared_ptr<Buffer> parent,
                             cu::ptr<CUdeviceptr> mem_, size_t bytes)
    : BufferCUDA(mem_, bytes), _parent(parent) {}

void SubBufferCUDA::markUsed(CUstream s) {
  // Propagate to the parent so the parent's deferred release waits for any
  // pending work referencing this sub-region. Also record locally so direct
  // uses of this sub-buffer wrapper are tracked (defensive: a sub-buffer's
  // mem is non-owning, but recording is cheap and keeps invariants uniform).
  if (auto* p = static_cast<BufferCUDA*>(_parent.get())) {
    p->markUsed(s);
  }
  BufferCUDA::markUsed(s);
}

MappedBufferCUDA::MappedBufferCUDA(cu::ptr<void*> ptr_)
    : BufferCUDA(cu::ptr<CUdeviceptr>(), 0), ptr(ptr_) {
  CUdeviceptr p;
  CUresult err;
  err = cuMemHostGetDevicePointer(&p, ptr, 0);
  checkError(err);
  mem = cu::ptr<CUdeviceptr>(p, false);  // do not free the device pointer
}

MappedBufferCUDA::~MappedBufferCUDA() {
  if (_allocator) {
    // The base BufferCUDA destructor would also try to free `mem`, but
    // `mem` is non-owning (cuMemHostGetDevicePointer derived). Clearing
    // _allocator prevents BufferCUDA::~BufferCUDA from calling freeBuffer.
    _allocator->freeMappedBuffer(ptr.release(), _size);
    _allocator = nullptr;
  }
}

MappedBufferCUDA::MappedBufferCUDA(const DeviceCUDA& dev, size_t bytes,
                                   const BufferOptions& opts)
    : BufferCUDA(cu::ptr<CUdeviceptr>(), bytes) {
  unsigned int flags = CU_MEMHOSTALLOC_DEVICEMAP;
  if (opts.access == Access::WriteOnly) {
    flags |= CU_MEMHOSTALLOC_WRITECOMBINED;
  }
  CUresult err;
  err = cuMemHostAlloc(&ptr, bytes, flags);
  checkError(err);
  CUdeviceptr p;
  err = cuMemHostGetDevicePointer(&p, ptr, 0);
  checkError(err);
  mem = cu::ptr<CUdeviceptr>(p, false);  // do not free the device pointer
}

void* MappedBufferCUDA::map(const ghost::Encoder& s, Access access, bool sync) {
  if (sync) {
    // TODO
  }
  return ptr;
}

void MappedBufferCUDA::unmap(const ghost::Encoder&) {}

ImageCUDA::ImageCUDA(cu::ptr<CUdeviceptr> mem_, const ImageDescription& descr_)
    : mem(mem_), descr(descr_) {}

ImageCUDA::~ImageCUDA() {
  if (_allocator) {
    _allocator->freeImage(
        reinterpret_cast<void*>(static_cast<uintptr_t>(mem.release())), descr);
    return;
  }
  scheduleDeferredRelease(std::move(mem), _useStreams);
}

void ImageCUDA::markUsed(CUstream s) {
  for (CUstream existing : _useStreams) {
    if (existing == s) return;
  }
  _useStreams.push_back(s);
}

ImageCUDA::ImageCUDA(const DeviceCUDA& dev, const ImageDescription& descr_)
    : descr(descr_) {
  CUresult err;
  size_t pitch;
  descr = descr_;
  size_t bytes = descr.pixelSize();
  size_t elementSize = std::max((size_t)4, std::min(bytes, (size_t)16));
  err = cuMemAllocPitch(&mem, &pitch, descr.size.x * bytes,
                        descr.size.y * descr.size.z, elementSize);
  checkError(err);
  descr.stride = Stride2(static_cast<int32_t>(pitch),
                         static_cast<int32_t>(pitch * descr.size.y));
}

ImageCUDA::ImageCUDA(const DeviceCUDA& dev, const ImageDescription& descr_,
                     BufferCUDA& buffer)
    : descr(descr_) {
  descr = descr_;
  mem = buffer.mem;
}

ImageCUDA::ImageCUDA(const DeviceCUDA& dev, const ImageDescription& descr_,
                     ImageCUDA& image)
    : descr(descr_) {
  descr = descr_;
  mem = image.mem;
}

void ImageCUDA::copy(const ghost::Encoder& s, const ghost::Image& src) {
  auto src_impl = static_cast<implementation::ImageCUDA*>(src.impl().get());
  {
    auto sc = static_cast<implementation::StreamCUDA*>(s.impl().get());
    markUsed(sc->queue);
    src_impl->markUsed(sc->queue);
  }
  size_t srcPixSize = src_impl->descr.pixelSize();
  size_t srcRowBytes = src_impl->descr.rowBytes(srcPixSize);
  size_t srcSliceBytes = src_impl->descr.sliceBytes(srcRowBytes);
  size_t dstPixSize = descr.pixelSize();
  size_t dstRowBytes = descr.rowBytes(dstPixSize);
  size_t dstSliceBytes = descr.sliceBytes(dstRowBytes);
  if (descr.size.z > 1) {
    CUDA_MEMCPY3D a;
    a.srcXInBytes = 0;
    a.srcY = 0;
    a.srcZ = 0;
    a.srcLOD = 0;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = src_impl->mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.srcHeight = srcSliceBytes / srcRowBytes;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstZ = 0;
    a.dstLOD = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.dstHeight = dstSliceBytes / dstRowBytes;
    a.WidthInBytes = descr.size.x * dstPixSize;
    a.Height = descr.size.y;
    a.Depth = descr.size.z;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(stream_impl->queue);
    CUresult err;
    err = cuMemcpy3DAsync(&a, stream_impl->queue);
    checkError(err);
  } else {
    CUDA_MEMCPY2D a;
    a.srcXInBytes = 0;
    a.srcY = 0;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = src_impl->mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.WidthInBytes = descr.size.x * dstPixSize;
    a.Height = descr.size.y;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(stream_impl->queue);
    CUresult err;
    err = cuMemcpy2DAsync(&a, stream_impl->queue);
    checkError(err);
  }
}

void ImageCUDA::copy(const ghost::Encoder& s, const ghost::Buffer& src,
                     const BufferLayout& layout) {
  auto src_impl = static_cast<implementation::BufferCUDA*>(src.impl().get());
  {
    auto sc = static_cast<implementation::StreamCUDA*>(s.impl().get());
    markUsed(sc->queue);
    src_impl->markUsed(sc->queue);
  }
  size_t pixSize = descr.pixelSize();
  size_t srcRowBytes = layout.rowBytes(pixSize);
  size_t srcSliceBytes = layout.sliceBytes(srcRowBytes);
  size_t dstRowBytes = descr.rowBytes(pixSize);
  size_t dstSliceBytes = descr.sliceBytes(dstRowBytes);
  if (descr.size.z > 1) {
    CUDA_MEMCPY3D a;
    a.srcXInBytes = 0;
    a.srcY = 0;
    a.srcZ = 0;
    a.srcLOD = 0;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = src_impl->mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.srcHeight = srcSliceBytes / srcRowBytes;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstZ = 0;
    a.dstLOD = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.dstHeight = dstSliceBytes / dstRowBytes;
    a.WidthInBytes = descr.size.x * pixSize;
    a.Height = descr.size.y;
    a.Depth = descr.size.z;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(stream_impl->queue);
    CUresult err;
    err = cuMemcpy3DAsync(&a, stream_impl->queue);
    checkError(err);
  } else {
    CUDA_MEMCPY2D a;
    a.srcXInBytes = 0;
    a.srcY = 0;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = src_impl->mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.WidthInBytes = descr.size.x * pixSize;
    a.Height = descr.size.y;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(stream_impl->queue);
    CUresult err;
    err = cuMemcpy2DAsync(&a, stream_impl->queue);
    checkError(err);
  }
}

void ImageCUDA::copy(const ghost::Encoder& s, const void* src,
                     const BufferLayout& layout) {
  size_t pixSize = descr.pixelSize();
  size_t srcRowBytes = layout.rowBytes(pixSize);
  size_t srcSliceBytes = layout.sliceBytes(srcRowBytes);
  size_t dstRowBytes = descr.rowBytes(pixSize);
  size_t dstSliceBytes = descr.sliceBytes(dstRowBytes);
  if (descr.size.z > 1) {
    CUDA_MEMCPY3D a;
    a.srcXInBytes = 0;
    a.srcY = 0;
    a.srcZ = 0;
    a.srcLOD = 0;
    a.srcMemoryType = CU_MEMORYTYPE_HOST;
    a.srcHost = src;
    a.srcDevice = (CUdeviceptr)0;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.srcHeight = srcSliceBytes / srcRowBytes;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstZ = 0;
    a.dstLOD = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.dstHeight = dstSliceBytes / dstRowBytes;
    a.WidthInBytes = descr.size.x * pixSize;
    a.Height = descr.size.y;
    a.Depth = descr.size.z;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(stream_impl->queue);
    CUresult err;
    err = cuMemcpy3DAsync(&a, stream_impl->queue);
    checkError(err);
  } else {
    CUDA_MEMCPY2D a;
    a.srcXInBytes = 0;
    a.srcY = 0;
    a.srcMemoryType = CU_MEMORYTYPE_HOST;
    a.srcHost = src;
    a.srcDevice = (CUdeviceptr)0;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.WidthInBytes = descr.size.x * pixSize;
    a.Height = descr.size.y;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(stream_impl->queue);
    CUresult err;
    err = cuMemcpy2DAsync(&a, stream_impl->queue);
    checkError(err);
  }
}

void ImageCUDA::copyTo(const ghost::Encoder& s, ghost::Buffer& dst,
                       const BufferLayout& layout) const {
  auto dst_impl = static_cast<implementation::BufferCUDA*>(dst.impl().get());
  {
    auto sc = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(sc->queue);
    dst_impl->markUsed(sc->queue);
  }
  size_t pixSize = descr.pixelSize();
  size_t srcRowBytes = descr.rowBytes(pixSize);
  size_t srcSliceBytes = descr.sliceBytes(srcRowBytes);
  size_t dstRowBytes = layout.rowBytes(pixSize);
  size_t dstSliceBytes = layout.sliceBytes(dstRowBytes);
  if (descr.size.z > 1) {
    CUDA_MEMCPY3D a;
    a.srcXInBytes = 0;
    a.srcY = 0;
    a.srcZ = 0;
    a.srcLOD = 0;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.srcHeight = srcSliceBytes / srcRowBytes;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstZ = 0;
    a.dstLOD = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = dst_impl->mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.dstHeight = dstSliceBytes / dstRowBytes;
    a.WidthInBytes = layout.size.x * pixSize;
    a.Height = layout.size.y;
    a.Depth = layout.size.z;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(stream_impl->queue);
    CUresult err;
    err = cuMemcpy3DAsync(&a, stream_impl->queue);
    checkError(err);
  } else {
    CUDA_MEMCPY2D a;
    a.srcXInBytes = 0;
    a.srcY = 0;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = dst_impl->mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.WidthInBytes = layout.size.x * pixSize;
    a.Height = layout.size.y;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(stream_impl->queue);
    CUresult err;
    err = cuMemcpy2DAsync(&a, stream_impl->queue);
    checkError(err);
  }
}

void ImageCUDA::copyTo(const ghost::Encoder& s, void* dst,
                       const BufferLayout& layout) const {
  size_t pixSize = descr.pixelSize();
  size_t srcRowBytes = descr.rowBytes(pixSize);
  size_t srcSliceBytes = descr.sliceBytes(srcRowBytes);
  size_t dstRowBytes = layout.rowBytes(pixSize);
  size_t dstSliceBytes = layout.sliceBytes(dstRowBytes);
  if (descr.size.z > 1) {
    CUDA_MEMCPY3D a;
    a.srcXInBytes = 0;
    a.srcY = 0;
    a.srcZ = 0;
    a.srcLOD = 0;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.srcHeight = srcSliceBytes / srcRowBytes;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstZ = 0;
    a.dstLOD = 0;
    a.dstMemoryType = CU_MEMORYTYPE_HOST;
    a.dstHost = dst;
    a.dstDevice = (CUdeviceptr)0;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.dstHeight = dstSliceBytes / dstRowBytes;
    a.WidthInBytes = layout.size.x * pixSize;
    a.Height = layout.size.y;
    a.Depth = layout.size.z;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(stream_impl->queue);
    CUresult err;
    err = cuMemcpy3DAsync(&a, stream_impl->queue);
    checkError(err);
  } else {
    CUDA_MEMCPY2D a;
    a.srcXInBytes = 0;
    a.srcY = 0;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstMemoryType = CU_MEMORYTYPE_HOST;
    a.dstHost = dst;
    a.dstDevice = (CUdeviceptr)0;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.WidthInBytes = layout.size.x * pixSize;
    a.Height = layout.size.y;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(stream_impl->queue);
    CUresult err;
    err = cuMemcpy2DAsync(&a, stream_impl->queue);
    checkError(err);
  }
}

void ImageCUDA::copy(const ghost::Encoder& s, const ghost::Buffer& src,
                     const BufferLayout& layout, const Origin3& imageOrigin) {
  auto src_impl = static_cast<implementation::BufferCUDA*>(src.impl().get());
  {
    auto sc = static_cast<implementation::StreamCUDA*>(s.impl().get());
    markUsed(sc->queue);
    src_impl->markUsed(sc->queue);
  }
  size_t pixSize = descr.pixelSize();
  size_t srcRowBytes = layout.rowBytes(pixSize);
  size_t srcSliceBytes = layout.sliceBytes(srcRowBytes);
  size_t dstRowBytes = descr.rowBytes(pixSize);
  size_t dstSliceBytes = descr.sliceBytes(dstRowBytes);
  if (descr.size.z > 1) {
    CUDA_MEMCPY3D a;
    a.srcXInBytes = 0;
    a.srcY = 0;
    a.srcZ = 0;
    a.srcLOD = 0;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = src_impl->mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.srcHeight = srcSliceBytes / srcRowBytes;
    a.dstXInBytes = imageOrigin.x * pixSize;
    a.dstY = imageOrigin.y;
    a.dstZ = imageOrigin.z;
    a.dstLOD = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.dstHeight = dstSliceBytes / dstRowBytes;
    a.WidthInBytes = layout.size.x * pixSize;
    a.Height = layout.size.y;
    a.Depth = layout.size.z;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(stream_impl->queue);
    CUresult err;
    err = cuMemcpy3DAsync(&a, stream_impl->queue);
    checkError(err);
  } else {
    CUDA_MEMCPY2D a;
    a.srcXInBytes = 0;
    a.srcY = 0;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = src_impl->mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.dstXInBytes = imageOrigin.x * pixSize;
    a.dstY = imageOrigin.y;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.WidthInBytes = layout.size.x * pixSize;
    a.Height = layout.size.y;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(stream_impl->queue);
    CUresult err;
    err = cuMemcpy2DAsync(&a, stream_impl->queue);
    checkError(err);
  }
}

void ImageCUDA::copyTo(const ghost::Encoder& s, ghost::Buffer& dst,
                       const BufferLayout& layout,
                       const Origin3& imageOrigin) const {
  auto dst_impl = static_cast<implementation::BufferCUDA*>(dst.impl().get());
  {
    auto sc = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(sc->queue);
    dst_impl->markUsed(sc->queue);
  }
  size_t pixSize = descr.pixelSize();
  size_t srcRowBytes = descr.rowBytes(pixSize);
  size_t srcSliceBytes = descr.sliceBytes(srcRowBytes);
  size_t dstRowBytes = layout.rowBytes(pixSize);
  size_t dstSliceBytes = layout.sliceBytes(dstRowBytes);
  if (descr.size.z > 1) {
    CUDA_MEMCPY3D a;
    a.srcXInBytes = imageOrigin.x * pixSize;
    a.srcY = imageOrigin.y;
    a.srcZ = imageOrigin.z;
    a.srcLOD = 0;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.srcHeight = srcSliceBytes / srcRowBytes;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstZ = 0;
    a.dstLOD = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = dst_impl->mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.dstHeight = dstSliceBytes / dstRowBytes;
    a.WidthInBytes = layout.size.x * pixSize;
    a.Height = layout.size.y;
    a.Depth = layout.size.z;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(stream_impl->queue);
    CUresult err;
    err = cuMemcpy3DAsync(&a, stream_impl->queue);
    checkError(err);
  } else {
    CUDA_MEMCPY2D a;
    a.srcXInBytes = imageOrigin.x * pixSize;
    a.srcY = imageOrigin.y;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = dst_impl->mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.WidthInBytes = layout.size.x * pixSize;
    a.Height = layout.size.y;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(stream_impl->queue);
    CUresult err;
    err = cuMemcpy2DAsync(&a, stream_impl->queue);
    checkError(err);
  }
}

void ImageCUDA::copy(const ghost::Encoder& s, const ghost::Image& src,
                     const Size3& region, const Origin3& srcOrigin,
                     const Origin3& dstOrigin) {
  auto src_impl = static_cast<implementation::ImageCUDA*>(src.impl().get());
  {
    auto sc = static_cast<implementation::StreamCUDA*>(s.impl().get());
    markUsed(sc->queue);
    src_impl->markUsed(sc->queue);
  }
  size_t srcPixSize = src_impl->descr.pixelSize();
  size_t srcRowBytes = src_impl->descr.rowBytes(srcPixSize);
  size_t srcSliceBytes = src_impl->descr.sliceBytes(srcRowBytes);
  size_t dstPixSize = descr.pixelSize();
  size_t dstRowBytes = descr.rowBytes(dstPixSize);
  size_t dstSliceBytes = descr.sliceBytes(dstRowBytes);
  if (region.z > 1) {
    CUDA_MEMCPY3D a;
    a.srcXInBytes = srcOrigin.x * srcPixSize;
    a.srcY = srcOrigin.y;
    a.srcZ = srcOrigin.z;
    a.srcLOD = 0;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = src_impl->mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.srcHeight = srcSliceBytes / srcRowBytes;
    a.dstXInBytes = dstOrigin.x * dstPixSize;
    a.dstY = dstOrigin.y;
    a.dstZ = dstOrigin.z;
    a.dstLOD = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.dstHeight = dstSliceBytes / dstRowBytes;
    a.WidthInBytes = region.x * dstPixSize;
    a.Height = region.y;
    a.Depth = region.z;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(stream_impl->queue);
    CUresult err;
    err = cuMemcpy3DAsync(&a, stream_impl->queue);
    checkError(err);
  } else {
    CUDA_MEMCPY2D a;
    a.srcXInBytes = srcOrigin.x * srcPixSize;
    a.srcY = srcOrigin.y;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = src_impl->mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.dstXInBytes = dstOrigin.x * dstPixSize;
    a.dstY = dstOrigin.y;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.WidthInBytes = region.x * dstPixSize;
    a.Height = region.y;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(stream_impl->queue);
    CUresult err;
    err = cuMemcpy2DAsync(&a, stream_impl->queue);
    checkError(err);
  }
}

thread_local size_t CU_CurrentContext::_pushCount;

CUcontext CU_CurrentContext::get() {
  CUcontext ctx;
  if (cuCtxGetCurrent(&ctx) == CUDA_SUCCESS) return ctx;
  return nullptr;
}

CUresult CU_CurrentContext::set(CUcontext c) {
  CUresult err;
  CUcontext ctx;
  err = cuCtxGetCurrent(&ctx);
  if (err != CUDA_SUCCESS) {
    return err;
  }
  if (c == ctx) {
    return err;
  }
  if (ctx != nullptr) {
    (void)cuCtxSynchronize();
  }
  size_t count = _pushCount;
  if (count > 0) {
    err = cuCtxSetCurrent(c);
  } else {
    err = cuCtxPushCurrent(c);
    if (err == CUDA_SUCCESS) {
      count++;
      _pushCount = count;
    }
  }
  return err;
}

void CU_CurrentContext::pushed() { _pushCount++; }

void CU_CurrentContext::pop() {
  CUresult err;
  CUcontext ctx;
  size_t count = _pushCount;
  if (count > 0) _pushCount = 0;
  while (count > 0) {
    err = cuCtxGetCurrent(&ctx);
    if (err == CUDA_SUCCESS && ctx != nullptr) {
      (void)cuCtxSynchronize();
    }
    cuCtxSynchronize();
    cuCtxPopCurrent(&ctx);
    count--;
  }
}

DeviceCUDA::DeviceCUDA(const SharedContext& share) {
  if (!isCudaDriverAvailable()) {
    checkError(CUDA_ERROR_NOT_INITIALIZED);
  }
  CUresult err = cuInit(0);
  checkError(err);
  context =
      cu::ptr<CUcontext>(reinterpret_cast<CUcontext>(share.device), false);
  queue = cu::ptr<CUstream>(reinterpret_cast<CUstream>(share.queue), false);
  if (!context) {
    CU_CurrentContext::pop();  // clear current stack
    device = (CUdevice)0;
#if CUDA_VERSION >= 13000
    CUctxCreateParams ctxCreateParams = {};
    err = cuCtxCreate(&context, &ctxCreateParams, 0, device);
#else
    err = cuCtxCreate(&context, 0, device);
#endif
    checkError(err);
    CU_CurrentContext::pushed();
  } else {
    err = cuCtxGetDevice(&device);
    checkError(err);
  }
  if (!queue) {
    err = cuStreamCreate(&queue, CU_STREAM_NON_BLOCKING);
    checkError(err);
  }
  checkError(cuDeviceGetAttribute(&computeCapability.major,
                                  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                  device));
  checkError(cuDeviceGetAttribute(&computeCapability.minor,
                                  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                  device));
}

DeviceCUDA::DeviceCUDA(const GpuInfo& info) : DeviceCUDA(info.index) {}

DeviceCUDA::DeviceCUDA(int deviceOrdinal) {
  if (!isCudaDriverAvailable()) {
    checkError(CUDA_ERROR_NOT_INITIALIZED);
  }
  CUresult err;
  err = cuInit(0);
  checkError(err);
  err = cuDeviceGet(&device, deviceOrdinal);
  checkError(err);
  CU_CurrentContext::pop();  // clear current stack
#if CUDA_VERSION >= 13000
  CUctxCreateParams ctxCreateParams = {};
  err = cuCtxCreate(&context, &ctxCreateParams, 0, device);
#else
  err = cuCtxCreate(&context, 0, device);
#endif
  checkError(err);
  CU_CurrentContext::pushed();
  err = cuStreamCreate(&queue, CU_STREAM_NON_BLOCKING);
  checkError(err);
  checkError(cuDeviceGetAttribute(&computeCapability.major,
                                  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                  device));
  checkError(cuDeviceGetAttribute(&computeCapability.minor,
                                  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                  device));
}

DeviceCUDA::~DeviceCUDA() {
  try {
    // Need to clear context before we can destroy it.
    if (context.get() && CU_CurrentContext::get() == context.get())
      CU_CurrentContext::pop();
  } catch (...) {
  }
}

void DeviceCUDA::activate(void** prevOut) {
  CUcontext prev = CU_CurrentContext::get();
  if (prevOut) *prevOut = reinterpret_cast<void*>(prev);
  checkError(CU_CurrentContext::set(context.get()));
}

void DeviceCUDA::deactivate(void* prev) {
  // Surface any prior async error to the caller instead of losing it.
  CUresult syncErr = cuCtxSynchronize();
  CUcontext prevCtx = reinterpret_cast<CUcontext>(prev);
  if (prevCtx != nullptr && prevCtx != context.get()) {
    (void)CU_CurrentContext::set(prevCtx);
  }
  if (syncErr != CUDA_SUCCESS) checkError(syncErr);
}

ghost::Library DeviceCUDA::loadLibraryFromText(const std::string& text,
                                               const CompilerOptions& options,
                                               bool retainBinary) const {
  auto ptr = std::make_shared<implementation::LibraryCUDA>(*this, retainBinary);
  ptr->loadFromText(text, options);
  return ghost::Library(ptr);
}

ghost::Library DeviceCUDA::loadLibraryFromData(const void* data, size_t len,
                                               const CompilerOptions& options,
                                               bool retainBinary) const {
  auto ptr = std::make_shared<implementation::LibraryCUDA>(*this, retainBinary);
  ptr->loadFromData(data, len, options);
  return ghost::Library(ptr);
}

SharedContext DeviceCUDA::shareContext() const {
  SharedContext c(context.get(), queue.get());
  return c;
}

ghost::Stream DeviceCUDA::createStream(const StreamOptions& options) const {
  auto ptr = std::make_shared<implementation::StreamCUDA>(context.get());
  return ghost::Stream(ptr);
}

size_t DeviceCUDA::getMemoryPoolSize() const {
  return Device::getMemoryPoolSize();
}

void DeviceCUDA::setMemoryPoolSize(size_t bytes) {
  Device::setMemoryPoolSize(bytes);
#if CUDA_VERSION >= 11020
  memPool.reset();
  if (bytes > 0) {
    CUmemPoolProps poolProps = {};
    poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    poolProps.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
    poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    poolProps.location.id = device;
    CUresult err = cuMemPoolCreate(&memPool, &poolProps);
    if (err == CUDA_SUCCESS) {
      cuuint64_t maxBytes = static_cast<cuuint64_t>(bytes);
      cuMemPoolSetAttribute(memPool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                            &maxBytes);
    } else {
      memPool.reset();
    }
  }
#endif
}

ghost::Buffer DeviceCUDA::allocateBuffer(size_t bytes,
                                         const BufferOptions& opts) const {
  // AllocHint::Staging routes to pinned host memory (mapped buffer path).
  if (opts.hint == AllocHint::Staging) {
    auto ptr =
        std::make_shared<implementation::MappedBufferCUDA>(*this, bytes, opts);
    return ghost::Buffer(ptr);
  }
#if CUDA_VERSION >= 11020
  // Use the memory pool for Default/Transient allocations. Persistent
  // allocations bypass the pool to avoid fragmenting long-lived resources.
  if (memPool && opts.hint != AllocHint::Persistent) {
    CUdeviceptr devPtr;
    CUresult err = cuMemAllocFromPoolAsync(&devPtr, bytes, memPool, queue);
    if (err == CUDA_SUCCESS) {
      // Sync to ensure the allocation is complete before use
      cuStreamSynchronize(queue);
      auto ptr = std::make_shared<implementation::BufferCUDA>(
          cu::ptr<CUdeviceptr>(devPtr, false), bytes);
      return ghost::Buffer(ptr);
    }
    // Pool allocation failed — fall through to allocator / standard
  }
#endif
  if (auto* a = allocator()) {
    if (void* handle = a->allocateBuffer(bytes, opts)) {
      CUdeviceptr devPtr =
          static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(handle));
      auto ptr = std::make_shared<implementation::BufferCUDA>(
          cu::ptr<CUdeviceptr>(devPtr, /*retainObject=*/true), bytes);
      ptr->setAllocator(a);
      return ghost::Buffer(ptr);
    }
  }
  auto ptr = std::make_shared<implementation::BufferCUDA>(*this, bytes, opts);
  return ghost::Buffer(ptr);
}

ghost::MappedBuffer DeviceCUDA::allocateMappedBuffer(
    size_t bytes, const BufferOptions& opts) const {
  if (auto* a = allocator()) {
    if (void* handle = a->allocateMappedBuffer(bytes, opts)) {
      // For CUDA mapped buffers the host returns the host pointer; Ghost
      // derives the device pointer via cuMemHostGetDevicePointer.
      auto ptr = std::make_shared<implementation::MappedBufferCUDA>(
          cu::ptr<void*>(handle, /*retainObject=*/true));
      ptr->_size = bytes;
      ptr->setAllocator(a);
      return ghost::MappedBuffer(ptr);
    }
  }
  auto ptr =
      std::make_shared<implementation::MappedBufferCUDA>(*this, bytes, opts);
  return ghost::MappedBuffer(ptr);
}

ghost::Image DeviceCUDA::allocateImage(const ImageDescription& descr) const {
  if (auto* a = allocator()) {
    if (void* handle = a->allocateImage(descr)) {
      CUdeviceptr devPtr =
          static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(handle));
      auto ptr = std::make_shared<implementation::ImageCUDA>(
          cu::ptr<CUdeviceptr>(devPtr, /*retainObject=*/true), descr);
      ptr->setAllocator(a);
      return ghost::Image(ptr);
    }
  }
  auto ptr = std::make_shared<implementation::ImageCUDA>(*this, descr);
  return ghost::Image(ptr);
}

ghost::Image DeviceCUDA::sharedImage(const ImageDescription& descr,
                                     ghost::Buffer& buffer) const {
  auto b = static_cast<implementation::BufferCUDA*>(buffer.impl().get());
  auto ptr = std::make_shared<implementation::ImageCUDA>(*this, descr, *b);
  return ghost::Image(ptr);
}

ghost::Image DeviceCUDA::sharedImage(const ImageDescription& descr,
                                     ghost::Image& image) const {
  auto i = static_cast<implementation::ImageCUDA*>(image.impl().get());
  auto ptr = std::make_shared<implementation::ImageCUDA>(*this, descr, *i);
  return ghost::Image(ptr);
}

ghost::Buffer DeviceCUDA::wrapBuffer(const SharedBuffer& shared) const {
  CUdeviceptr devPtr =
      static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(shared.handle));
  // owned=false: cu::ptr stores the value but never calls cuMemFree.
  auto ptr = std::make_shared<implementation::BufferCUDA>(
      cu::ptr<CUdeviceptr>(devPtr, /*retainObject=*/false), shared.bytes);
  return ghost::Buffer(ptr);
}

ghost::Image DeviceCUDA::wrapImage(const SharedImage& shared) const {
  CUdeviceptr devPtr =
      static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(shared.handle));
  auto ptr = std::make_shared<implementation::ImageCUDA>(
      cu::ptr<CUdeviceptr>(devPtr, /*retainObject=*/false), shared.descr);
  return ghost::Image(ptr);
}

Attribute DeviceCUDA::getAttribute(DeviceAttributeId what) const {
  switch (what) {
    case kDeviceImplementation:
      return "CUDA";
    case kDeviceName: {
      char buf[128];
      checkError(cuDeviceGetName(buf, sizeof(buf), device));
      return buf;
    }
    case kDeviceVendor:
      return "NVIDIA";
    case kDeviceDriverVersion: {
      int version = 0;
      checkError(cuDriverGetVersion(&version));
      std::stringstream stream;
      stream << version;
      return stream.str();
    }
    case kDeviceFamily: {
      std::stringstream stream;
      stream << computeCapability.major << "." << computeCapability.minor;
      return stream.str();
    }
    case kDeviceCount:
      return 1;
    case kDeviceProcessorCount: {
      int multiProcessorCount;
      checkError(cuDeviceGetAttribute(&multiProcessorCount,
                                      CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                      device));
      return multiProcessorCount;
    }
    case kDeviceUnifiedMemory: {
      int v;
      checkError(
          cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_INTEGRATED, device));
      return v != 0;
    }
    case kDeviceMemory: {
      size_t v;
      checkError(cuDeviceTotalMem(&v, device));
      return v;
    }
    case kDeviceLocalMemory: {
      int v;
      checkError(cuDeviceGetAttribute(
          &v, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device));
      return v;
    }
    case kDeviceMaxThreads: {
      int v;
      checkError(cuDeviceGetAttribute(
          &v, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
      return v;
    }
    case kDeviceMaxWorkSize: {
      int x, y, z;
      checkError(cuDeviceGetAttribute(&x, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                                      device));
      checkError(cuDeviceGetAttribute(&y, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
                                      device));
      checkError(cuDeviceGetAttribute(&z, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
                                      device));
      return Attribute(x, y, z);
    }
    case kDeviceMaxRegisters: {
      int v;
      checkError(cuDeviceGetAttribute(
          &v, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device));
      return v;
    }
    case kDeviceMaxImageSize1: {
      int x, y, z;
      checkError(cuDeviceGetAttribute(
          &x, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, device));
      return x;
    }
    case kDeviceMaxImageSize2: {
      int x, y;
      checkError(cuDeviceGetAttribute(
          &x, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, device));
      checkError(cuDeviceGetAttribute(
          &y, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, device));
      return Attribute(x, y);
    }
    case kDeviceMaxImageSize3: {
      int x, y, z;
      checkError(cuDeviceGetAttribute(
          &x, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, device));
      checkError(cuDeviceGetAttribute(
          &y, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, device));
      checkError(cuDeviceGetAttribute(
          &z, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, device));
      return Attribute(x, y, z);
    }
    case kDeviceMaxImageAlignment: {
      int v;
      checkError(cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
                                      device));
      return v;
    }
    case kDeviceSupportsImageIntegerFiltering:
      return true;
    case kDeviceSupportsImageFloatFiltering:
      return true;
    case kDeviceSupportsMappedBuffer: {
      int canMap;
      checkError(cuDeviceGetAttribute(
          &canMap, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, device));
      return canMap != 0;
    }
    case kDeviceSupportsProgramConstants:
      return false;
    case kDeviceSupportsProgramGlobals:
      return true;
    case kDeviceSupportsSubgroup:
      return true;
    case kDeviceSupportsSubgroupShuffle:
      return true;
    case kDeviceSubgroupWidth: {
      int v;
      checkError(
          cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device));
      return v;
    }
    case kDeviceMaxComputeUnits: {
      int v;
      checkError(cuDeviceGetAttribute(
          &v, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
      return v;
    }
    case kDeviceMemoryAlignment: {
      int v;
      checkError(cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
                                      device));
      return v;
    }
    case kDeviceBufferAlignment:
      return 256;
    case kDeviceMaxBufferSize: {
      size_t v;
      checkError(cuDeviceTotalMem(&v, device));
      return (uint64_t)v;
    }
    case kDeviceMaxConstantBufferSize: {
      int v;
      checkError(cuDeviceGetAttribute(
          &v, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, device));
      return v;
    }
    case kDeviceTimestampPeriod:
      return 1000.0f;
    case kDeviceSupportsProfilingTimer:
      return true;
    case kDeviceSupportsCooperativeMatrix:
      // WMMA requires compute capability >= 7.0 (Volta+)
      return computeCapability.major >= 7;
    default:
      return Attribute();
  }
}
}  // namespace implementation

DeviceCUDA::DeviceCUDA(const SharedContext& share)
    : Device(std::make_shared<implementation::DeviceCUDA>(share)) {
  auto cuda = static_cast<implementation::DeviceCUDA*>(impl().get());
  setDefaultStream(std::make_shared<implementation::StreamCUDA>(cuda->queue));
}

DeviceCUDA::DeviceCUDA(const GpuInfo& info)
    : Device(std::make_shared<implementation::DeviceCUDA>(info)) {
  auto cuda = static_cast<implementation::DeviceCUDA*>(impl().get());
  setDefaultStream(std::make_shared<implementation::StreamCUDA>(cuda->queue));
}

std::vector<GpuInfo> DeviceCUDA::enumerateDevices() {
  std::vector<GpuInfo> result;
  if (!isCudaDriverAvailable()) return result;
  CUresult err = cuInit(0);
  if (err != CUDA_SUCCESS) return result;

  int count = 0;
  err = cuDeviceGetCount(&count);
  if (err != CUDA_SUCCESS) return result;

  for (int i = 0; i < count; i++) {
    CUdevice dev;
    err = cuDeviceGet(&dev, i);
    if (err != CUDA_SUCCESS) continue;

    GpuInfo info;

    char name[256];
    if (cuDeviceGetName(name, sizeof(name), dev) == CUDA_SUCCESS)
      info.name = name;

    info.vendor = "NVIDIA";
    info.implementation = "CUDA";

    size_t totalMem;
    if (cuDeviceTotalMem(&totalMem, dev) == CUDA_SUCCESS)
      info.memory = totalMem;

    int unified = 0;
    if (cuDeviceGetAttribute(&unified, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
                             dev) == CUDA_SUCCESS)
      info.unifiedMemory = unified != 0;

    info.index = i;
    result.push_back(info);
  }
  return result;
}
}  // namespace ghost
#endif
