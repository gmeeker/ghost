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

#include <ghost/cuda/device.h>
#include <ghost/cuda/exception.h>
#include <ghost/cuda/impl_device.h>
#include <ghost/cuda/impl_function.h>

#include <cstring>
#include <sstream>
#include <vector>

namespace ghost {
namespace implementation {
using namespace cu;

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

BufferCUDA::BufferCUDA(const DeviceCUDA& dev, size_t bytes, Access)
    : _size(bytes) {
  CUresult err;
  err = cuMemAlloc(&mem, bytes);
  checkError(err);
}

size_t BufferCUDA::size() const { return _size; }

void BufferCUDA::copy(const ghost::Stream& s, const ghost::Buffer& src,
                      size_t bytes) {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  auto src_impl = static_cast<implementation::BufferCUDA*>(src.impl().get());
  CUresult err;
  err = cuMemcpyDtoDAsync(mem, src_impl->mem, bytes, stream_impl->queue);
  checkError(err);
}

void BufferCUDA::copy(const ghost::Stream& s, const void* src, size_t bytes) {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  CUresult err;
  err = cuMemcpyHtoDAsync(mem, src, bytes, stream_impl->queue);
  checkError(err);
}

void BufferCUDA::copyTo(const ghost::Stream& s, void* dst, size_t bytes) const {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  CUresult err;
  err = cuMemcpyDtoHAsync(dst, mem, bytes, stream_impl->queue);
  checkError(err);
}

void BufferCUDA::copy(const ghost::Stream& s, const ghost::Buffer& src,
                      size_t srcOffset, size_t dstOffset, size_t bytes) {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  auto src_impl = static_cast<implementation::BufferCUDA*>(src.impl().get());
  CUresult err;
  err =
      cuMemcpyDtoDAsync(mem.get() + dstOffset, src_impl->mem.get() + srcOffset,
                        bytes, stream_impl->queue);
  checkError(err);
}

void BufferCUDA::copy(const ghost::Stream& s, const void* src, size_t dstOffset,
                      size_t bytes) {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  CUresult err;
  err =
      cuMemcpyHtoDAsync(mem.get() + dstOffset, src, bytes, stream_impl->queue);
  checkError(err);
}

void BufferCUDA::copyTo(const ghost::Stream& s, void* dst, size_t srcOffset,
                        size_t bytes) const {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  CUresult err;
  err =
      cuMemcpyDtoHAsync(dst, mem.get() + srcOffset, bytes, stream_impl->queue);
  checkError(err);
}

void BufferCUDA::fill(const ghost::Stream& s, size_t offset, size_t size,
                      uint8_t value) {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  CUresult err;
  err = cuMemsetD8Async(mem.get() + offset, value, size, stream_impl->queue);
  checkError(err);
}

void BufferCUDA::fill(const ghost::Stream& s, size_t offset, size_t size,
                      const void* pattern, size_t patternSize) {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
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

MappedBufferCUDA::MappedBufferCUDA(cu::ptr<void*> ptr_)
    : BufferCUDA(cu::ptr<CUdeviceptr>(), 0), ptr(ptr_) {
  CUdeviceptr p;
  CUresult err;
  err = cuMemHostGetDevicePointer(&p, ptr, 0);
  checkError(err);
  mem = cu::ptr<CUdeviceptr>(p, false);  // do not free the device pointer
}

MappedBufferCUDA::MappedBufferCUDA(const DeviceCUDA& dev, size_t bytes,
                                   Access access)
    : BufferCUDA(cu::ptr<CUdeviceptr>(), bytes) {
  unsigned int flags = CU_MEMHOSTALLOC_DEVICEMAP;
  if (access == Access_WriteOnly) {
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

void* MappedBufferCUDA::map(const ghost::Stream& s, Access access, bool sync) {
  if (sync) {
    // TODO
  }
  return ptr;
}

void MappedBufferCUDA::unmap(const ghost::Stream&) {}

ImageCUDA::ImageCUDA(cu::ptr<CUdeviceptr> mem_, const ImageDescription& descr_)
    : mem(mem_), descr(descr_) {}

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

void ImageCUDA::copy(const ghost::Stream& s, const ghost::Image& src) {
  auto src_impl = static_cast<implementation::ImageCUDA*>(src.impl().get());
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
    a.srcPitch = src_impl->descr.stride.x;
    a.srcHeight = src_impl->descr.stride.y;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstZ = 0;
    a.dstLOD = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = descr.stride.x;
    a.dstHeight = descr.stride.y;
    a.WidthInBytes = descr.size.x * descr.pixelSize();
    a.Height = descr.size.y;
    a.Depth = descr.size.z;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
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
    a.srcPitch = src_impl->descr.stride.x;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = descr.stride.x;
    a.WidthInBytes = descr.size.x * descr.pixelSize();
    a.Height = descr.size.y;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    CUresult err;
    err = cuMemcpy2DAsync(&a, stream_impl->queue);
    checkError(err);
  }
}

void ImageCUDA::copy(const ghost::Stream& s, const ghost::Buffer& src,
                     const ImageDescription& d) {
  auto src_impl = static_cast<implementation::BufferCUDA*>(src.impl().get());
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
    a.srcPitch = d.stride.x;
    a.srcHeight = d.stride.y;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstZ = 0;
    a.dstLOD = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = descr.stride.x;
    a.dstHeight = descr.stride.y;
    a.WidthInBytes = descr.size.x * descr.pixelSize();
    a.Height = descr.size.y;
    a.Depth = descr.size.z;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
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
    a.srcPitch = descr.stride.x;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = descr.stride.x;
    a.WidthInBytes = descr.size.x * descr.pixelSize();
    a.Height = descr.size.y;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    CUresult err;
    err = cuMemcpy2DAsync(&a, stream_impl->queue);
    checkError(err);
  }
}

void ImageCUDA::copy(const ghost::Stream& s, const void* src,
                     const ImageDescription& d) {
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
    a.srcPitch = d.stride.x;
    a.srcHeight = d.stride.y;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstZ = 0;
    a.dstLOD = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = descr.stride.x;
    a.dstHeight = descr.stride.y;
    a.WidthInBytes = descr.size.x * descr.pixelSize();
    a.Height = descr.size.y;
    a.Depth = descr.size.z;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
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
    a.srcPitch = d.stride.x;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = descr.stride.x;
    a.WidthInBytes = descr.size.x * descr.pixelSize();
    a.Height = descr.size.y;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    CUresult err;
    err = cuMemcpy2DAsync(&a, stream_impl->queue);
    checkError(err);
  }
}

void ImageCUDA::copyTo(const ghost::Stream& s, ghost::Buffer& dst,
                       const ImageDescription& d) const {
  auto dst_impl = static_cast<implementation::BufferCUDA*>(dst.impl().get());
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
    a.srcPitch = descr.stride.x;
    a.srcHeight = descr.stride.y;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstZ = 0;
    a.dstLOD = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = dst_impl->mem;
    a.dstArray = nullptr;
    a.dstPitch = d.stride.x;
    a.dstHeight = d.stride.y;
    a.WidthInBytes = d.size.x * d.pixelSize();
    a.Height = d.size.y;
    a.Depth = d.size.z;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
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
    a.srcPitch = descr.stride.x;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = dst_impl->mem;
    a.dstArray = nullptr;
    a.dstPitch = d.stride.x;
    a.WidthInBytes = d.size.x * d.pixelSize();
    a.Height = d.size.y;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    CUresult err;
    err = cuMemcpy2DAsync(&a, stream_impl->queue);
    checkError(err);
  }
}

void ImageCUDA::copyTo(const ghost::Stream& s, void* dst,
                       const ImageDescription& d) const {
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
    a.srcPitch = descr.stride.x;
    a.srcHeight = descr.stride.y;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstZ = 0;
    a.dstLOD = 0;
    a.dstMemoryType = CU_MEMORYTYPE_HOST;
    a.dstHost = dst;
    a.dstDevice = (CUdeviceptr)0;
    a.dstArray = nullptr;
    a.dstPitch = d.stride.x;
    a.dstHeight = d.stride.y;
    a.WidthInBytes = d.size.x * d.pixelSize();
    a.Height = d.size.y;
    a.Depth = d.size.z;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
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
    a.srcPitch = descr.stride.x;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = dst;
    a.dstDevice = (CUdeviceptr)0;
    a.dstArray = nullptr;
    a.dstPitch = d.stride.x;
    a.WidthInBytes = d.size.x * d.pixelSize();
    a.Height = d.size.y;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    CUresult err;
    err = cuMemcpy2DAsync(&a, stream_impl->queue);
    checkError(err);
  }
}

DeviceCUDA::DeviceCUDA(const SharedContext& share) {
  context =
      cu::ptr<CUcontext>(reinterpret_cast<CUcontext>(share.device), false);
  queue = cu::ptr<CUstream>(reinterpret_cast<CUstream>(share.queue), false);
  CUresult err;
  if (!context) {
    device = (CUdevice)0;
#if CUDA_VERSION >= 13000
    CUctxCreateParams ctxCreateParams = {};
    err = cuCtxCreate(&context, &ctxCreateParams, 0, device);
#else
    err = cuCtxCreate(&context, 0, device);
#endif
    checkError(err);
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
  CUresult err;
  err = cuDeviceGet(&device, deviceOrdinal);
  checkError(err);
#if CUDA_VERSION >= 13000
  CUctxCreateParams ctxCreateParams = {};
  err = cuCtxCreate(&context, &ctxCreateParams, 0, device);
#else
  err = cuCtxCreate(&context, 0, device);
#endif
  checkError(err);
  err = cuStreamCreate(&queue, CU_STREAM_NON_BLOCKING);
  checkError(err);
  checkError(cuDeviceGetAttribute(&computeCapability.major,
                                  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                  device));
  checkError(cuDeviceGetAttribute(&computeCapability.minor,
                                  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                  device));
}

ghost::Library DeviceCUDA::loadLibraryFromText(
    const std::string& text, const std::string& options) const {
  auto ptr = std::make_shared<implementation::LibraryCUDA>(*this);
  ptr->loadFromText(text, options);
  return ghost::Library(ptr);
}

ghost::Library DeviceCUDA::loadLibraryFromData(
    const void* data, size_t len, const std::string& options) const {
  auto ptr = std::make_shared<implementation::LibraryCUDA>(*this);
  ptr->loadFromData(data, len, options);
  return ghost::Library(ptr);
}

SharedContext DeviceCUDA::shareContext() const {
  SharedContext c(context.get(), queue.get());
  return c;
}

ghost::Stream DeviceCUDA::createStream() const {
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

ghost::Buffer DeviceCUDA::allocateBuffer(size_t bytes, Access access) const {
#if CUDA_VERSION >= 11020
  if (memPool) {
    CUdeviceptr devPtr;
    CUresult err = cuMemAllocFromPoolAsync(&devPtr, bytes, memPool, queue);
    if (err == CUDA_SUCCESS) {
      // Sync to ensure the allocation is complete before use
      cuStreamSynchronize(queue);
      auto ptr = std::make_shared<implementation::BufferCUDA>(
          cu::ptr<CUdeviceptr>(devPtr, false), bytes);
      return ghost::Buffer(ptr);
    }
    // Pool allocation failed — fall through to standard allocation
  }
#endif
  auto ptr = std::make_shared<implementation::BufferCUDA>(*this, bytes, access);
  return ghost::Buffer(ptr);
}

ghost::MappedBuffer DeviceCUDA::allocateMappedBuffer(size_t bytes,
                                                     Access access) const {
  auto ptr =
      std::make_shared<implementation::MappedBufferCUDA>(*this, bytes, access);
  return ghost::MappedBuffer(ptr);
}

ghost::Image DeviceCUDA::allocateImage(const ImageDescription& descr) const {
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
      return v;
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
    case kDeviceImageAlignment: {
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
