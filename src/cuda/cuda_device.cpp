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

#include <sstream>

namespace ghost {
namespace implementation {
using namespace cu;

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

BufferCUDA::BufferCUDA(cu::ptr<CUdeviceptr> mem_) : mem(mem_) {}

BufferCUDA::BufferCUDA(const DeviceCUDA& dev, size_t bytes, Access) {
  CUresult err;
  err = cuMemAlloc(&mem, bytes);
  checkError(err);
}

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

MappedBufferCUDA::MappedBufferCUDA(cu::ptr<void*> ptr_)
    : BufferCUDA(cu::ptr<CUdeviceptr>()), ptr(ptr_) {
  CUdeviceptr p;
  CUresult err;
  err = cuMemHostGetDevicePointer(&p, ptr, 0);
  checkError(err);
  mem = cu::ptr<CUdeviceptr>(p, false);  // do not free the device pointer
}

MappedBufferCUDA::MappedBufferCUDA(const DeviceCUDA& dev, size_t bytes,
                                   Access access)
    : BufferCUDA(cu::ptr<CUdeviceptr>()) {
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

size_t DeviceCUDA::getMemoryPoolSize() const {}

void DeviceCUDA::setMemoryPoolSize(size_t bytes) {}

ghost::Buffer DeviceCUDA::allocateBuffer(size_t bytes, Access access) const {
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
      return Attribute("CUDA");
    case kDeviceName: {
      char buf[128];
      checkError(cuDeviceGetName(buf, sizeof(buf), device));
      return Attribute(buf);
    }
    case kDeviceVendor:
      return Attribute("NVIDIA");
    case kDeviceDriverVersion: {
      int version = 0;
      checkError(cuDriverGetVersion(&version));
      std::stringstream stream;
      stream << version;
      return Attribute(stream.str());
    }
    case kDeviceCount: {
      int multiProcessorCount;
      checkError(cuDeviceGetAttribute(&multiProcessorCount,
                                      CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                      device));
      return Attribute(multiProcessorCount);
    }
    case kDeviceSupportsMappedBuffer: {
      int canMap;
      checkError(cuDeviceGetAttribute(
          &canMap, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, device));
      return Attribute(canMap != 0);
    }
    case kDeviceSupportsProgramConstants:
      return Attribute(false);
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
}  // namespace ghost
#endif
