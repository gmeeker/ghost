// Copyright (c) 2025 Digital Anarchy, Inc. All rights reserve
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

#if WITH_OPENCL

#include <ghost/opencl/device.h>
#include <ghost/opencl/exception.h>
#include <ghost/opencl/impl_device.h>
#include <ghost/opencl/impl_function.h>
#include <string.h>

#ifndef __APPLE_CC__
#include <CL/cl_ext.h>
#endif

namespace ghost {
namespace implementation {
using namespace opencl;

namespace {
void split(std::vector<std::string>& strs, const std::string& str,
           const std::string& delims = " ") {
  size_t index, last;
  last = 0;
  index = str.find_first_of(delims);
  while (index != std::string::npos) {
    if (index > last) {
      strs.push_back(str.substr(last, index - last));
    }
    last = index + 1;
    index = str.find_first_of(delims, last);
  }
  std::string s = str.substr(last);
  if (!s.empty()) {
    strs.push_back(s);
  }
}

void set_of(std::set<std::string>& strs, const std::string& str,
            const std::string& delims = " ") {
  size_t index, last;
  last = 0;
  index = str.find_first_of(delims);
  while (index != std::string::npos) {
    if (index > last) {
      strs.insert(str.substr(last, index - last));
    }
    last = index + 1;
    index = str.find_first_of(delims, last);
  }
  std::string s = str.substr(last);
  if (!s.empty()) {
    strs.insert(s);
  }
}

cl_mem_flags getMemFlags(Access access) {
  switch (access) {
    case Access_WriteOnly:
      return CL_MEM_WRITE_ONLY;
    case Access_ReadOnly:
      return CL_MEM_READ_ONLY;
    case Access_ReadWrite:
    default:
      return CL_MEM_READ_WRITE;
  }
}

cl_mem_flags getMemFlags(const ImageDescription& descr) {
  return getMemFlags(descr.access);
}

cl_image_format getFormat(cl_context ctx, const ImageDescription& descr,
                          cl_mem_object_type type, cl_mem_flags flags) {
  cl_image_format fmt;
  switch (descr.type) {
    case DataType_Float16:
      fmt.image_channel_data_type = CL_HALF_FLOAT;
      break;
    case DataType_Float:
      fmt.image_channel_data_type = CL_FLOAT;
      break;
    case DataType_Double:
      throw ghost::unsupported_error();
    case DataType_UInt16:
      fmt.image_channel_data_type = CL_UNORM_INT16;
      break;
    case DataType_Int16:
      fmt.image_channel_data_type = CL_SNORM_INT16;
      break;
    case DataType_Int8:
      fmt.image_channel_data_type = CL_SNORM_INT8;
      break;
    case DataType_UInt8:
    default:
      fmt.image_channel_data_type = CL_UNORM_INT8;
      break;
  }
  switch (descr.channels) {
    case 1:
      fmt.image_channel_order = CL_INTENSITY;
      break;
    case 2:
      fmt.image_channel_order = CL_RA;
      break;
    case 3:
      fmt.image_channel_order = CL_RGB;
      break;
    case 4:
    default:
      switch (descr.order) {
        default:
        case PixelOrder_RGBA:
          fmt.image_channel_order = CL_RGBA;
          break;
        case PixelOrder_ARGB:
          fmt.image_channel_order = CL_ARGB;
          break;
        case PixelOrder_ABGR:
#ifdef CL_ABGR_APPLE
          fmt.image_channel_order = CL_ABGR_APPLE;
#endif
          break;
        case PixelOrder_BGRA:
          fmt.image_channel_order = CL_BGRA;
          break;
      }
      break;
  }
  cl_uint num;
  cl_int err;
  err = clGetSupportedImageFormats(ctx, flags, type, 0, nullptr, &num);
  checkError(err);
  std::vector<cl_image_format> formats;
  formats.resize(size_t(num));
  if (!formats.empty()) {
    err =
        clGetSupportedImageFormats(ctx, flags, type, num, &formats[0], nullptr);
    checkError(err);
  }
  bool valid = false;
  for (cl_uint i = 0; i < num; i++) {
    if (formats[i].image_channel_data_type == fmt.image_channel_data_type &&
        formats[i].image_channel_order == fmt.image_channel_order) {
      valid = true;
    }
  }
  if (!valid) throw ghost::unsupported_error();
  return fmt;
}
}  // namespace

EventOpenCL::EventOpenCL(opencl::ptr<cl_event> event_) : event(event_) {}

void EventOpenCL::wait() {
  cl_event ev = event.get();
  cl_int err = clWaitForEvents(1, &ev);
  checkError(err);
}

bool EventOpenCL::isComplete() const {
  cl_int status;
  cl_int err = clGetEventInfo(event.get(), CL_EVENT_COMMAND_EXECUTION_STATUS,
                              sizeof(status), &status, nullptr);
  checkError(err);
  return status == CL_COMPLETE;
}

double EventOpenCL::elapsed(const Event& other) const {
  auto& otherOCL = static_cast<const EventOpenCL&>(other);
  cl_ulong startTime, endTime;
  cl_int err;
  err = clGetEventProfilingInfo(event.get(), CL_PROFILING_COMMAND_END,
                                sizeof(startTime), &startTime, nullptr);
  if (err != CL_SUCCESS) return 0.0;
  err = clGetEventProfilingInfo(otherOCL.event.get(), CL_PROFILING_COMMAND_END,
                                sizeof(endTime), &endTime, nullptr);
  if (err != CL_SUCCESS) return 0.0;
  return static_cast<double>(endTime - startTime) / 1e9;
}

StreamOpenCL::StreamOpenCL(opencl::ptr<cl_command_queue> queue_)
    : queue(queue_), outOfOrder(true) {}

StreamOpenCL::StreamOpenCL(const DeviceOpenCL& dev) : outOfOrder(true) {
  cl_int err;
  bool profiling = false;
  cl_command_queue_properties queueProperties = 0;
  cl_command_queue_properties devQueueProperties = 0;
  auto devices = dev.getDevices();
  if (outOfOrder) queueProperties |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
  if (profiling) queueProperties |= CL_QUEUE_PROFILING_ENABLE;
  err = clGetDeviceInfo(devices[0], CL_DEVICE_QUEUE_PROPERTIES,
                        sizeof(devQueueProperties), &devQueueProperties, NULL);
  checkError(err);
  queueProperties &= devQueueProperties;
  if (queue.get() == nullptr && dev.context.get() != nullptr) {
    if (dev.checkVersion("2.0")) {
#ifdef CL_VERSION_2_0
      cl_queue_properties p[] = {CL_QUEUE_PROPERTIES, queueProperties, 0};
      queue =
          clCreateCommandQueueWithProperties(dev.context, devices[0], p, &err);
      checkError(err);
#endif
    }
    if (!queue) {
      queue =
          clCreateCommandQueue(dev.context, devices[0], queueProperties, &err);
      checkError(err);
    }
  }
}

void StreamOpenCL::sync() {
  cl_int err = CL_SUCCESS;
  if (outOfOrder) {
    if (!events.empty()) {
      err = clWaitForEvents(events.size(), events);
    }
  } else {
    err = clFinish(queue);
  }
  checkError(err);
}

void StreamOpenCL::addEvent() {
  if (outOfOrder) {
    events.reset();
    events.push(lastEvent);
  }
}

cl_event* StreamOpenCL::event() { return outOfOrder ? &lastEvent : nullptr; }

std::shared_ptr<Event> StreamOpenCL::record() {
  cl_event ev;
  cl_int err;
#ifdef CL_VERSION_1_2
  err = clEnqueueMarkerWithWaitList(queue, events.size(), events, &ev);
#else
  err = clEnqueueMarker(queue, &ev);
#endif
  checkError(err);
  addEvent();
  return std::make_shared<EventOpenCL>(opencl::ptr<cl_event>(ev));
}

void StreamOpenCL::waitForEvent(const std::shared_ptr<Event>& e) {
  auto eventOCL = static_cast<EventOpenCL*>(e.get());
  cl_event ev = eventOCL->event.get();
  cl_int err;
#ifdef CL_VERSION_1_2
  err = clEnqueueBarrierWithWaitList(queue, 1, &ev, nullptr);
#else
  err = clEnqueueWaitForEvents(queue, 1, &ev);
#endif
  checkError(err);
}

BufferOpenCL::BufferOpenCL(opencl::ptr<cl_mem> mem_, size_t bytes)
    : mem(mem_), _size(bytes) {}

BufferOpenCL::BufferOpenCL(const DeviceOpenCL& dev, size_t bytes, Access access)
    : _size(bytes) {
  cl_int err;
  cl_mem_flags flags = getMemFlags(access);
  mem = opencl::ptr<cl_mem>(
      clCreateBuffer(dev.context, flags, bytes, nullptr, &err));
  checkError(err);
}

size_t BufferOpenCL::size() const { return _size; }

void BufferOpenCL::copy(const ghost::Stream& s, const ghost::Buffer& src,
                        size_t bytes) {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  auto src_impl = static_cast<implementation::BufferOpenCL*>(src.impl().get());
  cl_int err;
  err = clEnqueueCopyBuffer(stream_impl->queue, src_impl->mem, mem, 0, 0, bytes,
                            stream_impl->events.size(), stream_impl->events,
                            stream_impl->event());
  checkError(err);
  stream_impl->addEvent();
}

void BufferOpenCL::copy(const ghost::Stream& s, const void* src, size_t bytes) {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  cl_int err;
  err = clEnqueueWriteBuffer(stream_impl->queue, mem, false, 0, bytes, src,
                             stream_impl->events.size(), stream_impl->events,
                             stream_impl->event());
  checkError(err);
  stream_impl->addEvent();
}

void BufferOpenCL::copyTo(const ghost::Stream& s, void* dst,
                          size_t bytes) const {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  cl_int err;
  err = clEnqueueReadBuffer(stream_impl->queue, mem, false, 0, bytes, dst,
                            stream_impl->events.size(), stream_impl->events,
                            stream_impl->event());
  checkError(err);
  stream_impl->addEvent();
}

void BufferOpenCL::copy(const ghost::Stream& s, const ghost::Buffer& src,
                        size_t srcOffset, size_t dstOffset, size_t bytes) {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  auto src_impl = static_cast<implementation::BufferOpenCL*>(src.impl().get());
  cl_int err;
  err = clEnqueueCopyBuffer(stream_impl->queue, src_impl->mem, mem, srcOffset,
                            dstOffset, bytes, stream_impl->events.size(),
                            stream_impl->events, stream_impl->event());
  checkError(err);
  stream_impl->addEvent();
}

void BufferOpenCL::copy(const ghost::Stream& s, const void* src,
                        size_t dstOffset, size_t bytes) {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  cl_int err;
  err = clEnqueueWriteBuffer(stream_impl->queue, mem, false, dstOffset, bytes,
                             src, stream_impl->events.size(),
                             stream_impl->events, stream_impl->event());
  checkError(err);
  stream_impl->addEvent();
}

void BufferOpenCL::copyTo(const ghost::Stream& s, void* dst, size_t srcOffset,
                          size_t bytes) const {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  cl_int err;
  err = clEnqueueReadBuffer(stream_impl->queue, mem, false, srcOffset, bytes,
                            dst, stream_impl->events.size(),
                            stream_impl->events, stream_impl->event());
  checkError(err);
  stream_impl->addEvent();
}

void BufferOpenCL::fill(const ghost::Stream& s, size_t offset, size_t size,
                        uint8_t value) {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  cl_int err;
  err = clEnqueueFillBuffer(stream_impl->queue, mem, &value, sizeof(value),
                            offset, size, stream_impl->events.size(),
                            stream_impl->events, stream_impl->event());
  checkError(err);
  stream_impl->addEvent();
}

void BufferOpenCL::fill(const ghost::Stream& s, size_t offset, size_t size,
                        const void* pattern, size_t patternSize) {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  cl_int err;
  err = clEnqueueFillBuffer(stream_impl->queue, mem, pattern, patternSize,
                            offset, size, stream_impl->events.size(),
                            stream_impl->events, stream_impl->event());
  checkError(err);
  stream_impl->addEvent();
}

std::shared_ptr<Buffer> BufferOpenCL::createSubBuffer(
    const std::shared_ptr<Buffer>& self, size_t offset, size_t size) {
  cl_int err;
  cl_buffer_region region;
  region.origin = offset;
  region.size = size;
  cl_mem sub = clCreateSubBuffer(mem, CL_MEM_READ_WRITE,
                                 CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
  checkError(err);
  return std::make_shared<SubBufferOpenCL>(self, opencl::ptr<cl_mem>(sub),
                                           size);
}

SubBufferOpenCL::SubBufferOpenCL(std::shared_ptr<Buffer> parent,
                                 opencl::ptr<cl_mem> mem_, size_t bytes)
    : BufferOpenCL(mem_, bytes), _parent(parent) {}

MappedBufferOpenCL::MappedBufferOpenCL(opencl::ptr<cl_mem> mem_, size_t bytes,
                                       size_t allocSize)
    : BufferOpenCL(mem_, allocSize), length(bytes), ptr(nullptr) {}

MappedBufferOpenCL::MappedBufferOpenCL(const DeviceOpenCL& dev, size_t bytes,
                                       Access access)
    : BufferOpenCL(opencl::ptr<cl_mem>(), bytes), length(bytes), ptr(nullptr) {
  cl_int err;
  cl_mem_flags flags = getMemFlags(access);
  flags |= CL_MEM_ALLOC_HOST_PTR;
  mem = opencl::ptr<cl_mem>(
      clCreateBuffer(dev.context, flags, bytes, nullptr, &err));
  checkError(err);
}

void* MappedBufferOpenCL::map(const ghost::Stream& s, Access access,
                              bool sync) {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  cl_int err;
  cl_map_flags flags;
  switch (access) {
    case Access_ReadOnly:
      flags = CL_MAP_READ;
      break;
    case Access_WriteOnly:
      flags = CL_MAP_WRITE_INVALIDATE_REGION;
      break;
    default:
      throw ghost::unsupported_error();
  }
  ptr = clEnqueueMapBuffer(stream_impl->queue, mem, sync, flags, 0, length,
                           stream_impl->events.size(), stream_impl->events,
                           stream_impl->event(), &err);
  checkError(err);
  stream_impl->addEvent();
  return ptr;
}

void MappedBufferOpenCL::unmap(const ghost::Stream& s) {
  if (ptr) {
    auto stream_impl =
        static_cast<implementation::StreamOpenCL*>(s.impl().get());
    cl_int err;
    err = clEnqueueUnmapMemObject(stream_impl->queue, mem, ptr,
                                  stream_impl->events.size(),
                                  stream_impl->events, stream_impl->event());
    ptr = nullptr;
    checkError(err);
    stream_impl->addEvent();
  }
}

ImageOpenCL::ImageOpenCL(opencl::ptr<cl_mem> mem_,
                         const ImageDescription& descr_)
    : mem(mem_), descr(descr_) {}

ImageOpenCL::ImageOpenCL(const DeviceOpenCL& dev,
                         const ImageDescription& descr_)
    : descr(descr_) {
  cl_int err;
  cl_mem_flags flags = getMemFlags(descr);
  cl_image_desc desc;
  memset(&desc, 0, sizeof(desc));
  if (descr.size.z > 1) {
    desc.image_type = CL_MEM_OBJECT_IMAGE3D;
  } else if (descr.size.y > 1) {
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  } else {
    desc.image_type = CL_MEM_OBJECT_IMAGE1D;
  }
  desc.image_width = descr.size.x;
  desc.image_height = descr.size.y;
  desc.image_depth = descr.size.z;
  desc.image_row_pitch = descr.stride.x;
  desc.image_slice_pitch = descr.stride.y;
  void* host_ptr = nullptr;
  cl_image_format format =
      getFormat(dev.context, descr, desc.image_type, flags);
  mem = opencl::ptr<cl_mem>(
      clCreateImage(dev.context, flags, &format, &desc, host_ptr, &err));
  checkError(err);
}

ImageOpenCL::ImageOpenCL(const DeviceOpenCL& dev,
                         const ImageDescription& descr_, BufferOpenCL& buffer)
    : descr(descr_) {
  cl_int err;
  cl_mem_flags flags = getMemFlags(descr);
  cl_image_desc desc;
  memset(&desc, 0, sizeof(desc));
  if (descr.size.z > 1) {
    desc.image_type = CL_MEM_OBJECT_IMAGE3D;
  } else if (descr.size.y > 1) {
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  } else {
    desc.image_type = CL_MEM_OBJECT_IMAGE1D;
  }
  desc.image_width = descr.size.x;
  desc.image_height = descr.size.y;
  desc.image_depth = descr.size.z;
  desc.image_row_pitch = descr.stride.x;
  desc.image_slice_pitch = descr.stride.y;
  desc.buffer = buffer.mem;
  void* host_ptr = nullptr;
  cl_image_format format =
      getFormat(dev.context, descr, desc.image_type, flags);
  mem = opencl::ptr<cl_mem>(
      clCreateImage(dev.context, flags, &format, &desc, host_ptr, &err));
  checkError(err);
}

ImageOpenCL::ImageOpenCL(const DeviceOpenCL& dev,
                         const ImageDescription& descr_, ImageOpenCL& image)
    : mem(image.mem), descr(descr_) {}

void ImageOpenCL::copy(const ghost::Stream& s, const ghost::Image& src) {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  auto src_impl = static_cast<implementation::ImageOpenCL*>(src.impl().get());
  cl_int err;
  size_t src_origin[] = {0, 0, 0};
  size_t dst_origin[] = {0, 0, 0};
  size_t region[] = {descr.size.x, descr.size.y, descr.size.z};
  err = clEnqueueCopyImage(stream_impl->queue, src_impl->mem, mem, src_origin,
                           dst_origin, region, stream_impl->events.size(),
                           stream_impl->events, stream_impl->event());
  checkError(err);
  stream_impl->addEvent();
}

void ImageOpenCL::copy(const ghost::Stream& s, const ghost::Buffer& src,
                       const ImageDescription& descr) {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  auto src_impl = static_cast<implementation::BufferOpenCL*>(src.impl().get());
  cl_int err;
  size_t src_offset = 0;
  size_t dst_origin[] = {0, 0, 0};
  size_t region[] = {descr.size.x, descr.size.y, descr.size.z};
  err = clEnqueueCopyBufferToImage(
      stream_impl->queue, src_impl->mem, mem, src_offset, dst_origin, region,
      stream_impl->events.size(), stream_impl->events, stream_impl->event());
  checkError(err);
  stream_impl->addEvent();
}

void ImageOpenCL::copy(const ghost::Stream& s, const void* src,
                       const ImageDescription& descr) {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  cl_int err;
  size_t origin[] = {0, 0, 0};
  size_t region[] = {descr.size.x, descr.size.y, descr.size.z};
  err = clEnqueueWriteImage(stream_impl->queue, mem, false, origin, region,
                            descr.stride.x, descr.stride.y, src,
                            stream_impl->events.size(), stream_impl->events,
                            stream_impl->event());
  checkError(err);
  stream_impl->addEvent();
}

void ImageOpenCL::copyTo(const ghost::Stream& s, ghost::Buffer& dst,
                         const ImageDescription& descr) const {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  auto dst_impl = static_cast<implementation::BufferOpenCL*>(dst.impl().get());
  cl_int err;
  size_t src_origin[] = {0, 0, 0};
  size_t dst_offset = 0;
  size_t region[] = {descr.size.x, descr.size.y, descr.size.z};
  err = clEnqueueCopyImageToBuffer(
      stream_impl->queue, mem, dst_impl->mem, src_origin, region, dst_offset,
      stream_impl->events.size(), stream_impl->events, stream_impl->event());
  checkError(err);
  stream_impl->addEvent();
}

void ImageOpenCL::copyTo(const ghost::Stream& s, void* dst,
                         const ImageDescription& descr) const {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  cl_int err;
  size_t origin[] = {0, 0, 0};
  size_t region[] = {descr.size.x, descr.size.y, descr.size.z};
  err = clEnqueueReadImage(stream_impl->queue, mem, false, origin, region,
                           descr.stride.x, descr.stride.y, dst,
                           stream_impl->events.size(), stream_impl->events,
                           stream_impl->event());
  checkError(err);
  stream_impl->addEvent();
}

// ---------------------------------------------------------------------------
// BufferPool
// ---------------------------------------------------------------------------

BufferPool::~BufferPool() { clear(); }

size_t BufferPool::getLimit() const { return _limit; }

void BufferPool::setLimit(size_t limit) {
  _limit = limit;
  if (_limit == 0) {
    clear();
  } else {
    purge();
  }
}

opencl::ptr<cl_mem> BufferPool::lookupBuffer(size_t bytes) {
  for (auto it = _buffers.begin(); it != _buffers.end(); ++it) {
    if (it->bytes == bytes) {
      auto mem = std::move(it->mem);
      _buffers.erase(it);
      return mem;
    }
  }
  purge(bytes);
  return opencl::ptr<cl_mem>();
}

opencl::ptr<cl_mem> BufferPool::lookupImage(const ImageDescription& descr) {
  for (auto it = _images.begin(); it != _images.end(); ++it) {
    if (imageMatch(it->descr, descr)) {
      auto mem = std::move(it->mem);
      _images.erase(it);
      return mem;
    }
  }
  return opencl::ptr<cl_mem>();
}

void BufferPool::reserve(size_t bytes) {
  _current += bytes;
  purge();
}

void BufferPool::recycleBuffer(opencl::ptr<cl_mem> mem, size_t bytes) {
  _buffers.push_back({std::move(mem), bytes});
}

void BufferPool::recycleImage(opencl::ptr<cl_mem> mem,
                              const ImageDescription& descr, size_t bytes) {
  _images.push_back({std::move(mem), descr, bytes});
}

void BufferPool::clear() {
  _buffers.clear();
  _images.clear();
  _current = 0;
}

void BufferPool::purge(size_t needed) {
  while (!_buffers.empty() && _current + needed > _limit) {
    size_t sz = _buffers.front().bytes;
    _current -= std::min(_current, sz);
    _buffers.pop_front();
  }
  while (!_images.empty() && _current + needed > _limit) {
    size_t sz = _images.front().bytes;
    _current -= std::min(_current, sz);
    _images.pop_front();
  }
}

bool BufferPool::imageMatch(const ImageDescription& a,
                            const ImageDescription& b) {
  return a.size.x == b.size.x && a.size.y == b.size.y && a.size.z == b.size.z &&
         a.channels == b.channels && a.order == b.order && a.type == b.type &&
         a.stride.x == b.stride.x && a.stride.y == b.stride.y &&
         a.access == b.access;
}

// ---------------------------------------------------------------------------
// PooledBufferOpenCL / PooledImageOpenCL
// ---------------------------------------------------------------------------

PooledBufferOpenCL::PooledBufferOpenCL(opencl::ptr<cl_mem> mem_, size_t bytes,
                                       std::shared_ptr<BufferPool> pool_)
    : BufferOpenCL(std::move(mem_), bytes), pool(std::move(pool_)) {}

PooledBufferOpenCL::~PooledBufferOpenCL() {
  if (pool && pool->getLimit() > 0) {
    pool->recycleBuffer(std::move(mem), _size);
  }
}

PooledImageOpenCL::PooledImageOpenCL(opencl::ptr<cl_mem> mem_,
                                     const ImageDescription& descr_,
                                     size_t bytes,
                                     std::shared_ptr<BufferPool> pool_)
    : ImageOpenCL(std::move(mem_), descr_),
      pool(std::move(pool_)),
      imageBytes(bytes) {}

PooledImageOpenCL::~PooledImageOpenCL() {
  if (pool && pool->getLimit() > 0) {
    pool->recycleImage(std::move(mem), descr, imageBytes);
  }
}

// ---------------------------------------------------------------------------
// DeviceOpenCL
// ---------------------------------------------------------------------------

DeviceOpenCL::DeviceOpenCL(const SharedContext& share) {
  cl_int err;
  if (share.context) {
    context = opencl::ptr<cl_context>(
        reinterpret_cast<cl_context>(share.context), true);
  }
  if (share.queue) {
    queue = opencl::ptr<cl_command_queue>(
        reinterpret_cast<cl_command_queue>(share.queue), true);
  }
  if (!context) {
    cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
    std::vector<cl_platform_id> platforms;
    std::vector<cl_device_id> devices;
    cl_platform_id platform = reinterpret_cast<cl_platform_id>(share.platform);
    if (share.device) {
      cl_device_id device = reinterpret_cast<cl_device_id>(share.device);
      devices.push_back(device);
      if (!platform) {
        err = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform),
                              &platform, nullptr);
        checkError(err);
      }
    } else {
      if (share.platform) {
        platforms.push_back(reinterpret_cast<cl_platform_id>(share.platform));
      } else {
        cl_uint num;
        err = clGetPlatformIDs(0, nullptr, &num);
#ifdef CL_PLATFORM_NOT_FOUND_KHR
        if (err == CL_PLATFORM_NOT_FOUND_KHR) {
          err = CL_SUCCESS;
          num = 0;
        }
#endif
        checkError(err);
        platforms.resize(size_t(num));
        if (!platforms.empty()) {
          err = clGetPlatformIDs(num, &platforms[0], nullptr);
          checkError(err);
        }
      }
      for (size_t i = 0; i < platforms.size() && devices.empty(); i++) {
        platform = platforms[i];
        cl_uint num;
        err = clGetDeviceIDs(platform, deviceType, 0, nullptr, &num);
        if (err != CL_SUCCESS) continue;
        devices.resize(size_t(num));
        if (!devices.empty()) {
          err = clGetDeviceIDs(platform, deviceType, num, &devices[0], nullptr);
          checkError(err);
        }
        if (devices.size() > 1) {
          devices.resize(1);
        }
      }
      // If no GPU devices found, fall back to any device type.
      if (devices.empty() && deviceType != CL_DEVICE_TYPE_ALL) {
        deviceType = CL_DEVICE_TYPE_ALL;
        for (size_t i = 0; i < platforms.size() && devices.empty(); i++) {
          platform = platforms[i];
          cl_uint num;
          err = clGetDeviceIDs(platform, deviceType, 0, nullptr, &num);
          if (err != CL_SUCCESS) continue;
          devices.resize(size_t(num));
          if (!devices.empty()) {
            err =
                clGetDeviceIDs(platform, deviceType, num, &devices[0], nullptr);
            checkError(err);
          }
          if (devices.size() > 1) {
            devices.resize(1);
          }
        }
      }
    }
    if (devices.empty()) {
      throw opencl::runtime_error(CL_DEVICE_NOT_FOUND);
    }
    cl_context_properties properties[] = {
        (cl_context_properties)CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform, (cl_context_properties)0};
    context = opencl::ptr<cl_context>(
        clCreateContext(properties, (cl_uint)devices.size(), &devices[0],
                        nullptr, nullptr, &err),
        false);
    checkError(err);
  }
  _fullProfile = getString(CL_DEVICE_PROFILE) != "EMBEDDED_PROFILE";
  if (!queue) {
    implementation::StreamOpenCL stream(*this);
    queue = stream.queue;
  }
  setVersion();
  set_of(_extensions, getPlatformString(CL_PLATFORM_EXTENSIONS));
  set_of(_extensions, getString(CL_DEVICE_EXTENSIONS));
}

DeviceOpenCL::DeviceOpenCL(const GpuInfo& info) {
  cl_int err;
  cl_uint numPlatforms;
  err = clGetPlatformIDs(0, nullptr, &numPlatforms);
  checkError(err);
  std::vector<cl_platform_id> platforms(numPlatforms);
  err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
  checkError(err);

  int platformIdx = info.index / 1000;
  int deviceIdx = info.index % 1000;

  cl_platform_id selectedPlatform = nullptr;
  cl_device_id selectedDevice = nullptr;

  if (platformIdx >= 0 && platformIdx < (int)platforms.size()) {
    cl_uint numDevices;
    err = clGetDeviceIDs(platforms[platformIdx], CL_DEVICE_TYPE_ALL, 0, nullptr,
                         &numDevices);
    checkError(err);
    std::vector<cl_device_id> devices(numDevices);
    err = clGetDeviceIDs(platforms[platformIdx], CL_DEVICE_TYPE_ALL, numDevices,
                         devices.data(), nullptr);
    checkError(err);
    if (deviceIdx >= 0 && deviceIdx < (int)devices.size()) {
      selectedPlatform = platforms[platformIdx];
      selectedDevice = devices[deviceIdx];
    }
  }

  if (selectedDevice) {
    cl_context_properties properties[] = {
        (cl_context_properties)CL_CONTEXT_PLATFORM,
        (cl_context_properties)selectedPlatform, (cl_context_properties)0};
    context = opencl::ptr<cl_context>(
        clCreateContext(properties, 1, &selectedDevice, nullptr, nullptr, &err),
        false);
    checkError(err);
  } else {
    // Fallback: default device via SharedContext
    DeviceOpenCL tmp{SharedContext()};
    context = std::move(tmp.context);
    queue = std::move(tmp.queue);
    _version = std::move(tmp._version);
    _extensions = std::move(tmp._extensions);
    _fullProfile = tmp._fullProfile;
    return;
  }
  _fullProfile = getString(CL_DEVICE_PROFILE) != "EMBEDDED_PROFILE";
  implementation::StreamOpenCL stream(*this);
  queue = stream.queue;
  setVersion();
  set_of(_extensions, getPlatformString(CL_PLATFORM_EXTENSIONS));
  set_of(_extensions, getString(CL_DEVICE_EXTENSIONS));
}

DeviceOpenCL::DeviceOpenCL(cl_platform_id platform, cl_device_id device) {
  cl_int err;
  cl_context_properties properties[] = {
      (cl_context_properties)CL_CONTEXT_PLATFORM,
      (cl_context_properties)platform, (cl_context_properties)0};
  context = opencl::ptr<cl_context>(
      clCreateContext(properties, 1, &device, nullptr, nullptr, &err), false);
  checkError(err);
  _fullProfile = getString(CL_DEVICE_PROFILE) != "EMBEDDED_PROFILE";
  implementation::StreamOpenCL stream(*this);
  queue = stream.queue;
  setVersion();
  set_of(_extensions, getPlatformString(CL_PLATFORM_EXTENSIONS));
  set_of(_extensions, getString(CL_DEVICE_EXTENSIONS));
}

ghost::Library DeviceOpenCL::loadLibraryFromText(
    const std::string& text, const std::string& options) const {
  auto ptr = std::make_shared<implementation::LibraryOpenCL>(*this);
  ptr->loadFromText(text, options);
  return ghost::Library(ptr);
}

ghost::Library DeviceOpenCL::loadLibraryFromData(
    const void* data, size_t len, const std::string& options) const {
  auto ptr = std::make_shared<implementation::LibraryOpenCL>(*this);
  ptr->loadFromData(data, len, options);
  return ghost::Library(ptr);
}

SharedContext DeviceOpenCL::shareContext() const {
  SharedContext c(context.get(), queue.get());
  return c;
}

ghost::Stream DeviceOpenCL::createStream() const {
  auto ptr = std::make_shared<implementation::StreamOpenCL>(*this);
  return ghost::Stream(ptr);
}

size_t DeviceOpenCL::getMemoryPoolSize() const {
  return Device::getMemoryPoolSize();
}

void DeviceOpenCL::setMemoryPoolSize(size_t bytes) {
  Device::setMemoryPoolSize(bytes);
  if (bytes > 0) {
    if (!_pool) _pool = std::make_shared<BufferPool>();
    _pool->setLimit(bytes);
  } else if (_pool) {
    _pool->setLimit(0);
  }
}

ghost::Buffer DeviceOpenCL::allocateBuffer(size_t bytes, Access access) const {
  if (_pool && _pool->getLimit() > 0) {
    auto mem = _pool->lookupBuffer(bytes);
    if (mem.get()) {
      return ghost::Buffer(
          std::make_shared<PooledBufferOpenCL>(std::move(mem), bytes, _pool));
    }
    // Allocate new, but wrap in pooled buffer for recycling on destruction.
    cl_int err;
    cl_mem_flags flags = getMemFlags(access);
    auto newMem = opencl::ptr<cl_mem>(
        clCreateBuffer(context, flags, bytes, nullptr, &err));
    checkError(err);
    _pool->reserve(bytes);
    return ghost::Buffer(
        std::make_shared<PooledBufferOpenCL>(std::move(newMem), bytes, _pool));
  }
  auto ptr =
      std::make_shared<implementation::BufferOpenCL>(*this, bytes, access);
  return ghost::Buffer(ptr);
}

ghost::MappedBuffer DeviceOpenCL::allocateMappedBuffer(size_t bytes,
                                                       Access access) const {
  auto ptr = std::make_shared<implementation::MappedBufferOpenCL>(*this, bytes,
                                                                  access);
  return ghost::MappedBuffer(ptr);
}

ghost::Image DeviceOpenCL::allocateImage(const ImageDescription& descr) const {
  if (_pool && _pool->getLimit() > 0) {
    auto mem = _pool->lookupImage(descr);
    if (mem.get()) {
      // Estimate image size for pool accounting.
      size_t rowBytes =
          descr.stride.x > 0 ? (size_t)descr.stride.x : descr.size.x * 4;
      size_t bytes = descr.size.y * rowBytes * descr.size.z;
      return ghost::Image(std::make_shared<PooledImageOpenCL>(
          std::move(mem), descr, bytes, _pool));
    }
    // Create new image and wrap for pooling.
    auto img = std::make_shared<ImageOpenCL>(*this, descr);
    size_t rowBytes =
        descr.stride.x > 0 ? (size_t)descr.stride.x : descr.size.x * 4;
    size_t bytes = descr.size.y * rowBytes * descr.size.z;
    _pool->reserve(bytes);
    return ghost::Image(std::make_shared<PooledImageOpenCL>(
        std::move(img->mem), descr, bytes, _pool));
  }
  auto ptr = std::make_shared<implementation::ImageOpenCL>(*this, descr);
  return ghost::Image(ptr);
}

ghost::Image DeviceOpenCL::sharedImage(const ImageDescription& descr,
                                       ghost::Buffer& buffer) const {
  auto b = static_cast<implementation::BufferOpenCL*>(buffer.impl().get());
  auto ptr = std::make_shared<implementation::ImageOpenCL>(*this, descr, *b);
  return ghost::Image(ptr);
}

ghost::Image DeviceOpenCL::sharedImage(const ImageDescription& descr,
                                       ghost::Image& image) const {
  auto i = static_cast<implementation::ImageOpenCL*>(image.impl().get());
  auto ptr = std::make_shared<implementation::ImageOpenCL>(*this, descr, *i);
  return ghost::Image(ptr);
}

Attribute DeviceOpenCL::getAttribute(DeviceAttributeId what) const {
  switch (what) {
    case kDeviceImplementation:
      return "OpenCL";
    case kDeviceName:
      return getString(CL_DEVICE_NAME);
    case kDeviceVendor:
      return getString(CL_DEVICE_VENDOR);
    case kDeviceDriverVersion:
      return getString(CL_DRIVER_VERSION);
    case kDeviceCount:
      return (int32_t)getDevices().size();
    case kDeviceProcessorCount:
      return (uint64_t)getInt(CL_DEVICE_MAX_COMPUTE_UNITS);
    case kDeviceUnifiedMemory:
      return getInt(CL_DEVICE_HOST_UNIFIED_MEMORY) != 0;
    case kDeviceMemory:
      return (uint64_t)getInt(CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    case kDeviceLocalMemory:
      return (uint64_t)getInt(CL_DEVICE_LOCAL_MEM_SIZE);
    case kDeviceMaxThreads:
      return (uint64_t)getInt(CL_DEVICE_MAX_WORK_GROUP_SIZE);
    case kDeviceMaxWorkSize: {
      auto devices = getDevices();
      cl_int err;
      cl_ulong v[3];
      err = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_ITEM_SIZES,
                            sizeof(v[0]) * 3, v, nullptr);
      checkError(err);
      return Attribute((uint64_t)v[0], (uint64_t)v[1], (uint64_t)v[2]);
    }
    case kDeviceMaxRegisters:
      return 0;
    case kDeviceMaxImageSize1:
      return (uint64_t)getInt(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE);
    case kDeviceMaxImageSize2:
      return Attribute((uint64_t)getInt(CL_DEVICE_IMAGE2D_MAX_WIDTH),
                       (uint64_t)getInt(CL_DEVICE_IMAGE2D_MAX_HEIGHT));
    case kDeviceMaxImageSize3:
      return Attribute((uint64_t)getInt(CL_DEVICE_IMAGE3D_MAX_WIDTH),
                       (uint64_t)getInt(CL_DEVICE_IMAGE3D_MAX_HEIGHT),
                       (uint64_t)getInt(CL_DEVICE_IMAGE3D_MAX_DEPTH));
    case kDeviceImageAlignment:
      return (uint64_t)getInt(CL_DEVICE_IMAGE_PITCH_ALIGNMENT);
    case kDeviceSupportsImageFloatFiltering:
      if (!_fullProfile) {
        return false;
      }
      // continue below
    case kDeviceSupportsImageIntegerFiltering:
#ifdef __APPLE_CC__
      if (getString(CL_DEVICE_VENDOR) == "Apple") {
        // Interpolation is broken on M1 as of macOS 11.2
        return false;
      }
#endif
      return true;
    case kDeviceSupportsMappedBuffer:
      return true;
    case kDeviceSupportsProgramConstants:
      return false;
    case kDeviceSupportsSubgroup:
      return checkExtension("cl_khr_subgroups");
    case kDeviceSupportsSubgroupShuffle:
      return checkExtension("cl_intel_subgroups");
    case kDeviceSubgroupWidth: {
#ifndef __APPLE_CC__
      if (checkExtension("cl_nv_device_attribute_query")) {
        return (uint64_t)getInt(CL_DEVICE_WARP_SIZE_NV);
      } else if (checkExtension("cl_amd_device_attribute_query")) {
        return (uint64_t)getInt(CL_DEVICE_WAVEFRONT_WIDTH_AMD);
      }
#endif
      std::string vendor = getString(CL_DEVICE_VENDOR);
      uint32_t size = 1;
      if (vendor.find("NVIDIA") != std::string::npos) {
        size = 32;
      } else if (vendor.find("ATI") != std::string::npos ||
                 vendor.find("AMD") != std::string::npos) {
        size = 64;
      } else if (vendor.find("Intel") != std::string::npos) {
        // According to Intel's DirectX docs (but not mentioned in
        // OpenCL docs)
        // this can vary between 8 and 32.
        size = 8;
      }
      return size;
    }
    case kDeviceMaxComputeUnits:
      return (uint64_t)getInt(CL_DEVICE_MAX_COMPUTE_UNITS);
    case kDeviceMemoryAlignment:
      return (uint64_t)getInt(CL_DEVICE_MEM_BASE_ADDR_ALIGN) / 8;
    case kDeviceBufferAlignment:
      return (uint64_t)getInt(CL_DEVICE_MEM_BASE_ADDR_ALIGN) / 8;
    case kDeviceMaxBufferSize:
      return (uint64_t)getInt(CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    case kDeviceMaxConstantBufferSize:
      return (uint64_t)getInt(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
    case kDeviceTimestampPeriod: {
      auto devices = getDevices();
      cl_int err;
      cl_ulong res;
      err = clGetDeviceInfo(devices[0], CL_DEVICE_PROFILING_TIMER_RESOLUTION,
                            sizeof(res), &res, nullptr);
      checkError(err);
      return (float)res;
    }
    case kDeviceSupportsProfilingTimer:
      return true;
    default:
      return Attribute();
  }
}

void DeviceOpenCL::setVersion() {
  _version = getString(CL_DEVICE_VERSION);
  // _version is "OpenCL X.Y ..." — extract the numeric part after "OpenCL ".
  auto pos = _version.find(' ');
  if (pos != std::string::npos) {
    _version = _version.substr(pos + 1);
  }
}

bool DeviceOpenCL::checkVersion(const std::string& version) const {
  return _version.compare(0, version.size(), version) >= 0;
}

bool DeviceOpenCL::checkExtension(const std::string& extension) const {
  return _extensions.find(extension) != _extensions.end();
}

std::vector<cl_device_id> DeviceOpenCL::getDevices() const {
  int err;
  std::vector<cl_device_id> devices;
  size_t numDevs;
  err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, nullptr, &numDevs);
  checkError(err);
  numDevs /= sizeof(cl_device_id);
  devices.resize(size_t(numDevs));
  err = clGetContextInfo(context, CL_CONTEXT_DEVICES,
                         numDevs * sizeof(cl_device_id), &devices[0], nullptr);
  checkError(err);

  return devices;
}

cl_platform_id DeviceOpenCL::getPlatform() const {
  cl_platform_id platform;
  cl_int err;
  err = clGetDeviceInfo(getDevices()[0], CL_DEVICE_PLATFORM, sizeof(platform),
                        &platform, nullptr);
  checkError(err);
  return platform;
}

cl_ulong DeviceOpenCL::getInt(cl_device_info param_name) const {
  auto devices = getDevices();
  cl_int err;
  cl_ulong v;
  err = clGetDeviceInfo(devices[0], param_name, sizeof(v), &v, nullptr);
  checkError(err);
  return v;
}

std::string DeviceOpenCL::getString(cl_device_info param_name) const {
  auto devices = getDevices();
  cl_int err;
  size_t infoSize;
  std::vector<char> str;
  err = clGetDeviceInfo(devices[0], param_name, 0, nullptr, &infoSize);
  checkError(err);
  if (infoSize == 0) return "";
  str.resize(size_t(infoSize));
  err = clGetDeviceInfo(devices[0], param_name, infoSize, &str[0], nullptr);
  checkError(err);
  return &str[0];
}

std::string DeviceOpenCL::getPlatformString(cl_platform_info param_name) const {
  auto platform = getPlatform();
  cl_int err;
  size_t infoSize;
  std::vector<char> str;
  err = clGetPlatformInfo(platform, param_name, 0, nullptr, &infoSize);
  checkError(err);
  if (infoSize == 0) return "";
  str.resize(size_t(infoSize));
  err = clGetPlatformInfo(platform, param_name, infoSize, &str[0], nullptr);
  checkError(err);
  return &str[0];
}
}  // namespace implementation

DeviceOpenCL::DeviceOpenCL(const SharedContext& share)
    : Device(std::make_shared<implementation::DeviceOpenCL>(share)) {
  auto opencl = static_cast<implementation::DeviceOpenCL*>(impl().get());
  setDefaultStream(
      std::make_shared<implementation::StreamOpenCL>(opencl->queue));
}

DeviceOpenCL::DeviceOpenCL(const GpuInfo& info)
    : Device(std::make_shared<implementation::DeviceOpenCL>(info)) {
  auto opencl = static_cast<implementation::DeviceOpenCL*>(impl().get());
  setDefaultStream(
      std::make_shared<implementation::StreamOpenCL>(opencl->queue));
}

std::vector<GpuInfo> DeviceOpenCL::enumerateDevices() {
  std::vector<GpuInfo> result;
  cl_int err;
  cl_uint numPlatforms;
  err = clGetPlatformIDs(0, nullptr, &numPlatforms);
#ifdef CL_PLATFORM_NOT_FOUND_KHR
  if (err == CL_PLATFORM_NOT_FOUND_KHR) return result;
#endif
  if (err != CL_SUCCESS) return result;
  std::vector<cl_platform_id> platforms(numPlatforms);
  err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
  if (err != CL_SUCCESS) return result;

  for (cl_uint pi = 0; pi < numPlatforms; pi++) {
    cl_uint numDevices;
    err = clGetDeviceIDs(platforms[pi], CL_DEVICE_TYPE_ALL, 0, nullptr,
                         &numDevices);
    if (err != CL_SUCCESS) continue;
    std::vector<cl_device_id> devices(numDevices);
    err = clGetDeviceIDs(platforms[pi], CL_DEVICE_TYPE_ALL, numDevices,
                         devices.data(), nullptr);
    if (err != CL_SUCCESS) continue;

    for (cl_uint di = 0; di < numDevices; di++) {
      GpuInfo info;
      char buf[256];
      size_t len;

      if (clGetDeviceInfo(devices[di], CL_DEVICE_NAME, sizeof(buf), buf,
                          &len) == CL_SUCCESS)
        info.name = std::string(buf, len > 0 ? len - 1 : 0);

      if (clGetDeviceInfo(devices[di], CL_DEVICE_VENDOR, sizeof(buf), buf,
                          &len) == CL_SUCCESS)
        info.vendor = std::string(buf, len > 0 ? len - 1 : 0);

      info.implementation = "OpenCL";

      cl_ulong memSize;
      if (clGetDeviceInfo(devices[di], CL_DEVICE_GLOBAL_MEM_SIZE,
                          sizeof(memSize), &memSize, nullptr) == CL_SUCCESS)
        info.memory = memSize;

      cl_bool unified;
      if (clGetDeviceInfo(devices[di], CL_DEVICE_HOST_UNIFIED_MEMORY,
                          sizeof(unified), &unified, nullptr) == CL_SUCCESS)
        info.unifiedMemory = unified != 0;

      info.index = (int)pi * 1000 + (int)di;
      result.push_back(info);
    }
  }
  return result;
}
}  // namespace ghost
#endif
