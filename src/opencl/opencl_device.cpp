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

#include <ghost/allocator.h>
#include <ghost/argument_buffer.h>
#include <ghost/exception.h>
#include <ghost/opencl/device.h>
#include <ghost/opencl/exception.h>
#include <ghost/opencl/impl_device.h>
#include <ghost/opencl/impl_function.h>
#include <string.h>

#include <type_traits>
#include <variant>

#if WITH_OPENCL_COMMAND_BUFFERS
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
    case Access::WriteOnly:
      return CL_MEM_WRITE_ONLY;
    case Access::ReadOnly:
      return CL_MEM_READ_ONLY;
    case Access::ReadWrite:
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

double EventOpenCL::timestamp() const {
  cl_ulong time;
  cl_int err = clGetEventProfilingInfo(event.get(), CL_PROFILING_COMMAND_END,
                                       sizeof(time), &time, nullptr);
  if (err != CL_SUCCESS) return 0.0;
  return static_cast<double>(time) / 1e9;
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
    // outOfOrder here means "use the event-chain machinery to order dependent
    // ops" (addEvent / event() / sync's wait-on-events path), not "the queue
    // is OOO." The event chain works correctly against both OOO and in-order
    // queues — wait_list is honored regardless — so we always enable it for
    // externally-supplied queues. Querying the queue's actual OOO bit and
    // disabling the chain on in-order queues (e.g. Apple's OpenCL 1.2, which
    // doesn't expose OOO queues) leaves dependent fill→read pairs racing on
    // the host-driver path: BufferTest.Fill* and similar fail.
    : queue(queue_), outOfOrder(true) {}

StreamOpenCL::StreamOpenCL(const DeviceOpenCL& dev,
                           const StreamOptions& options)
    // outOfOrder here means "use the event-chain machinery to order
    // dependent ops" (addEvent / event() / sync's wait-on-events path), not
    // "the queue type is OOO." The event chain works correctly against both
    // OOO and in-order queues — wait_list is honored regardless — so we
    // always enable it. Drivers that don't expose an OOO queue (Apple's
    // OpenCL 1.2) silently get an in-order queue below; the event chain
    // still orders dependent ops correctly there. forceEventChain is
    // therefore redundant and kept only for source compatibility.
    : outOfOrder(true) {
  cl_int err;
  cl_command_queue_properties queueProperties = 0;
  cl_command_queue_properties devQueueProperties = 0;
  auto devices = dev.getDevices();
  queueProperties |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
  if (options.profiling) queueProperties |= CL_QUEUE_PROFILING_ENABLE;
  err = clGetDeviceInfo(devices[0], CL_DEVICE_QUEUE_PROPERTIES,
                        sizeof(devQueueProperties), &devQueueProperties, NULL);
  checkError(err);
  queueProperties &= devQueueProperties;
  if (queue.get() == nullptr && dev.context.get() != nullptr) {
    if (dev.checkVersion("2.0")) {
#ifdef CL_VERSION_2_0
      cl_queue_properties p[] = {CL_QUEUE_PROPERTIES, queueProperties, 0};
      queue = opencl::ptr<cl_command_queue>(
          clCreateCommandQueueWithProperties(dev.context, devices[0], p, &err));
      checkError(err);
#endif
    }
    if (!queue) {
      queue = opencl::ptr<cl_command_queue>(
          clCreateCommandQueue(dev.context, devices[0], queueProperties, &err));
      checkError(err);
    }
  }
}

StreamOpenCL::~StreamOpenCL() {
  // Drain before letting owner deleters run for any host memory still held
  // alive by a queued DMA. Without this, async owned-handle uploads/readbacks
  // could free their backing memory while the GPU is still reading or
  // writing it.
  if (!pendingHostMemory.empty() && queue.get()) {
    clFinish(queue);
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
  // After sync, every enqueued op has completed → drop all retained owners
  // (which will run their user-supplied deleters here on this thread).
  pendingHostMemory.clear();
}

void StreamOpenCL::addEvent() {
  if (outOfOrder) {
    events.reset();
    events.push(lastEvent);
  }
}

cl_event* StreamOpenCL::event() { return outOfOrder ? &lastEvent : nullptr; }

cl_event* StreamOpenCL::eventForOwned() { return &lastEvent; }

void StreamOpenCL::retainHostUntilDone(std::shared_ptr<void> owner) {
  if (!owner || lastEvent.get() == nullptr) return;
  reapPendingHostMemory();
  pendingHostMemory.push_back({lastEvent, std::move(owner)});
}

void StreamOpenCL::reapPendingHostMemory() {
  auto it = pendingHostMemory.begin();
  while (it != pendingHostMemory.end()) {
    cl_int status = CL_QUEUED;
    cl_int err = clGetEventInfo(it->event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                sizeof(status), &status, nullptr);
    if (err == CL_SUCCESS && status == CL_COMPLETE) {
      it = pendingHostMemory.erase(it);
    } else {
      ++it;
    }
  }
}

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

void StreamOpenCL::barrier() {
  // No host drain. Intra-queue ordering is already guaranteed: on an
  // out-of-order queue every enqueue waits on the prior op's event (the
  // events/lastEvent chain in execute() and the copy/fill paths), and an
  // in-order queue orders by FIFO. So a CommandBuffer barrier between two
  // ops needs no work here — the op after it already happens-after the op
  // before it. Host visibility is provided by the caller's Stream::sync().
}

void CommandBufferOpenCL::encodeNative(
    std::function<void(cl_command_queue)> body) {
  addEncodeNative([body = std::move(body)](void* ctx) {
    body(static_cast<cl_command_queue>(ctx));
  });
}

void CommandBufferOpenCL::replayEncodeNative(const EncodeNativeCmd& cmd,
                                             const ghost::Stream& stream) {
  auto* sCL = static_cast<StreamOpenCL*>(stream.impl().get());
  cmd.body(sCL->queue.get());
}

BufferOpenCL::BufferOpenCL(opencl::ptr<cl_mem> mem_, size_t bytes)
    : mem(mem_), _size(bytes) {}

BufferOpenCL::~BufferOpenCL() {
  if (_allocator) {
    _allocator->freeBuffer((void*)mem.release(), _size);
    return;
  }
  // OpenCL's runtime defers actual deletion of a cl_mem until queued
  // commands referencing it have finished (CL spec 5.6.2). So calling
  // clReleaseMemObject here while a fire-and-forget dispatch is still
  // in flight is safe — the runtime keeps the allocation alive until
  // the pending kernels/copies complete. No explicit per-use event
  // tracking is needed on this path.
  // opencl::ptr destructor calls clReleaseMemObject.
}

BufferOpenCL::BufferOpenCL(const DeviceOpenCL& dev, size_t bytes,
                           const BufferOptions& opts)
    : _size(bytes) {
  cl_int err;
  cl_mem_flags flags = getMemFlags(opts.access);
  // Staging hint: prefer host-visible memory for fast CPU<->GPU transfers.
  if (opts.hint == AllocHint::Staging) {
    flags |= CL_MEM_ALLOC_HOST_PTR;
  }
  mem = opencl::ptr<cl_mem>(
      clCreateBuffer(dev.context, flags, bytes, nullptr, &err));
  checkError(err);
}

size_t BufferOpenCL::size() const { return _size; }

void BufferOpenCL::copy(const ghost::Encoder& s, const ghost::Buffer& src,
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

void BufferOpenCL::copy(const ghost::Encoder& s, const void* src,
                        size_t bytes) {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  cl_int err;
  // Sync: src is consumed before return (HostBytes::borrow contract). The
  // HostBytes::adopt overload below is the async-with-managed-lifetime path.
  err = clEnqueueWriteBuffer(stream_impl->queue, mem, /*blocking=*/true, 0,
                             bytes, src, stream_impl->events.size(),
                             stream_impl->events, stream_impl->event());
  checkError(err);
  stream_impl->addEvent();
}

void BufferOpenCL::copyTo(const ghost::Encoder& s, void* dst,
                          size_t bytes) const {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  cl_int err;
  // Sync: readback completes before return (HostBytes::borrow contract). The
  // HostBytes::adopt overload below is the async-with-managed-lifetime path.
  err = clEnqueueReadBuffer(stream_impl->queue, mem, /*blocking=*/true, 0,
                            bytes, dst, stream_impl->events.size(),
                            stream_impl->events, stream_impl->event());
  checkError(err);
  stream_impl->addEvent();
}

void BufferOpenCL::copy(const ghost::Encoder& s, const ghost::Buffer& src,
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

void BufferOpenCL::copy(const ghost::Encoder& s, const void* src,
                        size_t dstOffset, size_t bytes) {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  cl_int err;
  // Sync: see BufferOpenCL::copy(Encoder, const void*, size_t).
  err = clEnqueueWriteBuffer(stream_impl->queue, mem, /*blocking=*/true,
                             dstOffset, bytes, src, stream_impl->events.size(),
                             stream_impl->events, stream_impl->event());
  checkError(err);
  stream_impl->addEvent();
}

void BufferOpenCL::copy(const ghost::Encoder& s, HostBytes src,
                        size_t dstOffset, size_t bytes) {
  if (!src.ownsBytes()) {
    // Borrow → fall through to the synchronous-consume overload.
    this->copy(s, src.data(), dstOffset, bytes);
    return;
  }
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  cl_int err;
  err = clEnqueueWriteBuffer(stream_impl->queue, mem, /*blocking=*/false,
                             dstOffset, bytes, src.data(),
                             stream_impl->events.size(), stream_impl->events,
                             stream_impl->eventForOwned());
  checkError(err);
  stream_impl->retainHostUntilDone(src.owner());
  stream_impl->addEvent();
}

void BufferOpenCL::copyTo(const ghost::Encoder& s, void* dst, size_t srcOffset,
                          size_t bytes) const {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  cl_int err;
  // Sync: see BufferOpenCL::copyTo(Encoder, void*, size_t).
  err = clEnqueueReadBuffer(stream_impl->queue, mem, /*blocking=*/true,
                            srcOffset, bytes, dst, stream_impl->events.size(),
                            stream_impl->events, stream_impl->event());
  checkError(err);
  stream_impl->addEvent();
}

void BufferOpenCL::copyTo(const ghost::Encoder& s, HostBytes dst,
                          size_t srcOffset, size_t bytes) const {
  if (!dst.ownsBytes()) {
    this->copyTo(s, dst.data(), srcOffset, bytes);
    return;
  }
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  cl_int err;
  err = clEnqueueReadBuffer(stream_impl->queue, mem, /*blocking=*/false,
                            srcOffset, bytes, dst.data(),
                            stream_impl->events.size(), stream_impl->events,
                            stream_impl->eventForOwned());
  checkError(err);
  stream_impl->retainHostUntilDone(dst.owner());
  stream_impl->addEvent();
}

void BufferOpenCL::fill(const ghost::Encoder& s, size_t offset, size_t size,
                        uint8_t value) {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  cl_int err;
  err = clEnqueueFillBuffer(stream_impl->queue, mem, &value, sizeof(value),
                            offset, size, stream_impl->events.size(),
                            stream_impl->events, stream_impl->event());
  checkError(err);
  stream_impl->addEvent();
}

void BufferOpenCL::fill(const ghost::Encoder& s, size_t offset, size_t size,
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
  return std::make_shared<SubBufferOpenCL>(self, opencl::ptr<cl_mem>(sub), size,
                                           offset);
}

SubBufferOpenCL::SubBufferOpenCL(std::shared_ptr<Buffer> parent,
                                 opencl::ptr<cl_mem> mem_, size_t bytes,
                                 size_t offset)
    : BufferOpenCL(mem_, bytes), _parent(parent), _offset(offset) {}

std::shared_ptr<Buffer> SubBufferOpenCL::createSubBuffer(
    const std::shared_ptr<Buffer>& self, size_t offset, size_t size) {
  // OpenCL forbids creating a sub-buffer from a sub-buffer
  // (CL_INVALID_MEM_OBJECT). Walk up to the root buffer and accumulate offsets.
  size_t totalOffset = _offset + offset;
  auto root = _parent;
  while (auto* sub = dynamic_cast<SubBufferOpenCL*>(root.get())) {
    totalOffset += sub->_offset;
    root = sub->_parent;
  }
  return root->createSubBuffer(root, totalOffset, size);
}

MappedBufferOpenCL::MappedBufferOpenCL(opencl::ptr<cl_mem> mem_, size_t bytes,
                                       size_t allocSize)
    : BufferOpenCL(mem_, allocSize), length(bytes), ptr(nullptr) {}

MappedBufferOpenCL::~MappedBufferOpenCL() {
  if (_allocator) {
    _allocator->freeMappedBuffer((void*)mem.release(), _size);
    _allocator = nullptr;  // suppress base BufferOpenCL destructor
  }
}

MappedBufferOpenCL::MappedBufferOpenCL(const DeviceOpenCL& dev, size_t bytes,
                                       const BufferOptions& opts)
    : BufferOpenCL(opencl::ptr<cl_mem>(), bytes), length(bytes), ptr(nullptr) {
  cl_int err;
  cl_mem_flags flags = getMemFlags(opts.access);
  flags |= CL_MEM_ALLOC_HOST_PTR;
  mem = opencl::ptr<cl_mem>(
      clCreateBuffer(dev.context, flags, bytes, nullptr, &err));
  checkError(err);
}

void* MappedBufferOpenCL::map(const ghost::Encoder& s, Access access,
                              bool sync) {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  cl_int err;
  cl_map_flags flags;
  switch (access) {
    case Access::ReadOnly:
      flags = CL_MAP_READ;
      break;
    case Access::WriteOnly:
      flags = CL_MAP_WRITE_INVALIDATE_REGION;
      break;
    case Access::ReadWrite:
      flags = CL_MAP_READ | CL_MAP_WRITE;
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

void MappedBufferOpenCL::unmap(const ghost::Encoder& s) {
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

ImageOpenCL::~ImageOpenCL() {
  if (_allocator) {
    _allocator->freeImage((void*)mem.release(), descr);
  }
}

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

void ImageOpenCL::copy(const ghost::Encoder& s, const ghost::Image& src) {
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

void ImageOpenCL::copy(const ghost::Encoder& s, const ghost::Buffer& src,
                       const BufferLayout& layout) {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  auto src_impl = static_cast<implementation::BufferOpenCL*>(src.impl().get());
  cl_int err;
  size_t src_offset = 0;
  size_t dst_origin[] = {0, 0, 0};
  size_t region[] = {layout.size.x, layout.size.y, layout.size.z};
  err = clEnqueueCopyBufferToImage(
      stream_impl->queue, src_impl->mem, mem, src_offset, dst_origin, region,
      stream_impl->events.size(), stream_impl->events, stream_impl->event());
  checkError(err);
  stream_impl->addEvent();
}

void ImageOpenCL::copy(const ghost::Encoder& s, const void* src,
                       const BufferLayout& layout) {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  cl_int err;
  size_t origin[] = {0, 0, 0};
  size_t region[] = {layout.size.x, layout.size.y, layout.size.z};
  // Sync: see BufferOpenCL::copy(Encoder, const void*, size_t).
  err = clEnqueueWriteImage(stream_impl->queue, mem, /*blocking=*/true, origin,
                            region, layout.stride.x, layout.stride.y, src,
                            stream_impl->events.size(), stream_impl->events,
                            stream_impl->event());
  checkError(err);
  stream_impl->addEvent();
}

void ImageOpenCL::copy(const ghost::Encoder& s, HostBytes src,
                       const BufferLayout& layout) {
  if (!src.ownsBytes()) {
    this->copy(s, src.data(), layout);
    return;
  }
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  cl_int err;
  size_t origin[] = {0, 0, 0};
  size_t region[] = {layout.size.x, layout.size.y, layout.size.z};
  err = clEnqueueWriteImage(stream_impl->queue, mem, /*blocking=*/false, origin,
                            region, layout.stride.x, layout.stride.y,
                            src.data(), stream_impl->events.size(),
                            stream_impl->events, stream_impl->eventForOwned());
  checkError(err);
  stream_impl->retainHostUntilDone(src.owner());
  stream_impl->addEvent();
}

void ImageOpenCL::copyTo(const ghost::Encoder& s, ghost::Buffer& dst,
                         const BufferLayout& layout) const {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  auto dst_impl = static_cast<implementation::BufferOpenCL*>(dst.impl().get());
  cl_int err;
  size_t src_origin[] = {0, 0, 0};
  size_t dst_offset = 0;
  size_t region[] = {layout.size.x, layout.size.y, layout.size.z};
  err = clEnqueueCopyImageToBuffer(
      stream_impl->queue, mem, dst_impl->mem, src_origin, region, dst_offset,
      stream_impl->events.size(), stream_impl->events, stream_impl->event());
  checkError(err);
  stream_impl->addEvent();
}

void ImageOpenCL::copyTo(const ghost::Encoder& s, void* dst,
                         const BufferLayout& layout) const {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  cl_int err;
  size_t origin[] = {0, 0, 0};
  size_t region[] = {layout.size.x, layout.size.y, layout.size.z};
  // Sync: see BufferOpenCL::copyTo(Encoder, void*, size_t).
  err = clEnqueueReadImage(stream_impl->queue, mem, /*blocking=*/true, origin,
                           region, layout.stride.x, layout.stride.y, dst,
                           stream_impl->events.size(), stream_impl->events,
                           stream_impl->event());
  checkError(err);
  stream_impl->addEvent();
}

void ImageOpenCL::copyTo(const ghost::Encoder& s, HostBytes dst,
                         const BufferLayout& layout) const {
  if (!dst.ownsBytes()) {
    this->copyTo(s, dst.data(), layout);
    return;
  }
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  cl_int err;
  size_t origin[] = {0, 0, 0};
  size_t region[] = {layout.size.x, layout.size.y, layout.size.z};
  err = clEnqueueReadImage(stream_impl->queue, mem, /*blocking=*/false, origin,
                           region, layout.stride.x, layout.stride.y, dst.data(),
                           stream_impl->events.size(), stream_impl->events,
                           stream_impl->eventForOwned());
  checkError(err);
  stream_impl->retainHostUntilDone(dst.owner());
  stream_impl->addEvent();
}

void ImageOpenCL::copy(const ghost::Encoder& s, const ghost::Buffer& src,
                       const BufferLayout& layout, const Origin3& imageOrigin) {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  auto src_impl = static_cast<implementation::BufferOpenCL*>(src.impl().get());
  cl_int err;
  size_t src_offset = 0;
  size_t dst_origin[] = {imageOrigin.x, imageOrigin.y, imageOrigin.z};
  size_t region[] = {layout.size.x, layout.size.y, layout.size.z};
  err = clEnqueueCopyBufferToImage(
      stream_impl->queue, src_impl->mem, mem, src_offset, dst_origin, region,
      stream_impl->events.size(), stream_impl->events, stream_impl->event());
  checkError(err);
  stream_impl->addEvent();
}

void ImageOpenCL::copyTo(const ghost::Encoder& s, ghost::Buffer& dst,
                         const BufferLayout& layout,
                         const Origin3& imageOrigin) const {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  auto dst_impl = static_cast<implementation::BufferOpenCL*>(dst.impl().get());
  cl_int err;
  size_t src_origin[] = {imageOrigin.x, imageOrigin.y, imageOrigin.z};
  size_t dst_offset = 0;
  size_t region[] = {layout.size.x, layout.size.y, layout.size.z};
  err = clEnqueueCopyImageToBuffer(
      stream_impl->queue, mem, dst_impl->mem, src_origin, region, dst_offset,
      stream_impl->events.size(), stream_impl->events, stream_impl->event());
  checkError(err);
  stream_impl->addEvent();
}

void ImageOpenCL::copy(const ghost::Encoder& s, const ghost::Image& src,
                       const Size3& region, const Origin3& srcOrigin,
                       const Origin3& dstOrigin) {
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  auto src_impl = static_cast<implementation::ImageOpenCL*>(src.impl().get());
  cl_int err;
  size_t src_o[] = {srcOrigin.x, srcOrigin.y, srcOrigin.z};
  size_t dst_o[] = {dstOrigin.x, dstOrigin.y, dstOrigin.z};
  size_t reg[] = {region.x, region.y, region.z};
  err = clEnqueueCopyImage(stream_impl->queue, src_impl->mem, mem, src_o, dst_o,
                           reg, stream_impl->events.size(), stream_impl->events,
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

ghost::Library DeviceOpenCL::loadLibraryFromText(const std::string& text,
                                                 const CompilerOptions& options,
                                                 bool retainBinary) const {
  auto ptr = std::make_shared<implementation::LibraryOpenCL>(*this);
  ptr->loadFromText(text, options);
  return ghost::Library(ptr);
}

ghost::Library DeviceOpenCL::loadLibraryFromData(const void* data, size_t len,
                                                 const CompilerOptions& options,
                                                 bool retainBinary) const {
  auto ptr = std::make_shared<implementation::LibraryOpenCL>(*this);
  ptr->loadFromData(data, len, options);
  return ghost::Library(ptr);
}

SharedContext DeviceOpenCL::shareContext() const {
  SharedContext c(context.get(), queue.get());
  return c;
}

ghost::Stream DeviceOpenCL::createStream(const StreamOptions& options) const {
  auto ptr = std::make_shared<implementation::StreamOpenCL>(*this, options);
  return ghost::Stream(ptr);
}

std::shared_ptr<CommandBuffer> DeviceOpenCL::createCommandBuffer(
    const CommandBufferOptions&) const {
  // The cb still records-and-replays for immediate submit; passing the device
  // lets compile() reach the cl_khr_command_buffer entry points to build a
  // native ExecutableOpenCL when supported.
  return std::make_shared<CommandBufferOpenCL>(this);
}

#if WITH_OPENCL_COMMAND_BUFFERS
const CommandBufferExtCL* DeviceOpenCL::commandBufferExt() const {
  if (!_cmdBufExtLoaded) {
    _cmdBufExtLoaded = true;
    if (checkExtension("cl_khr_command_buffer")) {
      cl_platform_id platform = getPlatform();
      auto load = [&](const char* name) {
        return clGetExtensionFunctionAddressForPlatform(platform, name);
      };
      _cmdBufExt.create =
          (clCreateCommandBufferKHR_fn)load("clCreateCommandBufferKHR");
      _cmdBufExt.finalize =
          (clFinalizeCommandBufferKHR_fn)load("clFinalizeCommandBufferKHR");
      _cmdBufExt.release =
          (clReleaseCommandBufferKHR_fn)load("clReleaseCommandBufferKHR");
      _cmdBufExt.enqueue =
          (clEnqueueCommandBufferKHR_fn)load("clEnqueueCommandBufferKHR");
      _cmdBufExt.ndrange =
          (ghost_clCommandNDRangeKernelKHR_fn)load("clCommandNDRangeKernelKHR");
      _cmdBufExt.copyBuffer =
          (ghost_clCommandCopyBufferKHR_fn)load("clCommandCopyBufferKHR");
      _cmdBufExt.fillBuffer =
          (ghost_clCommandFillBufferKHR_fn)load("clCommandFillBufferKHR");
      _cmdBufExt.barrier = (ghost_clCommandBarrierWithWaitListKHR_fn)load(
          "clCommandBarrierWithWaitListKHR");
      _cmdBufExt.copyImage =
          (ghost_clCommandCopyImageKHR_fn)load("clCommandCopyImageKHR");
      _cmdBufExt.copyBufferToImage =
          (ghost_clCommandCopyBufferToImageKHR_fn)load(
              "clCommandCopyBufferToImageKHR");
      _cmdBufExt.copyImageToBuffer =
          (ghost_clCommandCopyImageToBufferKHR_fn)load(
              "clCommandCopyImageToBufferKHR");
      // mutable_dispatch is a layered extension; only load its entry point
      // when the device advertises it.
      if (checkExtension("cl_khr_command_buffer_mutable_dispatch")) {
        _cmdBufExt.updateMutable =
            (clUpdateMutableCommandsKHR_fn)load("clUpdateMutableCommandsKHR");
        // Only enable our in-place arg patching if the device can actually
        // mutate kernel arguments (CL_MUTABLE_DISPATCH_ARGUMENTS_KHR).
        auto devices = getDevices();
        if (!devices.empty()) {
          cl_bitfield caps = 0;
          if (clGetDeviceInfo(devices[0],
                              CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR,
                              sizeof(caps), &caps, nullptr) == CL_SUCCESS)
            _cmdBufExt.mutableArgs =
                (caps & CL_MUTABLE_DISPATCH_ARGUMENTS_KHR) != 0;
        }
      }
    }
  }
  return _cmdBufExt.loaded() ? &_cmdBufExt : nullptr;
}

namespace {

// Whether a recorded sequence can be recorded into a cl_command_buffer_khr via
// the clCommand*KHR entry points. Dispatch / device-to-device buffer copy /
// fill / barrier are always recordable; device-to-device image copies require
// the (optional) image-command entry points. Host transfers (no clCommand*
// analogue), host-side image transfers, events and native interop fall back
// to command replay.
bool isClExecutable(const std::vector<Command>& commands,
                    const CommandBufferExtCL& ext) {
  for (const auto& command : commands) {
    bool ok = std::visit(
        [&](const auto& cmd) -> bool {
          using T = std::decay_t<decltype(cmd)>;
          if constexpr (std::is_same_v<T, DispatchCmd> ||
                        std::is_same_v<T, CopyBufferCmd> ||
                        std::is_same_v<T, FillBufferCmd> ||
                        std::is_same_v<T, FillBufferPatternCmd> ||
                        std::is_same_v<T, BarrierCmd>) {
            return true;
          } else if constexpr (std::is_same_v<T, CopyImageCmd> ||
                               std::is_same_v<T, CopyImageFromBufferCmd> ||
                               std::is_same_v<T, CopyImageToBufferCmd>) {
            return ext.images();
          } else {
            return false;
          }
        },
        command);
    if (!ok) return false;
  }
  return true;
}

// True when the recorded sequence is dispatch/barrier-only — the case where
// mutable_dispatch can patch every dispatch's args on update() without having
// to prove the non-dispatch (copy/fill) commands are unchanged.
bool isDispatchOnly(const std::vector<Command>& commands) {
  for (const auto& command : commands) {
    bool ok = std::visit(
        [](const auto& cmd) -> bool {
          using T = std::decay_t<decltype(cmd)>;
          return std::is_same_v<T, DispatchCmd> ||
                 std::is_same_v<T, BarrierCmd>;
        },
        command);
    if (!ok) return false;
  }
  return true;
}

// Backing storage + descriptors for one dispatch's mutable kernel args. Scalar
// arg_values point into the live Attribute (valid for the update() call);
// buffer/image cl_mem handles need a stable address, held in @c mems.
struct DispatchArgStorage {
  std::vector<cl_mem> mems;
  std::vector<cl_mutable_dispatch_arg_khr> args;
};

// Collect the cl_mem objects a command references, so they can be migrated onto
// the device before recording. Some runtimes (e.g. pocl's CPU driver) realize
// a buffer's device storage lazily and reject clCommand*KHR recording of a
// buffer that has never been used on the target device.
void collectMems(const Command& command, std::vector<cl_mem>& mems) {
  std::visit(
      [&](const auto& cmd) {
        using T = std::decay_t<decltype(cmd)>;
        if constexpr (std::is_same_v<T, DispatchCmd>) {
          for (const auto& a : cmd.args) {
            if (a.type() == Attribute::Type_Buffer)
              mems.push_back(
                  static_cast<BufferOpenCL*>(a.bufferImpl().get())->mem.get());
            else if (a.type() == Attribute::Type_Image)
              mems.push_back(
                  static_cast<ImageOpenCL*>(a.imageImpl().get())->mem.get());
            else if (a.type() == Attribute::Type_ArgumentBuffer &&
                     !a.argumentBuffer()->isStruct())
              mems.push_back(static_cast<BufferOpenCL*>(
                                 a.argumentBuffer()->bufferImpl().get())
                                 ->mem.get());
          }
        } else if constexpr (std::is_same_v<T, CopyBufferCmd>) {
          mems.push_back(static_cast<BufferOpenCL*>(
                             const_cast<implementation::Buffer*>(cmd.src.get()))
                             ->mem.get());
          mems.push_back(static_cast<BufferOpenCL*>(cmd.dst.get())->mem.get());
        } else if constexpr (std::is_same_v<T, FillBufferCmd> ||
                             std::is_same_v<T, FillBufferPatternCmd>) {
          mems.push_back(static_cast<BufferOpenCL*>(cmd.dst.get())->mem.get());
        } else if constexpr (std::is_same_v<T, CopyImageCmd>) {
          mems.push_back(static_cast<ImageOpenCL*>(
                             const_cast<implementation::Image*>(cmd.src.get()))
                             ->mem.get());
          mems.push_back(static_cast<ImageOpenCL*>(cmd.dst.get())->mem.get());
        } else if constexpr (std::is_same_v<T, CopyImageFromBufferCmd>) {
          mems.push_back(static_cast<BufferOpenCL*>(cmd.src.get())->mem.get());
          mems.push_back(static_cast<ImageOpenCL*>(cmd.dst.get())->mem.get());
        } else if constexpr (std::is_same_v<T, CopyImageToBufferCmd>) {
          mems.push_back(static_cast<ImageOpenCL*>(
                             const_cast<implementation::Image*>(cmd.src.get()))
                             ->mem.get());
          mems.push_back(static_cast<BufferOpenCL*>(cmd.dst.get())->mem.get());
        }
      },
      command);
}

// Build the mutable arg descriptor list for @p args, mirroring the index and
// size rules of FunctionOpenCL::bindArgs.
void fillMutableArgs(const std::vector<Attribute>& args,
                     DispatchArgStorage& st) {
  st.mems.reserve(args.size());  // stable addresses for &mems.back()
  cl_uint idx = 0;
  for (const auto& a : args) {
    switch (a.type()) {
      case Attribute::Type_Float: {
        size_t c = a.count();
        if (c == 3) c = 4;
        st.args.push_back({idx++, sizeof(float) * c, a.floatArray()});
        break;
      }
      case Attribute::Type_Int: {
        size_t c = a.count();
        if (c == 3) c = 4;
        st.args.push_back({idx++, sizeof(int32_t) * c, a.intArray()});
        break;
      }
      case Attribute::Type_UInt: {
        size_t c = a.count();
        if (c == 3) c = 4;
        st.args.push_back({idx++, sizeof(uint32_t) * c, a.uintArray()});
        break;
      }
      case Attribute::Type_Bool: {
        size_t c = a.count();
        if (c == 3) c = 4;
        st.args.push_back({idx++, sizeof(bool) * c, a.boolArray()});
        break;
      }
      case Attribute::Type_Buffer: {
        st.mems.push_back(
            static_cast<BufferOpenCL*>(a.bufferImpl().get())->mem.get());
        st.args.push_back({idx++, sizeof(cl_mem), &st.mems.back()});
        break;
      }
      case Attribute::Type_Image: {
        st.mems.push_back(
            static_cast<ImageOpenCL*>(a.imageImpl().get())->mem.get());
        st.args.push_back({idx++, sizeof(cl_mem), &st.mems.back()});
        break;
      }
      case Attribute::Type_ArgumentBuffer: {
        auto ab = a.argumentBuffer();
        if (ab->isStruct()) {
          st.args.push_back({idx++, ab->size(), ab->data()});
        } else {
          st.mems.push_back(
              static_cast<BufferOpenCL*>(ab->bufferImpl().get())->mem.get());
          st.args.push_back({idx++, sizeof(cl_mem), &st.mems.back()});
        }
        break;
      }
      case Attribute::Type_LocalMem:
        st.args.push_back({idx++, (size_t)a.asUInt(), nullptr});
        break;
      default:
        break;
    }
  }
}

}  // namespace
#endif  // WITH_OPENCL_COMMAND_BUFFERS

std::shared_ptr<RecordedCommandBuffer> CommandBufferOpenCL::cloneEmpty() const {
  return std::make_shared<CommandBufferOpenCL>(_device);
}

std::shared_ptr<Executable> CommandBufferOpenCL::compile(
    const CompileOptions& options) {
#if WITH_OPENCL_COMMAND_BUFFERS
  const CommandBufferExtCL* ext =
      _device ? _device->commandBufferExt() : nullptr;
  if (!ext || !isClExecutable(commands, *ext)) {
    if (options.requireAccelerated) throw ghost::unsupported_error();
    return RecordedCommandBuffer::compile(options);
  }
  return std::make_shared<ExecutableOpenCL>(_device, ext, commands);
#else
  // No native cl_khr_command_buffer (Apple, or disabled): fall back to
  // record-and-replay, which honors requireAccelerated via the base.
  return RecordedCommandBuffer::compile(options);
#endif
}

#if WITH_OPENCL_COMMAND_BUFFERS
ExecutableOpenCL::ExecutableOpenCL(const DeviceOpenCL* device,
                                   const CommandBufferExtCL* ext,
                                   std::vector<Command> commands)
    : _device(device), _ext(ext), _commands(std::move(commands)) {}

ExecutableOpenCL::~ExecutableOpenCL() { releaseCommandBuffer(); }

void ExecutableOpenCL::releaseCommandBuffer() {
  if (_cb) {
    _ext->release(_cb);
    _cb = nullptr;
    _builtForQueue = nullptr;
  }
  _mutableDispatches.clear();
  _mutable = false;
}

void ExecutableOpenCL::build(cl_command_queue queue) {
  releaseCommandBuffer();
  _mutableDispatches.clear();
  // Build mutable only for dispatch/barrier-only sequences, so update() can
  // patch every dispatch's args without proving copy/fill commands unchanged.
  _mutable = _ext->mutableDispatch() && isDispatchOnly(_commands);

  cl_int err = CL_SUCCESS;
  cl_command_buffer_properties_khr mutProps[] = {
      CL_COMMAND_BUFFER_FLAGS_KHR, CL_COMMAND_BUFFER_MUTABLE_KHR, 0};
  _cb = _ext->create(1, &queue, _mutable ? mutProps : nullptr, &err);
  checkError(err);

  // Materialize every referenced buffer/image on the target device before
  // recording: runtimes that allocate device storage lazily (pocl CPU) reject
  // recording a command against a buffer not yet resident on the device.
  std::vector<cl_mem> mems;
  for (auto& command : _commands) collectMems(command, mems);
  if (!mems.empty()) {
    checkError(clEnqueueMigrateMemObjects(queue, (cl_uint)mems.size(),
                                          mems.data(), 0, 0, nullptr, nullptr));
    checkError(clFinish(queue));
  }

  // Record each gated command. command_queue=NULL uses the cb's queue; NULL
  // sync-point lists leave commands unordered except across an explicit
  // barrier, matching Ghost's concurrent cb semantics.
  for (auto& command : _commands) {
    std::visit(
        [&](auto& cmd) {
          using T = std::decay_t<decltype(cmd)>;
          if constexpr (std::is_same_v<T, DispatchCmd>) {
            auto* fn = static_cast<FunctionOpenCL*>(cmd.function.get());
            fn->bindArgs(cmd.args);
            size_t global[3], local[3];
            for (int i = 0; i < 3; i++) {
              global[i] = cmd.launchArgs.global_size()[i];
              local[i] = cmd.launchArgs.local_size()[i];
            }
            bool localDefined = cmd.launchArgs.is_local_defined();
            cl_uint dims = (cl_uint)cmd.launchArgs.dims();
            cl_mutable_command_khr handle = nullptr;
            cl_properties ndProps[] = {CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR,
                                       CL_MUTABLE_DISPATCH_ARGUMENTS_KHR, 0};
            checkError(_ext->ndrange(_cb, queue, _mutable ? ndProps : nullptr,
                                     fn->kernel.get(), dims, nullptr, global,
                                     localDefined ? local : nullptr, 0, nullptr,
                                     nullptr, _mutable ? &handle : nullptr));
            if (_mutable) {
              MutableDispatch md{};
              md.handle = handle;
              md.kernel = fn->kernel.get();
              md.dims = dims;
              for (int i = 0; i < 3; i++) {
                md.global[i] = global[i];
                md.local[i] = local[i];
              }
              md.localDefined = localDefined;
              _mutableDispatches.push_back(md);
            }
          } else if constexpr (std::is_same_v<T, CopyBufferCmd>) {
            auto* dst = static_cast<BufferOpenCL*>(cmd.dst.get());
            auto* src = static_cast<BufferOpenCL*>(
                const_cast<implementation::Buffer*>(cmd.src.get()));
            checkError(_ext->copyBuffer(_cb, queue, nullptr, src->mem.get(),
                                        dst->mem.get(), cmd.srcOffset,
                                        cmd.dstOffset, cmd.bytes, 0, nullptr,
                                        nullptr, nullptr));
          } else if constexpr (std::is_same_v<T, FillBufferCmd>) {
            auto* dst = static_cast<BufferOpenCL*>(cmd.dst.get());
            checkError(_ext->fillBuffer(_cb, queue, nullptr, dst->mem.get(),
                                        &cmd.value, sizeof(cmd.value),
                                        cmd.offset, cmd.size, 0, nullptr,
                                        nullptr, nullptr));
          } else if constexpr (std::is_same_v<T, FillBufferPatternCmd>) {
            auto* dst = static_cast<BufferOpenCL*>(cmd.dst.get());
            checkError(_ext->fillBuffer(_cb, queue, nullptr, dst->mem.get(),
                                        cmd.pattern.data(), cmd.pattern.size(),
                                        cmd.offset, cmd.size, 0, nullptr,
                                        nullptr, nullptr));
          } else if constexpr (std::is_same_v<T, CopyImageCmd>) {
            auto* dst = static_cast<ImageOpenCL*>(cmd.dst.get());
            auto* src = static_cast<ImageOpenCL*>(
                const_cast<implementation::Image*>(cmd.src.get()));
            size_t srcO[3] = {0, 0, 0}, dstO[3] = {0, 0, 0};
            const auto& sz = dst->description().size;
            size_t region[3] = {sz.x, sz.y, sz.z};
            checkError(_ext->copyImage(_cb, queue, nullptr, src->mem.get(),
                                       dst->mem.get(), srcO, dstO, region, 0,
                                       nullptr, nullptr, nullptr));
          } else if constexpr (std::is_same_v<T, CopyImageFromBufferCmd>) {
            auto* dst = static_cast<ImageOpenCL*>(cmd.dst.get());
            auto* src = static_cast<BufferOpenCL*>(cmd.src.get());
            size_t dstO[3] = {0, 0, 0};
            size_t region[3] = {cmd.layout.size.x, cmd.layout.size.y,
                                cmd.layout.size.z};
            checkError(_ext->copyBufferToImage(
                _cb, queue, nullptr, src->mem.get(), dst->mem.get(), 0, dstO,
                region, 0, nullptr, nullptr, nullptr));
          } else if constexpr (std::is_same_v<T, CopyImageToBufferCmd>) {
            auto* src = static_cast<ImageOpenCL*>(
                const_cast<implementation::Image*>(cmd.src.get()));
            auto* dst = static_cast<BufferOpenCL*>(cmd.dst.get());
            size_t srcO[3] = {0, 0, 0};
            size_t region[3] = {cmd.layout.size.x, cmd.layout.size.y,
                                cmd.layout.size.z};
            checkError(_ext->copyImageToBuffer(
                _cb, queue, nullptr, src->mem.get(), dst->mem.get(), srcO,
                region, 0, 0, nullptr, nullptr, nullptr));
          } else if constexpr (std::is_same_v<T, BarrierCmd>) {
            checkError(_ext->barrier(_cb, queue, nullptr, 0, nullptr, nullptr,
                                     nullptr));
          }
        },
        command);
  }

  checkError(_ext->finalize(_cb));
  _builtForQueue = queue;
}

void ExecutableOpenCL::submit(const ghost::Stream& stream) {
  auto* s = static_cast<StreamOpenCL*>(stream.impl().get());
  cl_command_queue queue = s->queue.get();
  // The native command buffer is tied to the queue it was created against;
  // (re)build if this is the first submit or the target queue changed.
  if (!_cb || queue != _builtForQueue) build(queue);
  checkError(
      _ext->enqueue(0, nullptr, _cb, s->events.size(), s->events, s->event()));
  s->addEvent();
}

bool ExecutableOpenCL::tryMutableUpdate(
    const std::vector<Command>& newCommands) {
  if (!_mutable || !_cb) return false;
  if (newCommands.size() != _commands.size()) return false;

  // Reserve so emplace_back never reallocates: configs hold pointers into
  // each storage's arg list (and each storage's mems), which must stay stable
  // through the clUpdateMutableCommandsKHR call below.
  std::vector<DispatchArgStorage> storages;
  std::vector<cl_mutable_dispatch_config_khr> configs;
  storages.reserve(_mutableDispatches.size());
  configs.reserve(_mutableDispatches.size());

  size_t di = 0;
  for (size_t i = 0; i < newCommands.size(); i++) {
    if (newCommands[i].index() != _commands[i].index()) return false;
    if (const auto* d = std::get_if<DispatchCmd>(&newCommands[i])) {
      if (di >= _mutableDispatches.size()) return false;
      const MutableDispatch& md = _mutableDispatches[di];
      auto* fn = static_cast<FunctionOpenCL*>(d->function.get());
      // Only ARGUMENTS are updatable, so kernel and work geometry must match.
      if (fn->kernel.get() != md.kernel) return false;
      if ((cl_uint)d->launchArgs.dims() != md.dims) return false;
      if (d->launchArgs.is_local_defined() != md.localDefined) return false;
      for (int k = 0; k < 3; k++) {
        if (d->launchArgs.global_size()[k] != md.global[k]) return false;
        if (d->launchArgs.local_size()[k] != md.local[k]) return false;
      }
      storages.emplace_back();
      fillMutableArgs(d->args, storages.back());
      cl_mutable_dispatch_config_khr cfg = {};
      cfg.command = md.handle;
      cfg.num_args = (cl_uint)storages.back().args.size();
      cfg.arg_list = storages.back().args.data();
      configs.push_back(cfg);
      di++;
    }
    // BarrierCmd carries nothing to patch; its position already matched.
  }
  if (di != _mutableDispatches.size()) return false;

  // Finalized clUpdateMutableCommandsKHR takes parallel type + pointer arrays.
  // configs is fully populated and reserved, so &cfg stays stable here.
  std::vector<cl_command_buffer_update_type_khr> types;
  std::vector<const void*> configPtrs;
  types.reserve(configs.size());
  configPtrs.reserve(configs.size());
  for (auto& cfg : configs) {
    types.push_back(CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR);
    configPtrs.push_back(&cfg);
  }
  return _ext->updateMutable(_cb, (cl_uint)configs.size(), types.data(),
                             configPtrs.data()) == CL_SUCCESS;
}

void ExecutableOpenCL::update(const std::vector<Command>& commands) {
  // Fast path: patch recorded dispatch args in place (mutable_dispatch).
  if (tryMutableUpdate(commands)) {
    _commands = commands;
    _lastPatched = true;
    return;
  }
  // Otherwise rebuild against the queue on next submit.
  _lastPatched = false;
  _commands = commands;
  releaseCommandBuffer();
}
#endif  // WITH_OPENCL_COMMAND_BUFFERS

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

ghost::Buffer DeviceOpenCL::allocateBuffer(size_t bytes,
                                           const BufferOptions& opts) const {
  // Use the recycling pool for Default/Transient. Persistent and Staging
  // bypass the pool — Persistent avoids fragmenting long-lived memory, and
  // Staging needs host-visible flags that the pool doesn't track.
  if (_pool && _pool->getLimit() > 0 && opts.hint != AllocHint::Persistent &&
      opts.hint != AllocHint::Staging) {
    auto mem = _pool->lookupBuffer(bytes);
    if (mem.get()) {
      return ghost::Buffer(
          std::make_shared<PooledBufferOpenCL>(std::move(mem), bytes, _pool));
    }
    // Allocate new, but wrap in pooled buffer for recycling on destruction.
    cl_int err;
    cl_mem_flags flags = getMemFlags(opts.access);
    auto newMem = opencl::ptr<cl_mem>(
        clCreateBuffer(context, flags, bytes, nullptr, &err));
    checkError(err);
    _pool->reserve(bytes);
    return ghost::Buffer(
        std::make_shared<PooledBufferOpenCL>(std::move(newMem), bytes, _pool));
  }
  if (auto* a = allocator()) {
    if (void* handle = a->allocateBuffer(bytes, opts)) {
      // Allocator owns this handle and reclaims it in freeBuffer() via
      // mem.release() (no clRelease). Adopt it without retaining.
      opencl::ptr<cl_mem> mem(reinterpret_cast<cl_mem>(handle));
      auto ptr = std::make_shared<implementation::BufferOpenCL>(mem, bytes);
      ptr->setAllocator(a);
      return ghost::Buffer(ptr);
    }
  }
  auto ptr = std::make_shared<implementation::BufferOpenCL>(*this, bytes, opts);
  return ghost::Buffer(ptr);
}

ghost::MappedBuffer DeviceOpenCL::allocateMappedBuffer(
    size_t bytes, const BufferOptions& opts) const {
  if (auto* a = allocator()) {
    if (void* handle = a->allocateMappedBuffer(bytes, opts)) {
      // Allocator-owned; adopt without retaining (reclaimed via mem.release()).
      opencl::ptr<cl_mem> mem(reinterpret_cast<cl_mem>(handle));
      auto ptr = std::make_shared<implementation::MappedBufferOpenCL>(
          mem, bytes, bytes);
      ptr->setAllocator(a);
      return ghost::MappedBuffer(ptr);
    }
  }
  auto ptr =
      std::make_shared<implementation::MappedBufferOpenCL>(*this, bytes, opts);
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
  if (auto* a = allocator()) {
    if (void* handle = a->allocateImage(descr)) {
      // Allocator-owned; adopt without retaining (reclaimed via mem.release()).
      opencl::ptr<cl_mem> mem(reinterpret_cast<cl_mem>(handle));
      auto ptr = std::make_shared<implementation::ImageOpenCL>(mem, descr);
      ptr->setAllocator(a);
      return ghost::Image(ptr);
    }
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

ghost::Buffer DeviceOpenCL::wrapBuffer(const SharedBuffer& shared) const {
  // Host retains ownership. retainObject=true makes opencl::ptr call
  // clRetainMemObject on construct and clReleaseMemObject on destroy:
  // balanced, host's count unchanged.
  opencl::ptr<cl_mem> mem(reinterpret_cast<cl_mem>(shared.handle),
                          /*retainObject=*/true);
  auto ptr = std::make_shared<implementation::BufferOpenCL>(mem, shared.bytes);
  return ghost::Buffer(ptr);
}

ghost::Image DeviceOpenCL::wrapImage(const SharedImage& shared) const {
  // Host retains ownership; retain+release to leave its count unchanged.
  opencl::ptr<cl_mem> mem(reinterpret_cast<cl_mem>(shared.handle),
                          /*retainObject=*/true);
  auto ptr = std::make_shared<implementation::ImageOpenCL>(mem, shared.descr);
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
    case kDeviceMaxImageAlignment: {
      // CL_DEVICE_IMAGE_PITCH_ALIGNMENT returns pixels; convert to bytes
      // using RGBA Float (largest common pixel format) for a conservative
      // alignment that works for any format.
      uint64_t pixels = getInt(CL_DEVICE_IMAGE_PITCH_ALIGNMENT);
      const size_t kRefPixelSize = 16;  // RGBA Float32
      return (uint64_t)(pixels * kRefPixelSize);
    }
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
    case kDeviceSupportsProgramGlobals:
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
    case kDeviceSupportsCooperativeMatrix:
      return false;
    default:
      return Attribute();
  }
}

size_t DeviceOpenCL::imageAlignment(const ImageDescription& descr) const {
  // CL_DEVICE_IMAGE_PITCH_ALIGNMENT is in pixels; convert to bytes using
  // the actual pixel size for the requested format.
  uint64_t pixels = getInt(CL_DEVICE_IMAGE_PITCH_ALIGNMENT);
  size_t pxSize = descr.pixelSize();
  if (pxSize == 0) pxSize = 16;  // fallback: RGBA Float32
  size_t v = static_cast<size_t>(pixels) * pxSize;
  return v > 0 ? v : 1;
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
  cl_ulong v = 0;
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
