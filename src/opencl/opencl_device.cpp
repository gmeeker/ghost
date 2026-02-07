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
  /*
  cl_event ev = queue;
  cl_int err = clWaitForEvents(1, &ev);
  checkError(err);
  */
}

void StreamOpenCL::addEvent() {
  if (outOfOrder) {
    events.reset();
    events.push(lastEvent);
  }
}

cl_event* StreamOpenCL::event() { return outOfOrder ? &lastEvent : nullptr; }

BufferOpenCL::BufferOpenCL(opencl::ptr<cl_mem> mem_) : mem(mem_) {}

BufferOpenCL::BufferOpenCL(const DeviceOpenCL& dev, size_t bytes,
                           Access access) {
  cl_int err;
  cl_mem_flags flags = getMemFlags(access);
  mem = opencl::ptr<cl_mem>(
      clCreateBuffer(dev.context, flags, bytes, nullptr, &err));
  checkError(err);
}

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

MappedBufferOpenCL::MappedBufferOpenCL(opencl::ptr<cl_mem> mem_, size_t bytes)
    : BufferOpenCL(mem_), length(bytes), ptr(nullptr) {}

MappedBufferOpenCL::MappedBufferOpenCL(const DeviceOpenCL& dev, size_t bytes,
                                       Access access)
    : BufferOpenCL(opencl::ptr<cl_mem>()), length(bytes), ptr(nullptr) {
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
        devices.resize(size_t(num));
        if (!devices.empty()) {
          err = clGetDeviceIDs(platform, deviceType, num, &devices[0], nullptr);
          checkError(err);
        }
        if (devices.size() > 1) {
          devices.resize(1);
        }
      }
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
  if (!queue) {
    implementation::StreamOpenCL stream(*this);
    queue = stream.queue;
  }
  _version = getString(CL_DEVICE_VERSION);
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

size_t DeviceOpenCL::getMemoryPoolSize() const {}

void DeviceOpenCL::setMemoryPoolSize(size_t bytes) {}

ghost::Buffer DeviceOpenCL::allocateBuffer(size_t bytes, Access access) const {
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
      return Attribute("OpenCL");
    case kDeviceName:
      return Attribute(getString(CL_DEVICE_NAME));
    case kDeviceVendor:
      return Attribute(getString(CL_DEVICE_VENDOR));
    case kDeviceDriverVersion:
      return Attribute(getString(CL_DRIVER_VERSION));
    case kDeviceCount:
      return Attribute((int32_t)getDevices().size());
    case kDeviceSupportsMappedBuffer:
      return Attribute(true);
    case kDeviceSupportsProgramConstants:
      return Attribute(false);
    default:
      return Attribute();
  }
}

bool DeviceOpenCL::checkVersion(const std::string& version) const {
  return strcmp(_version.c_str(), version.c_str()) >= 0;
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
}  // namespace ghost
#endif
