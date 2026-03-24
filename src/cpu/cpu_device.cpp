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

#include <ghost/cpu/device.h>
#include <ghost/cpu/impl_device.h>
#include <ghost/cpu/impl_function.h>
#include <string.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

#include <algorithm>
#include <limits>

namespace ghost {
namespace implementation {
void ThreadPoolDefault::worker() {
#if GHOST_USE_STD_THREAD
  bool working = true;
  while (working) {
    ThreadWork w;
    {
      std::unique_lock<std::mutex> lk(mutex);
      cv.wait(lk, [this] { return !work.empty(); });
      lk.release();
      std::lock_guard<std::mutex> guard(mutex, std::adopt_lock_t());

      w = work.front();
      work.pop();
      if (work.empty()) cv.notify_all();
    }
    if (w.quit) {
      working = false;
    } else {
      w.function(w.i, w.count, w.args);
    }
  }
#endif
}

ThreadPoolDefault::ThreadPoolDefault(const DeviceCPU& dev_) : dev(dev_) {
#if GHOST_USE_STD_THREAD
  for (size_t i = 0; i < dev.cores; i++) {
#if __cplusplus >= 201703L
    threads.emplace_back(&ThreadPoolDefault::worker, this);
#else
    threads.push_back(std::thread(&ThreadPoolDefault::worker, this));
#endif
  }
#elif defined(__APPLE_CC__)
  queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
  group = dispatch_group_create();
#endif
}

ThreadPoolDefault::~ThreadPoolDefault() {
  sync();
#if GHOST_USE_STD_THREAD
  {
    std::lock_guard<std::mutex> guard(mutex);
    for (auto i = threads.begin(); i != threads.end(); ++i) {
      ThreadWork w = {nullptr, std::vector<Attribute>(), 0, 1, false};
      work.push(w);
    }
  }
  cv.notify_all();
  for (auto i = threads.begin(); i != threads.end(); ++i) {
    i->join();
  }
#elif defined(__APPLE_CC__)
  dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
  dispatch_release(group);
#endif
}

void ThreadPoolDefault::thread(size_t count, FunctionCPU::Type function,
                               const std::vector<Attribute>& args) {
  if (count == 1) {
    function(0, 1, args);
  } else if (count > 1) {
#if GHOST_USE_STD_THREAD
    std::lock_guard<std::mutex> guard(mutex);
    for (size_t i = 0; i < count; i++) {
      ThreadWork w = {function, args, i, count, false};
      work.push(w);
    }
    cv.notify_all();
#elif __APPLE_CC__
    for (size_t i = 0; i < count; i++) {
      __block auto a = args;
      dispatch_group_async(group, queue, ^{
        function(i, count, a);
      });
    }
#endif
  }
}

void ThreadPoolDefault::sync() {
#if GHOST_USE_STD_THREAD
  std::unique_lock<std::mutex> lk(mutex);
  cv.wait(lk, [this] { return work.empty(); });
#elif __APPLE_CC__
  dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
#endif
}

EventCPU::EventCPU() {}

void EventCPU::wait() {}

bool EventCPU::isComplete() const { return true; }

StreamCPU::StreamCPU(std::shared_ptr<ThreadPool> pool_) : pool(pool_) {}

StreamCPU::~StreamCPU() {}

void StreamCPU::sync() { pool->sync(); }

std::shared_ptr<Event> StreamCPU::record() {
  pool->sync();
  return std::make_shared<EventCPU>();
}

void StreamCPU::waitForEvent(const std::shared_ptr<Event>& e) { e->wait(); }

BufferCPU::BufferCPU(void* ptr_, size_t bytes) : ptr(ptr_), _size(bytes) {}

BufferCPU::BufferCPU(const DeviceCPU& dev, size_t bytes) : _size(bytes) {
  ptr = dev.allocateHostMemory(bytes);
}

size_t BufferCPU::size() const { return _size; }

void BufferCPU::copy(const ghost::Stream& s, const ghost::Buffer& src,
                     size_t bytes) {
  memcpy(ptr, static_cast<const BufferCPU*>(src.impl().get())->ptr, bytes);
}

void BufferCPU::copy(const ghost::Stream& s, const void* src, size_t bytes) {
  memcpy(ptr, src, bytes);
}

void BufferCPU::copyTo(const ghost::Stream& s, void* dst, size_t bytes) const {
  memcpy(dst, ptr, bytes);
}

void BufferCPU::copy(const ghost::Stream& s, const ghost::Buffer& src,
                     size_t srcOffset, size_t dstOffset, size_t bytes) {
  auto srcPtr = static_cast<const BufferCPU*>(src.impl().get())->ptr;
  memcpy(static_cast<uint8_t*>(ptr) + dstOffset,
         static_cast<const uint8_t*>(srcPtr) + srcOffset, bytes);
}

void BufferCPU::copy(const ghost::Stream& s, const void* src, size_t dstOffset,
                     size_t bytes) {
  memcpy(static_cast<uint8_t*>(ptr) + dstOffset, src, bytes);
}

void BufferCPU::copyTo(const ghost::Stream& s, void* dst, size_t srcOffset,
                       size_t bytes) const {
  memcpy(dst, static_cast<const uint8_t*>(ptr) + srcOffset, bytes);
}

void BufferCPU::fill(const ghost::Stream& s, size_t offset, size_t size,
                     uint8_t value) {
  memset(static_cast<uint8_t*>(ptr) + offset, value, size);
}

void BufferCPU::fill(const ghost::Stream& s, size_t offset, size_t size,
                     const void* pattern, size_t patternSize) {
  uint8_t* dst = static_cast<uint8_t*>(ptr) + offset;
  if (patternSize == 1) {
    memset(dst, *static_cast<const uint8_t*>(pattern), size);
  } else {
    for (size_t i = 0; i < size; i += patternSize) {
      size_t n = std::min(patternSize, size - i);
      memcpy(dst + i, pattern, n);
    }
  }
}

std::shared_ptr<Buffer> BufferCPU::createSubBuffer(
    const std::shared_ptr<Buffer>& self, size_t offset, size_t size) {
  return std::make_shared<SubBufferCPU>(
      self, static_cast<uint8_t*>(ptr) + offset, size);
}

SubBufferCPU::SubBufferCPU(std::shared_ptr<Buffer> parent, void* ptr_,
                           size_t bytes)
    : BufferCPU(ptr_, bytes), _parent(parent) {}

ImageCPU::ImageCPU(const DeviceCPU& dev, const ImageDescription& descr_)
    : descr(descr_) {}

ImageCPU::ImageCPU(const DeviceCPU& dev, const ImageDescription& descr_,
                   BufferCPU& buffer)
    : descr(descr_) {}

ImageCPU::ImageCPU(const DeviceCPU& dev, const ImageDescription& descr_,
                   ImageCPU& image)
    : descr(descr_) {}

void ImageCPU::copy(const ghost::Stream& s, const ghost::Image& src) {}

void ImageCPU::copy(const ghost::Stream& s, const ghost::Buffer& src,
                    const ImageDescription& descr) {}

void ImageCPU::copy(const ghost::Stream& s, const void* src,
                    const ImageDescription& descr) {}

void ImageCPU::copyTo(const ghost::Stream& s, ghost::Buffer& dst,
                      const ImageDescription& descr) const {}

void ImageCPU::copyTo(const ghost::Stream& s, void* dst,
                      const ImageDescription& descr) const {}

DeviceCPU::DeviceCPU(const SharedContext& share) : cores(getNumberOfCores()) {}

ghost::Library DeviceCPU::loadLibraryFromText(
    const std::string& text, const std::string& options) const {
  throw ghost::unsupported_error();
}

ghost::Library DeviceCPU::loadLibraryFromData(
    const void* data, size_t len, const std::string& options) const {
  throw ghost::unsupported_error();
}

ghost::Library DeviceCPU::loadLibraryFromFile(
    const std::string& filename) const {
  auto ptr = std::make_shared<implementation::LibraryCPU>(*this);
  ptr->loadFromFile(filename);
  return ghost::Library(ptr);
}

SharedContext DeviceCPU::shareContext() const {
  SharedContext c;
  return c;
}

ghost::Stream DeviceCPU::createStream() const {
  auto ptr = std::make_shared<implementation::StreamCPU>(
      std::make_shared<ThreadPoolDefault>(*this));
  return ghost::Stream(ptr);
}

size_t DeviceCPU::getMemoryPoolSize() const {
  return Device::getMemoryPoolSize();
}

void DeviceCPU::setMemoryPoolSize(size_t bytes) {
  Device::setMemoryPoolSize(bytes);
}

ghost::Buffer DeviceCPU::allocateBuffer(size_t bytes, Access) const {
  auto ptr = std::make_shared<implementation::BufferCPU>(*this, bytes);
  return ghost::Buffer(ptr);
}

ghost::MappedBuffer DeviceCPU::allocateMappedBuffer(size_t, Access) const {
  throw ghost::unsupported_error();
}

ghost::Image DeviceCPU::allocateImage(const ImageDescription& descr) const {
  auto ptr = std::make_shared<implementation::ImageCPU>(*this, descr);
  return ghost::Image(ptr);
}

ghost::Image DeviceCPU::sharedImage(const ImageDescription& descr,
                                    ghost::Buffer& buffer) const {
  auto b = static_cast<implementation::BufferCPU*>(buffer.impl().get());
  auto ptr = std::make_shared<implementation::ImageCPU>(*this, descr, *b);
  return ghost::Image(ptr);
}

ghost::Image DeviceCPU::sharedImage(const ImageDescription& descr,
                                    ghost::Image& image) const {
  auto i = static_cast<implementation::ImageCPU*>(image.impl().get());
  auto ptr = std::make_shared<implementation::ImageCPU>(*this, descr, *i);
  return ghost::Image(ptr);
}

Attribute DeviceCPU::getAttribute(DeviceAttributeId what) const {
  switch (what) {
    case kDeviceImplementation:
      return "CPU";
    case kDeviceName:
      // TODO
      return "";
    case kDeviceVendor:
      // TODO
      return "";
    case kDeviceDriverVersion:
      return "";
    case kDeviceCount:
      return 1;
    case kDeviceProcessorCount:
      return (uint32_t)getNumberOfCores();
    case kDeviceUnifiedMemory:
      return true;
    case kDeviceMemory:
      // TODO
      return 0;
    case kDeviceLocalMemory:
      return 0;
    case kDeviceMaxThreads:
      return 1024;
    case kDeviceMaxWorkSize:
      return Attribute(1024, 1024, 1);
    case kDeviceMaxRegisters:
      return 0;
    case kDeviceMaxImageSize1:
      return std::numeric_limits<int32_t>::max();
    case kDeviceMaxImageSize2: {
      auto v = std::numeric_limits<int32_t>::max();
      return Attribute(v, v);
    }
    case kDeviceMaxImageSize3: {
      auto v = std::numeric_limits<int32_t>::max();
      return Attribute(v, v, v);
    }
    case kDeviceImageAlignment:
      return 64;
    case kDeviceSupportsImageIntegerFiltering:
      return false;
    case kDeviceSupportsImageFloatFiltering:
      return false;
    case kDeviceSupportsMappedBuffer:
      return false;
    case kDeviceSupportsProgramConstants:
      return false;
    case kDeviceSupportsSubgroup:
      return true;
    case kDeviceSupportsSubgroupShuffle:
      return true;
    case kDeviceSubgroupWidth:
      return 16;
    case kDeviceMaxComputeUnits:
      return (uint32_t)getNumberOfCores();
    case kDeviceMemoryAlignment:
      return (uint32_t)alignof(std::max_align_t);
    case kDeviceBufferAlignment:
      return (uint32_t)alignof(std::max_align_t);
    case kDeviceMaxBufferSize:
      return (uint64_t)std::numeric_limits<size_t>::max();
    case kDeviceMaxConstantBufferSize:
      return (uint64_t)std::numeric_limits<size_t>::max();
    case kDeviceTimestampPeriod:
      return 1.0f;
    case kDeviceSupportsProfilingTimer:
      return false;
    default:
      return Attribute();
  }
}
}  // namespace implementation

DeviceCPU::DeviceCPU(const SharedContext& share)
    : Device(std::make_shared<implementation::DeviceCPU>(share)) {}

DeviceCPU::DeviceCPU(const GpuInfo&)
    : Device(std::make_shared<implementation::DeviceCPU>(SharedContext())) {}

std::vector<GpuInfo> DeviceCPU::enumerateDevices() {
  std::vector<GpuInfo> result;
  GpuInfo info;
  info.name = "CPU";
  info.vendor = "";
  info.implementation = "CPU";
  info.memory = 0;
  info.unifiedMemory = true;
  info.index = 0;

#if defined(__APPLE__)
  int64_t memSize = 0;
  size_t len = sizeof(memSize);
  if (sysctlbyname("hw.memsize", &memSize, &len, nullptr, 0) == 0)
    info.memory = (uint64_t)memSize;
#elif defined(_WIN32)
  MEMORYSTATUSEX statex;
  statex.dwLength = sizeof(statex);
  if (GlobalMemoryStatusEx(&statex)) info.memory = statex.ullTotalPhys;
#elif defined(__linux__)
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  if (pages > 0 && page_size > 0)
    info.memory = (uint64_t)pages * (uint64_t)page_size;
#endif

  result.push_back(info);
  return result;
}
}  // namespace ghost
