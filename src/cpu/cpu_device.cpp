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
#include <sys/types.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#include <unistd.h>

#include <fstream>
#elif defined(_WIN32)
#include <windows.h>
#if defined(_M_IX86) || defined(_M_X64)
#include <intrin.h>
#endif
#endif

#include <algorithm>
#include <limits>
#include <string>

namespace {

std::string getCPUName() {
#if defined(__APPLE__)
  char name[256] = {};
  size_t len = sizeof(name);
  if (sysctlbyname("machdep.cpu.brand_string", name, &len, nullptr, 0) == 0)
    return name;
  return "CPU";
#elif defined(__linux__)
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpuinfo, line)) {
    if (line.rfind("model name", 0) == 0) {
      auto pos = line.find(':');
      if (pos != std::string::npos) {
        auto start = line.find_first_not_of(" \t", pos + 1);
        if (start != std::string::npos) return line.substr(start);
      }
    }
  }
  return "CPU";
#elif defined(_WIN32)
  // Try the registry first (works on both x86 and ARM).
  HKEY key;
  if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                    "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0,
                    KEY_READ, &key) == ERROR_SUCCESS) {
    char name[256] = {};
    DWORD size = sizeof(name);
    DWORD type = 0;
    if (RegQueryValueExA(key, "ProcessorNameString", nullptr, &type,
                         reinterpret_cast<LPBYTE>(name),
                         &size) == ERROR_SUCCESS &&
        type == REG_SZ) {
      RegCloseKey(key);
      std::string result(name);
      auto start = result.find_first_not_of(' ');
      return start != std::string::npos ? result.substr(start) : "CPU";
    }
    RegCloseKey(key);
  }
#if defined(_M_IX86) || defined(_M_X64)
  {
    int cpuInfo[4] = {};
    __cpuid(cpuInfo, 0x80000002);
    char brand[49] = {};
    memcpy(brand, cpuInfo, sizeof(cpuInfo));
    __cpuid(cpuInfo, 0x80000003);
    memcpy(brand + 16, cpuInfo, sizeof(cpuInfo));
    __cpuid(cpuInfo, 0x80000004);
    memcpy(brand + 32, cpuInfo, sizeof(cpuInfo));
    brand[48] = '\0';
    std::string result(brand);
    auto start = result.find_first_not_of(' ');
    return start != std::string::npos ? result.substr(start) : "CPU";
  }
#endif
  return "CPU";
#else
  return "CPU";
#endif
}

std::string getCPUVendor() {
#if defined(__APPLE__)
  char vendor[256] = {};
  size_t len = sizeof(vendor);
  if (sysctlbyname("machdep.cpu.vendor", vendor, &len, nullptr, 0) == 0)
    return vendor;
  // Apple Silicon doesn't have machdep.cpu.vendor
  return "Apple";
#elif defined(__linux__)
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpuinfo, line)) {
    if (line.rfind("vendor_id", 0) == 0) {
      auto pos = line.find(':');
      if (pos != std::string::npos) {
        auto start = line.find_first_not_of(" \t", pos + 1);
        if (start != std::string::npos) return line.substr(start);
      }
    }
  }
  return "";
#elif defined(_WIN32)
  // Try the registry first (works on both x86 and ARM).
  HKEY key;
  if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                    "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0,
                    KEY_READ, &key) == ERROR_SUCCESS) {
    char vendor[256] = {};
    DWORD size = sizeof(vendor);
    DWORD type = 0;
    if (RegQueryValueExA(key, "VendorIdentifier", nullptr, &type,
                         reinterpret_cast<LPBYTE>(vendor),
                         &size) == ERROR_SUCCESS &&
        type == REG_SZ) {
      RegCloseKey(key);
      return vendor;
    }
    RegCloseKey(key);
  }
#if defined(_M_IX86) || defined(_M_X64)
  {
    int cpuInfo[4] = {};
    __cpuid(cpuInfo, 0);
    char vendor[13] = {};
    memcpy(vendor, &cpuInfo[1], 4);
    memcpy(vendor + 4, &cpuInfo[3], 4);
    memcpy(vendor + 8, &cpuInfo[2], 4);
    vendor[12] = '\0';
    return vendor;
  }
#endif
  return "";
#else
  return "";
#endif
}

uint64_t getSystemMemory() {
#if defined(__APPLE__)
  uint64_t mem = 0;
  size_t len = sizeof(mem);
  if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) return mem;
  return 0;
#elif defined(__linux__)
  struct sysinfo si;
  if (sysinfo(&si) == 0)
    return static_cast<uint64_t>(si.totalram) * si.mem_unit;
  return 0;
#elif defined(_WIN32)
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  if (GlobalMemoryStatusEx(&status)) return status.ullTotalPhys;
  return 0;
#else
  return 0;
#endif
}

}  // namespace

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
      ThreadWork w = {nullptr, std::vector<Attribute>(), 0, 1, true};
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

namespace {
size_t alignUp(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

size_t imageRowBytes(const ImageDescription& d, size_t alignment) {
  size_t raw = d.stride.x > 0 ? static_cast<size_t>(d.stride.x)
                              : d.size.x * d.pixelSize();
  return alignUp(raw, alignment);
}

size_t imageDepthBytes(const ImageDescription& d, size_t rowBytes) {
  if (d.stride.y > 0) return static_cast<size_t>(d.stride.y);
  return d.size.y * rowBytes;
}

size_t srcRowBytes(const ImageDescription& d) {
  return d.stride.x > 0 ? static_cast<size_t>(d.stride.x)
                        : d.size.x * d.pixelSize();
}

size_t srcDepthBytes(const ImageDescription& d, size_t rowBytes) {
  if (d.stride.y > 0) return static_cast<size_t>(d.stride.y);
  return d.size.y * rowBytes;
}

void copyImageData(void* dst, size_t dstRow, size_t dstDepth, const void* src,
                   size_t srcRow, size_t srcDepth, const ImageDescription& d) {
  size_t copyWidth = std::min(srcRow, dstRow);
  if (srcRow == dstRow && srcDepth == dstDepth) {
    memcpy(dst, src, d.size.z * dstDepth);
  } else {
    auto d8 = static_cast<uint8_t*>(dst);
    auto s8 = static_cast<const uint8_t*>(src);
    for (size_t z = 0; z < d.size.z; z++) {
      for (size_t y = 0; y < d.size.y; y++) {
        memcpy(d8 + z * dstDepth + y * dstRow, s8 + z * srcDepth + y * srcRow,
               copyWidth);
      }
    }
  }
}
}  // namespace

ImageCPU::ImageCPU(const DeviceCPU& dev, const ImageDescription& descr_)
    : descr(descr_) {
  constexpr size_t kAlignment = 64;
  rowBytes = imageRowBytes(descr, kAlignment);
  depthBytes = imageDepthBytes(descr, rowBytes);
  size_t total = descr.size.z * depthBytes;
  data = dev.allocateHostMemory(total);
  memset(data, 0, total);
}

ImageCPU::ImageCPU(const DeviceCPU& dev, const ImageDescription& descr_,
                   BufferCPU& buffer)
    : descr(descr_), data(buffer.ptr) {
  rowBytes = srcRowBytes(descr);
  depthBytes = srcDepthBytes(descr, rowBytes);
}

ImageCPU::ImageCPU(const DeviceCPU& dev, const ImageDescription& descr_,
                   ImageCPU& image)
    : descr(descr_),
      data(image.data),
      rowBytes(image.rowBytes),
      depthBytes(image.depthBytes) {}

void ImageCPU::copy(const ghost::Stream& s, const ghost::Image& src) {
  auto srcImg = static_cast<const ImageCPU*>(src.impl().get());
  copyImageData(data, rowBytes, depthBytes, srcImg->data, srcImg->rowBytes,
                srcImg->depthBytes, descr);
}

void ImageCPU::copy(const ghost::Stream& s, const ghost::Buffer& src,
                    const ImageDescription& srcDescr) {
  auto srcBuf = static_cast<const BufferCPU*>(src.impl().get());
  size_t sRow = srcRowBytes(srcDescr);
  size_t sDepth = srcDepthBytes(srcDescr, sRow);
  copyImageData(data, rowBytes, depthBytes, srcBuf->ptr, sRow, sDepth, descr);
}

void ImageCPU::copy(const ghost::Stream& s, const void* src,
                    const ImageDescription& srcDescr) {
  size_t sRow = srcRowBytes(srcDescr);
  size_t sDepth = srcDepthBytes(srcDescr, sRow);
  copyImageData(data, rowBytes, depthBytes, src, sRow, sDepth, descr);
}

void ImageCPU::copyTo(const ghost::Stream& s, ghost::Buffer& dst,
                      const ImageDescription& dstDescr) const {
  auto dstBuf = static_cast<BufferCPU*>(dst.impl().get());
  size_t dRow = srcRowBytes(dstDescr);
  size_t dDepth = srcDepthBytes(dstDescr, dRow);
  copyImageData(dstBuf->ptr, dRow, dDepth, data, rowBytes, depthBytes, descr);
}

void ImageCPU::copyTo(const ghost::Stream& s, void* dst,
                      const ImageDescription& dstDescr) const {
  size_t dRow = srcRowBytes(dstDescr);
  size_t dDepth = srcDepthBytes(dstDescr, dRow);
  copyImageData(dst, dRow, dDepth, data, rowBytes, depthBytes, descr);
}

DeviceCPU::DeviceCPU(const SharedContext& share) : cores(getNumberOfCores()) {}

DeviceCPU::DeviceCPU(const GpuInfo&) : cores(getNumberOfCores()) {}

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
      return getCPUName();
    case kDeviceVendor:
      return getCPUVendor();
    case kDeviceDriverVersion:
      return "";
    case kDeviceCount:
      return 1;
    case kDeviceProcessorCount:
      return (uint32_t)getNumberOfCores();
    case kDeviceUnifiedMemory:
      return true;
    case kDeviceMemory:
      return getSystemMemory();
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
    : Device(std::make_shared<implementation::DeviceCPU>(share)) {
  auto cpu = static_cast<implementation::DeviceCPU*>(impl().get());
  setDefaultStream(std::make_shared<implementation::StreamCPU>(
      std::make_shared<implementation::ThreadPoolDefault>(*cpu)));
}

DeviceCPU::DeviceCPU(const GpuInfo& info)
    : Device(std::make_shared<implementation::DeviceCPU>(info)) {
  auto cpu = static_cast<implementation::DeviceCPU*>(impl().get());
  setDefaultStream(std::make_shared<implementation::StreamCPU>(
      std::make_shared<implementation::ThreadPoolDefault>(*cpu)));
}

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
