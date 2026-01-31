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

#ifndef GHOST_CPU_IMPL_DEVICE_H
#define GHOST_CPU_IMPL_DEVICE_H

#include <ghost/cpu/impl_function.h>
#include <ghost/device.h>

#ifdef __APPLE_CC__
#include <dispatch/dispatch.h>
#endif

#include <set>

#ifdef __APPLE_CC__
#define GHOST_USE_STD_THREAD 0
#else
#define GHOST_USE_STD_THREAD 1
#endif
#if GHOST_USE_STD_THREAD
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#endif

namespace ghost {
namespace implementation {
class DeviceCPU;

class ThreadWork {
 public:
  FunctionCPU::Type function;
  std::vector<Attribute> args;
  size_t i, count;
  bool quit;
};

class ThreadPool {
 public:
  virtual ~ThreadPool() {}

  virtual void thread(size_t count, FunctionCPU::Type function,
                      const std::vector<Attribute>& args) = 0;

  virtual void sync() {}
};

class ThreadPoolDefault : public ThreadPool {
 public:
  const DeviceCPU& dev;
#if GHOST_USE_STD_THREAD
  std::vector<std::thread> threads;
  std::queue<ThreadWork> work;
  std::mutex mutex;
  std::condition_variable cv;
#elif __APPLE_CC__
  dispatch_queue_t queue;
  dispatch_group_t group;
#endif

  ThreadPoolDefault(const DeviceCPU& dev_);
  ~ThreadPoolDefault();

  virtual void thread(size_t count, FunctionCPU::Type function,
                      const std::vector<Attribute>& args) override;
  virtual void sync() override;

 private:
  void worker();
};

class StreamCPU : public Stream {
 public:
  std::shared_ptr<ThreadPool> pool;

  StreamCPU(std::shared_ptr<ThreadPool> pool_);
  ~StreamCPU();

  virtual void sync() override;
};

class BufferCPU : public Buffer {
 public:
  void* ptr;
  size_t size;

  BufferCPU(const DeviceCPU& dev, size_t bytes);

  virtual void copy(const ghost::Stream& s, const ghost::Buffer& src,
                    size_t bytes) override;
  virtual void copy(const ghost::Stream& s, const void* src,
                    size_t bytes) override;
  virtual void copyTo(const ghost::Stream& s, void* dst,
                      size_t bytes) const override;
};

class ImageCPU : public Image {
 public:
  ImageDescription descr;

  ImageCPU(const DeviceCPU& dev, const ImageDescription& descr);
  ImageCPU(const DeviceCPU& dev, const ImageDescription& descr,
           BufferCPU& buffer);
  ImageCPU(const DeviceCPU& dev, const ImageDescription& descr,
           ImageCPU& image);

  virtual void copy(const ghost::Stream& s, const ghost::Image& src) override;
  virtual void copy(const ghost::Stream& s, const ghost::Buffer& src,
                    const ImageDescription& descr) override;
  virtual void copy(const ghost::Stream& s, const void* src,
                    const ImageDescription& descr) override;
  virtual void copyTo(const ghost::Stream& s, ghost::Buffer& dst,
                      const ImageDescription& descr) const override;
  virtual void copyTo(const ghost::Stream& s, void* dst,
                      const ImageDescription& descr) const override;
};

class DeviceCPU : public Device {
 public:
  size_t cores;

  DeviceCPU(const SharedContext& share);

  virtual ghost::Library loadLibraryFromText(
      const std::string& text, const std::string& options = "") const override;
  virtual ghost::Library loadLibraryFromData(
      const void* data, size_t len,
      const std::string& options = "") const override;
  virtual ghost::Library loadLibraryFromFile(
      const std::string& filename) const override;

  virtual SharedContext shareContext() const override;

  virtual ghost::Stream createStream() const override;

  virtual size_t getMemoryPoolSize() const override;
  virtual void setMemoryPoolSize(size_t bytes) override;
  virtual ghost::Buffer allocateBuffer(
      size_t bytes, Access access = Access_ReadWrite) const override;
  virtual ghost::MappedBuffer allocateMappedBuffer(
      size_t bytes, Access access = Access_ReadWrite) const override;
  virtual ghost::Image allocateImage(
      const ImageDescription& descr) const override;
  virtual ghost::Image sharedImage(const ImageDescription& descr,
                                   ghost::Buffer& buffer) const override;
  virtual ghost::Image sharedImage(const ImageDescription& descr,
                                   ghost::Image& image) const override;

  virtual Attribute getAttribute(DeviceAttributeId what) const override;

  static size_t getNumberOfCores();
};
}  // namespace implementation
}  // namespace ghost

#endif
