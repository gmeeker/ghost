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
#include <ghost/thread_pool.h>

#include <set>

namespace ghost {
namespace implementation {
class DeviceCPU;

class EventCPU : public Event {
 public:
  double _timestamp;

  EventCPU();

  virtual void wait() override;
  virtual bool isComplete() const override;
  virtual double timestamp() const override;
};

class StreamCPU : public Stream {
 public:
  std::shared_ptr<ghost::ThreadPool> pool;

  StreamCPU(std::shared_ptr<ghost::ThreadPool> pool_);
  ~StreamCPU();

  virtual void sync() override;
  virtual std::shared_ptr<Event> record() override;
  virtual void waitForEvent(const std::shared_ptr<Event>& e) override;
};

class BufferCPU : public Buffer {
 protected:
  BufferCPU(void* ptr_, size_t bytes);

 public:
  void* ptr;
  size_t _size;

  BufferCPU(const DeviceCPU& dev, size_t bytes);

  virtual size_t size() const override;

  virtual void copy(const ghost::Stream& s, const ghost::Buffer& src,
                    size_t bytes) override;
  virtual void copy(const ghost::Stream& s, const void* src,
                    size_t bytes) override;
  virtual void copyTo(const ghost::Stream& s, void* dst,
                      size_t bytes) const override;

  virtual void copy(const ghost::Stream& s, const ghost::Buffer& src,
                    size_t srcOffset, size_t dstOffset, size_t bytes) override;
  virtual void copy(const ghost::Stream& s, const void* src, size_t dstOffset,
                    size_t bytes) override;
  virtual void copyTo(const ghost::Stream& s, void* dst, size_t srcOffset,
                      size_t bytes) const override;

  virtual void fill(const ghost::Stream& s, size_t offset, size_t size,
                    uint8_t value) override;
  virtual void fill(const ghost::Stream& s, size_t offset, size_t size,
                    const void* pattern, size_t patternSize) override;

  virtual std::shared_ptr<Buffer> createSubBuffer(
      const std::shared_ptr<Buffer>& self, size_t offset, size_t size) override;
};

class MappedBufferCPU : public BufferCPU {
 public:
  MappedBufferCPU(const DeviceCPU& dev, size_t bytes);

  virtual void* map(const ghost::Stream& s, Access access,
                    bool sync = true) override;
  virtual void unmap(const ghost::Stream& s) override;
};

class SubBufferCPU : public BufferCPU {
 public:
  std::shared_ptr<Buffer> _parent;

  SubBufferCPU(std::shared_ptr<Buffer> parent, void* ptr_, size_t bytes);
};

class ImageCPU : public Image {
 public:
  ImageDescription descr;
  void* data;
  size_t rowBytes;
  size_t depthBytes;

  ImageCPU(const DeviceCPU& dev, const ImageDescription& descr);
  ImageCPU(const DeviceCPU& dev, const ImageDescription& descr,
           BufferCPU& buffer);
  ImageCPU(const DeviceCPU& dev, const ImageDescription& descr,
           ImageCPU& image);

  virtual const ImageDescription& description() const override { return descr; }

  virtual void copy(const ghost::Stream& s, const ghost::Image& src) override;
  virtual void copy(const ghost::Stream& s, const ghost::Buffer& src,
                    const BufferLayout& layout) override;
  virtual void copy(const ghost::Stream& s, const void* src,
                    const BufferLayout& layout) override;
  virtual void copyTo(const ghost::Stream& s, ghost::Buffer& dst,
                      const BufferLayout& layout) const override;
  virtual void copyTo(const ghost::Stream& s, void* dst,
                      const BufferLayout& layout) const override;
  virtual void copy(const ghost::Stream& s, const ghost::Buffer& src,
                    const BufferLayout& layout,
                    const Origin3& imageOrigin) override;
  virtual void copyTo(const ghost::Stream& s, ghost::Buffer& dst,
                      const BufferLayout& layout,
                      const Origin3& imageOrigin) const override;
  virtual void copy(const ghost::Stream& s, const ghost::Image& src,
                    const Size3& region, const Origin3& srcOrigin,
                    const Origin3& dstOrigin) override;
};

class DeviceCPU : public Device {
 public:
  size_t cores;
  std::shared_ptr<ghost::ThreadPool> pool;

  DeviceCPU(const SharedContext& share);
  DeviceCPU(const GpuInfo& info);
  DeviceCPU(std::shared_ptr<ghost::ThreadPool> pool);

  std::shared_ptr<ghost::ThreadPool> threadPool() const override {
    return pool;
  }

  void setThreadPool(std::shared_ptr<ghost::ThreadPool> p) { pool = p; }

  virtual ghost::Library loadLibraryFromText(
      const std::string& text,
      const CompilerOptions& options = CompilerOptions(),
      bool retainBinary = false) const override;
  virtual ghost::Library loadLibraryFromData(
      const void* data, size_t len,
      const CompilerOptions& options = CompilerOptions(),
      bool retainBinary = false) const override;
  virtual ghost::Library loadLibraryFromFile(
      const std::string& filename) const override;

  virtual SharedContext shareContext() const override;

  virtual ghost::Stream createStream() const override;

  virtual size_t getMemoryPoolSize() const override;
  virtual void setMemoryPoolSize(size_t bytes) override;
  virtual ghost::Buffer allocateBuffer(
      size_t bytes, const BufferOptions& opts = {}) const override;
  virtual ghost::MappedBuffer allocateMappedBuffer(
      size_t bytes, const BufferOptions& opts = {}) const override;
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
