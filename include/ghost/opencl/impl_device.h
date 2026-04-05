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

#ifndef GHOST_OPENCL_IMPL_DEVICE_H
#define GHOST_OPENCL_IMPL_DEVICE_H

#include <ghost/device.h>

#include <list>
#include <set>

#include "ptr.h"

namespace ghost {
namespace implementation {
class DeviceOpenCL;

class EventOpenCL : public Event {
 public:
  opencl::ptr<cl_event> event;

  EventOpenCL(opencl::ptr<cl_event> event_);

  virtual void wait() override;
  virtual bool isComplete() const override;
  virtual double timestamp() const override;
  virtual double elapsed(const Event& other) const override;
};

class StreamOpenCL : public Stream {
 protected:
  opencl::ptr<cl_event> lastEvent;

 public:
  opencl::ptr<cl_command_queue> queue;
  opencl::array<cl_event> events;
  bool outOfOrder;

  StreamOpenCL(opencl::ptr<cl_command_queue> queue_);
  StreamOpenCL(const DeviceOpenCL& dev);

  virtual void sync() override;
  virtual std::shared_ptr<Event> record() override;
  virtual void waitForEvent(const std::shared_ptr<Event>& e) override;
  void addEvent();
  cl_event* event();
};

class BufferOpenCL : public Buffer {
 public:
  opencl::ptr<cl_mem> mem;
  size_t _size;

  BufferOpenCL(opencl::ptr<cl_mem> mem_, size_t bytes);
  BufferOpenCL(const DeviceOpenCL& dev, size_t bytes,
               Access access = Access_ReadWrite);

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

class SubBufferOpenCL : public BufferOpenCL {
 public:
  std::shared_ptr<Buffer> _parent;

  SubBufferOpenCL(std::shared_ptr<Buffer> parent, opencl::ptr<cl_mem> mem_,
                  size_t bytes);
};

class MappedBufferOpenCL : public BufferOpenCL {
 public:
  size_t length;
  void* ptr;

  MappedBufferOpenCL(opencl::ptr<cl_mem> mem_, size_t bytes, size_t allocSize);
  MappedBufferOpenCL(const DeviceOpenCL& dev, size_t bytes,
                     Access access = Access_ReadWrite);

  virtual void* map(const ghost::Stream& s, Access access,
                    bool sync = true) override;
  virtual void unmap(const ghost::Stream& s) override;
};

class ImageOpenCL : public Image {
 public:
  opencl::ptr<cl_mem> mem;
  ImageDescription descr;

  ImageOpenCL(opencl::ptr<cl_mem> mem_, const ImageDescription& descr);
  ImageOpenCL(const DeviceOpenCL& dev, const ImageDescription& descr);
  ImageOpenCL(const DeviceOpenCL& dev, const ImageDescription& descr,
              BufferOpenCL& buffer);
  ImageOpenCL(const DeviceOpenCL& dev, const ImageDescription& descr,
              ImageOpenCL& image);

  virtual void copy(const ghost::Stream& s, const ghost::Image& src) override;
  virtual void copy(const ghost::Stream& s, const ghost::Buffer& src,
                    const ImageDescription& descr) override;
  virtual void copy(const ghost::Stream& s, const void* src,
                    const ImageDescription& descr) override;
  virtual void copyTo(const ghost::Stream& s, ghost::Buffer& dst,
                      const ImageDescription& descr) const override;
  virtual void copyTo(const ghost::Stream& s, void* dst,
                      const ImageDescription& descr) const override;
  virtual void copy(const ghost::Stream& s, const ghost::Buffer& src,
                    const ImageDescription& descr,
                    const Size3& imageOrigin) override;
  virtual void copyTo(const ghost::Stream& s, ghost::Buffer& dst,
                      const ImageDescription& descr,
                      const Size3& imageOrigin) const override;
};

class BufferPool {
 public:
  struct BufferEntry {
    opencl::ptr<cl_mem> mem;
    size_t bytes;
  };

  struct ImageEntry {
    opencl::ptr<cl_mem> mem;
    ImageDescription descr;
    size_t bytes;
  };

  ~BufferPool();

  size_t getLimit() const;
  void setLimit(size_t limit);

  opencl::ptr<cl_mem> lookupBuffer(size_t bytes);
  opencl::ptr<cl_mem> lookupImage(const ImageDescription& descr);

  void reserve(size_t bytes);
  void recycleBuffer(opencl::ptr<cl_mem> mem, size_t bytes);
  void recycleImage(opencl::ptr<cl_mem> mem, const ImageDescription& descr,
                    size_t bytes);
  void clear();

 private:
  void purge(size_t needed = 0);
  static bool imageMatch(const ImageDescription& a, const ImageDescription& b);

  std::list<BufferEntry> _buffers;
  std::list<ImageEntry> _images;
  size_t _current = 0;
  size_t _limit = 0;
};

class PooledBufferOpenCL : public BufferOpenCL {
 public:
  std::shared_ptr<BufferPool> pool;

  PooledBufferOpenCL(opencl::ptr<cl_mem> mem_, size_t bytes,
                     std::shared_ptr<BufferPool> pool_);
  ~PooledBufferOpenCL();
};

class PooledImageOpenCL : public ImageOpenCL {
 public:
  std::shared_ptr<BufferPool> pool;
  size_t imageBytes;

  PooledImageOpenCL(opencl::ptr<cl_mem> mem_, const ImageDescription& descr,
                    size_t bytes, std::shared_ptr<BufferPool> pool_);
  ~PooledImageOpenCL();
};

class DeviceOpenCL : public Device {
 private:
  std::string _version;
  std::set<std::string> _extensions;
  bool _fullProfile;
  mutable std::shared_ptr<BufferPool> _pool;

  void setVersion();

 public:
  opencl::ptr<cl_context> context;
  opencl::ptr<cl_command_queue> queue;

  DeviceOpenCL(const SharedContext& share);
  DeviceOpenCL(const GpuInfo& info);
  DeviceOpenCL(cl_platform_id platform, cl_device_id device);

  virtual ghost::Library loadLibraryFromText(
      const std::string& text, const std::string& options = "",
      bool retainBinary = false) const override;
  virtual ghost::Library loadLibraryFromData(
      const void* data, size_t len, const std::string& options = "",
      bool retainBinary = false) const override;

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

  bool checkVersion(const std::string& version) const;
  bool checkExtension(const std::string& extension) const;

  std::vector<cl_device_id> getDevices() const;
  cl_platform_id getPlatform() const;
  cl_ulong getInt(cl_device_info param_name) const;
  std::string getString(cl_device_info param_name) const;
  std::string getPlatformString(cl_platform_info param_name) const;
};
}  // namespace implementation
}  // namespace ghost

#endif
