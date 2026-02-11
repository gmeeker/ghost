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

#include <set>

#include "ptr.h"

namespace ghost {
namespace implementation {
class DeviceOpenCL;

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
  void addEvent();
  cl_event* event();
};

class BufferOpenCL : public Buffer {
 public:
  opencl::ptr<cl_mem> mem;

  BufferOpenCL(opencl::ptr<cl_mem> mem_);
  BufferOpenCL(const DeviceOpenCL& dev, size_t bytes,
               Access access = Access_ReadWrite);

  virtual void copy(const ghost::Stream& s, const ghost::Buffer& src,
                    size_t bytes) override;
  virtual void copy(const ghost::Stream& s, const void* src,
                    size_t bytes) override;
  virtual void copyTo(const ghost::Stream& s, void* dst,
                      size_t bytes) const override;
};

class MappedBufferOpenCL : public BufferOpenCL {
 public:
  size_t length;
  void* ptr;

  MappedBufferOpenCL(opencl::ptr<cl_mem> mem_, size_t bytes);
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
};

class DeviceOpenCL : public Device {
 private:
  std::string _version;
  std::set<std::string> _extensions;
  bool _fullProfile;

 public:
  opencl::ptr<cl_context> context;
  opencl::ptr<cl_command_queue> queue;

  DeviceOpenCL(const SharedContext& share);

  virtual ghost::Library loadLibraryFromText(
      const std::string& text, const std::string& options = "") const override;
  virtual ghost::Library loadLibraryFromData(
      const void* data, size_t len,
      const std::string& options = "") const override;

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
