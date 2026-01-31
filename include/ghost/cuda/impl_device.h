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

#ifndef GHOST_CUDA_IMPL_DEVICE_H
#define GHOST_CUDA_IMPL_DEVICE_H

#include <ghost/device.h>

#include "cu_ptr.h"

namespace ghost {
namespace implementation {
class DeviceCUDA;

class StreamCUDA : public Stream {
 public:
  cu::ptr<CUstream> queue;

  StreamCUDA(cu::ptr<CUstream> queue_);
  StreamCUDA(CUcontext dev);
};

class BufferCUDA : public Buffer {
 public:
  cu::ptr<CUdeviceptr> mem;

  BufferCUDA(cu::ptr<CUdeviceptr> mem_);
  BufferCUDA(const DeviceCUDA& dev, size_t bytes,
             Access access = Access_ReadWrite);

  virtual void copy(const ghost::Stream& s, const ghost::Buffer& src,
                    size_t bytes) override;
  virtual void copy(const ghost::Stream& s, const void* src,
                    size_t bytes) override;
  virtual void copyTo(const ghost::Stream& s, void* dst,
                      size_t bytes) const override;
};

class MappedBufferCUDA : public BufferCUDA {
 public:
  cu::ptr<void*> ptr;

  MappedBufferCUDA(cu::ptr<CUdeviceptr> mem_);
  MappedBufferCUDA(const DeviceCUDA& dev, size_t bytes,
                   Access access = Access_ReadWrite);

  virtual void* map(const ghost::Stream& s, Access access,
                    bool sync = true) override;
  virtual void unmap(const ghost::Stream& s) override;
};

class ImageCUDA : public Image {
 public:
  cu::ptr<void*> mem;
  ImageDescription descr;

  ImageCUDA(cu::ptr<void*> mem_, const ImageDescription& descr_);
  ImageCUDA(const DeviceCUDA& dev, const ImageDescription& descr_);
  ImageCUDA(const DeviceCUDA& dev, const ImageDescription& descr_,
            BufferCUDA& buffer);
  ImageCUDA(const DeviceCUDA& dev, const ImageDescription& descr_,
            ImageCUDA& image);

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

class DeviceCUDA : public Device {
 public:
  cu::ptr<CUcontext> context;
  cu::ptr<CUstream> queue;

  struct computeCapability {
    int major, minor;
  };

  DeviceCUDA(const SharedContext& share);

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
};
}  // namespace implementation
}  // namespace ghost

#endif