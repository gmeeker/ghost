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

#ifndef GHOST_METAL_IMPL_DEVICE_H
#define GHOST_METAL_IMPL_DEVICE_H

#import <Metal/Metal.h>
#include <ghost/device.h>
#include <ghost/objc/ptr.h>

namespace ghost {
namespace implementation {
class DeviceMetal;

class EventMetal : public Event {
 public:
  objc::ptr<id<MTLSharedEvent>> sharedEvent;
  uint64_t targetValue;

  EventMetal(objc::ptr<id<MTLSharedEvent>> event_, uint64_t value);

  virtual void wait() override;
  virtual bool isComplete() const override;
};

class StreamMetal : public Stream {
 public:
  objc::ptr<id<MTLCommandQueue>> queue;
  objc::ptr<id<MTLCommandBuffer>> lastCommandBuffer;
  objc::ptr<id<MTLEvent>> syncEvent;
  uint64_t syncCounter = 0;

  StreamMetal(
      objc::ptr<id<MTLCommandQueue>> queue_ = objc::ptr<id<MTLCommandQueue>>());
  StreamMetal(id<MTLDevice> dev);

  // Commit a command buffer, signal the sync event, and track it for sync().
  void commitAndTrack(id<MTLCommandBuffer> cb);

  // Encode a wait for all previously committed work on a new command buffer.
  // Must be called before encoding any operations that read untracked resources
  // written by prior command buffers.
  void encodeWait(id<MTLCommandBuffer> cb);

  virtual void sync() override;
  virtual std::shared_ptr<Event> record() override;
  virtual void waitForEvent(const std::shared_ptr<Event>& e) override;
};

class BufferMetal : public Buffer {
 public:
  objc::ptr<id<MTLBuffer>> mem;
  size_t _size;

  BufferMetal(objc::ptr<id<MTLBuffer>> mem_, size_t bytes);
  BufferMetal(const DeviceMetal& dev, size_t bytes,
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

class SubBufferMetal : public BufferMetal {
 public:
  std::shared_ptr<Buffer> _parent;
  size_t _offset;

  SubBufferMetal(std::shared_ptr<Buffer> parent, objc::ptr<id<MTLBuffer>> mem_,
                 size_t offset, size_t size);

  virtual size_t baseOffset() const override;

  virtual void copy(const ghost::Stream& s, const ghost::Buffer& src,
                    size_t bytes) override;
  virtual void copy(const ghost::Stream& s, const void* src,
                    size_t bytes) override;
  virtual void copyTo(const ghost::Stream& s, void* dst,
                      size_t bytes) const override;
};

class MappedBufferMetal : public BufferMetal {
 public:
  size_t length;

  MappedBufferMetal(objc::ptr<id<MTLBuffer>> mem_, size_t bytes);
  MappedBufferMetal(const DeviceMetal& dev, size_t bytes,
                    Access access = Access_ReadWrite);

  virtual void* map(const ghost::Stream& s, Access access,
                    bool sync = true) override;
  virtual void unmap(const ghost::Stream& s) override;
};

class ImageMetal : public Image {
 public:
  objc::ptr<id<MTLTexture>> mem;
  ImageDescription descr;

  ImageMetal(objc::ptr<id<MTLTexture>> mem_, const ImageDescription& descr);
  ImageMetal(const DeviceMetal& dev, const ImageDescription& descr);
  ImageMetal(const DeviceMetal& dev, const ImageDescription& descr,
             BufferMetal& buffer);
  ImageMetal(const DeviceMetal& dev, const ImageDescription& descr,
             ImageMetal& image);

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

class DeviceMetal : public Device {
 public:
  objc::ptr<id<MTLDevice>> dev;
  objc::ptr<id<MTLCommandQueue>> queue;
  objc::ptr<id<MTLHeap>> heap;

  DeviceMetal(const SharedContext& share);
  DeviceMetal(const GpuInfo& info);
  DeviceMetal(id<MTLDevice> device);

  virtual ghost::Library loadLibraryFromText(
      const std::string& text,
      const CompilerOptions& options = CompilerOptions(),
      bool retainBinary = false) const override;
  virtual ghost::Library loadLibraryFromData(
      const void* data, size_t len,
      const CompilerOptions& options = CompilerOptions(),
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
};
}  // namespace implementation
}  // namespace ghost

#endif

// vim: ts=2:sw=2:et:ft=mm
// -*- mode: objective-c++; indent-tabs-mode: nil; tab-width: 2 -*-
// code: language=objective-c++ insertSpaces=true tabSize=2
