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
#include <ghost/implementation/recorded_command_buffer.h>
#include <ghost/objc/ptr.h>

#include <functional>
#include <vector>

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

/// @brief State and lifecycle shared by every Metal encoder (Stream and
/// CommandBuffer).
///
/// BufferMetal / ImageMetal / FunctionMetal cross-cast a @c ghost::Encoder
/// to this type to find the @c MTLCommandBuffer to record into. Both
/// @c StreamMetal and @c CommandBufferMetal inherit from this mixin in
/// addition to their @c implementation::Encoder-derived base. Use
/// @c metalEncoder(s) to perform the cross-cast.
///
/// Metal allows only one of @c MTLBlitCommandEncoder /
/// @c MTLComputeCommandEncoder to be active on a command buffer at a time,
/// so the mixin owns the swap bookkeeping: @c getBlitEncoder /
/// @c getComputeEncoder will @c endEncoding the other if it's active.
class MetalEncoder {
 public:
  // commandBuffer is the current target for encoding. For StreamMetal it
  // is a transient cb held between sync points; for CommandBufferMetal it
  // is allocated in submit() bound to the target stream's queue. Public so
  // BufferMetal / ImageMetal / FunctionMetal can record directly.
  objc::ptr<id<MTLCommandBuffer>> commandBuffer;
  // At most one of these is non-nil at a time.
  objc::ptr<id<MTLBlitCommandEncoder>> blitEncoder;
  objc::ptr<id<MTLComputeCommandEncoder>> computeEncoder;
  // Per-cb fence used to chain encoder boundaries. Ghost allocates
  // resources with @c MTLHazardTrackingModeUntracked, so inter-encoder
  // memory ordering within a cb is not automatic; we update the fence at
  // the end of each encoder and wait on it at the start of the next.
  // Recreated lazily after each commit (a fence is bound to a cb).
  objc::ptr<id<MTLFence>> fence;
  // Set when an encoder is explicitly ended via @c endEncoding() (i.e. by
  // a user-issued @c cb.barrier() or by submit's final flush) AND the
  // closed encoder had updated the fence. The next encoder created in
  // this cb must @c waitForFence to honor the barrier. Without this,
  // consecutive same-type encoders separated by a barrier race because
  // resources are hazard-untracked.
  bool pendingFenceWait = false;
  // When true, @c getComputeEncoder uses @c MTLDispatchTypeConcurrent so
  // consecutive dispatches in the same encoder may run concurrently —
  // caller takes responsibility for inter-dispatch hazards. When false
  // (default for streams), uses serial dispatch so consecutive
  // dispatches are ordered without callers inserting explicit barriers.
  // Set once at construction by StreamMetal / CommandBufferMetal.
  bool concurrent = false;

  MetalEncoder() = default;
  virtual ~MetalEncoder() = default;

  /// @brief Ensure @c commandBuffer is allocated. Idempotent.
  virtual void begin() = 0;

  /// @brief Get or create a blit encoder. Ends any active compute encoder.
  id<MTLBlitCommandEncoder> getBlitEncoder();

  /// @brief Get or create a compute encoder. Ends any active blit encoder.
  id<MTLComputeCommandEncoder> getComputeEncoder();

  /// @brief End any active encoder.
  void endEncoding();
};

class StreamMetal : public Stream, public MetalEncoder {
 public:
  objc::ptr<id<MTLCommandQueue>> queue;
  // Ghost allocates Metal resources with hazard tracking disabled, so inter-cb
  // ordering is not automatic. @c syncEvent is a per-stream monotonic event
  // bumped on each cb commit and waited on by each fresh transient cb (and
  // by CommandBufferMetal::submit). Establishes happens-before across cbs.
  objc::ptr<id<MTLEvent>> syncEvent;
  uint64_t syncCounter = 0;

  StreamMetal(
      objc::ptr<id<MTLCommandQueue>> queue_ = objc::ptr<id<MTLCommandQueue>>(),
      const StreamOptions& options = {});
  StreamMetal(id<MTLDevice> dev, const StreamOptions& options = {});

  void begin() override;

  /// @brief Commit the current transient cb. After this @c commandBuffer is
  /// nil; the next op will call @c begin() to allocate a fresh one. Tracks
  /// the committed cb so @c sync() can wait on it.
  void commit();

  /// @brief Register an externally-committed cb (from CommandBufferMetal)
  /// as the stream's new tail so @c sync() waits for it. The cb must have
  /// been committed on this stream's queue.
  void attachCommitted(objc::ptr<id<MTLCommandBuffer>> cb);

  virtual void sync() override;
  virtual std::shared_ptr<Event> record() override;
  virtual void waitForEvent(const std::shared_ptr<Event>& e) override;

 private:
  // The most recently committed cb on this stream's queue. Metal queues
  // are FIFO so waiting on the last committed cb implicitly waits for all
  // prior ones.
  objc::ptr<id<MTLCommandBuffer>> _lastCommitted;
};

/// @brief Cross-cast a @c ghost::Encoder to its underlying @c MetalEncoder.
///
/// Throws @c ghost::unsupported_error if the encoder's impl is not a
/// Metal encoder.
MetalEncoder& metalEncoder(const ghost::Encoder& s);

/// @brief Native Metal @c CommandBuffer wrapping its own @c MTLCommandBuffer.
///
/// Inherits the variant-recording machinery from @ref RecordedCommandBuffer
/// (dispatch / copy / fill / barrier records accumulate into @c commands)
/// and the encoder interface from @ref MetalEncoder. On @c submit() a fresh
/// @c MTLCommandBuffer is allocated from the target stream's queue, the
/// recorded variants are replayed directly into it, then committed. An
/// @c addCompletedHandler block drops the retained resources captured by the
/// variants at GPU-safe time. @c reset() waits for completion before
/// clearing.
class CommandBufferMetal : public RecordedCommandBuffer, public MetalEncoder {
 public:
  CommandBufferMetal(const DeviceMetal& dev_,
                     const CommandBufferOptions& options = {});
  ~CommandBufferMetal();

  void begin() override;
  void submit(const ghost::Stream& stream) override;
  void reset() override;
  void onCompletion(std::function<void()> handler) override;

 private:
  const DeviceMetal& _dev;
  // The committed cb from the most recent submit(). Held until reset() or
  // destruction so we can waitUntilCompleted before clearing variants.
  objc::ptr<id<MTLCommandBuffer>> _submittedCB;

  /// @brief Wait on @c _submittedCB if a submission is in flight. Idempotent.
  void waitForCompletion();
};

class BufferMetal : public Buffer {
 public:
  objc::ptr<id<MTLBuffer>> mem;
  size_t _size;

  BufferMetal(objc::ptr<id<MTLBuffer>> mem_, size_t bytes);
  BufferMetal(const DeviceMetal& dev, size_t bytes,
              const BufferOptions& opts = {});
  ~BufferMetal();

  virtual size_t size() const override;

  virtual void copy(const ghost::Encoder& s, const ghost::Buffer& src,
                    size_t bytes) override;
  virtual void copy(const ghost::Encoder& s, const void* src,
                    size_t bytes) override;
  virtual void copyTo(const ghost::Encoder& s, void* dst,
                      size_t bytes) const override;

  virtual void copy(const ghost::Encoder& s, const ghost::Buffer& src,
                    size_t srcOffset, size_t dstOffset, size_t bytes) override;
  virtual void copy(const ghost::Encoder& s, const void* src, size_t dstOffset,
                    size_t bytes) override;
  virtual void copyTo(const ghost::Encoder& s, void* dst, size_t srcOffset,
                      size_t bytes) const override;

  virtual void fill(const ghost::Encoder& s, size_t offset, size_t size,
                    uint8_t value) override;
  virtual void fill(const ghost::Encoder& s, size_t offset, size_t size,
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

  virtual void copy(const ghost::Encoder& s, const ghost::Buffer& src,
                    size_t bytes) override;
  virtual void copy(const ghost::Encoder& s, const void* src,
                    size_t bytes) override;
  virtual void copyTo(const ghost::Encoder& s, void* dst,
                      size_t bytes) const override;
};

class MappedBufferMetal : public BufferMetal {
 public:
  size_t length;

  MappedBufferMetal(objc::ptr<id<MTLBuffer>> mem_, size_t bytes);
  MappedBufferMetal(const DeviceMetal& dev, size_t bytes,
                    const BufferOptions& opts = {});
  ~MappedBufferMetal();

  virtual void* map(const ghost::Encoder& s, Access access,
                    bool sync = true) override;
  virtual void unmap(const ghost::Encoder& s) override;
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
  ~ImageMetal();

  virtual const ImageDescription& description() const override { return descr; }

  virtual void copy(const ghost::Encoder& s, const ghost::Image& src) override;
  virtual void copy(const ghost::Encoder& s, const ghost::Buffer& src,
                    const BufferLayout& layout) override;
  virtual void copy(const ghost::Encoder& s, const void* src,
                    const BufferLayout& layout) override;
  virtual void copyTo(const ghost::Encoder& s, ghost::Buffer& dst,
                      const BufferLayout& layout) const override;
  virtual void copyTo(const ghost::Encoder& s, void* dst,
                      const BufferLayout& layout) const override;
  virtual void copy(const ghost::Encoder& s, const ghost::Buffer& src,
                    const BufferLayout& layout,
                    const Origin3& imageOrigin) override;
  virtual void copyTo(const ghost::Encoder& s, ghost::Buffer& dst,
                      const BufferLayout& layout,
                      const Origin3& imageOrigin) const override;
  virtual void copy(const ghost::Encoder& s, const ghost::Image& src,
                    const Size3& region, const Origin3& srcOrigin,
                    const Origin3& dstOrigin) override;
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

  virtual ghost::Stream createStream(
      const StreamOptions& options = {}) const override;

  virtual std::shared_ptr<CommandBuffer> createCommandBuffer(
      const CommandBufferOptions& options = {}) const override;

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

  virtual ghost::Buffer wrapBuffer(const SharedBuffer& shared) const override;
  virtual ghost::Image wrapImage(const SharedImage& shared) const override;

  virtual Attribute getAttribute(DeviceAttributeId what) const override;
  virtual size_t imageAlignment(const ImageDescription& descr) const override;
};
}  // namespace implementation
}  // namespace ghost

#endif

// vim: ts=2:sw=2:et:ft=mm
// -*- mode: objective-c++; indent-tabs-mode: nil; tab-width: 2 -*-
// code: language=objective-c++ insertSpaces=true tabSize=2
