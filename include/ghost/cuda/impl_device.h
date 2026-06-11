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

#include <ghost/cuda/cu_ptr.h>
#include <ghost/device.h>
#include <ghost/implementation/recorded_command_buffer.h>

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace ghost {
namespace implementation {
class DeviceCUDA;

class EventCUDA : public Event {
 public:
  cu::ptr<CUevent> event;

  EventCUDA(cu::ptr<CUevent> event_);

  virtual void wait() override;
  virtual bool isComplete() const override;
  virtual double elapsed(const Event& other) const override;
};

class StreamCUDA : public Stream {
 public:
  cu::ptr<CUstream> queue;

  /// @brief Liveness witness recorded by buffers/images in markUsed().
  /// Expires when this StreamCUDA is destroyed, letting deferred release
  /// detect that the recorded @c CUstream handle is no longer valid.
  std::shared_ptr<void> aliveToken = std::make_shared<char>(0);

  StreamCUDA(cu::ptr<CUstream> queue_);
  StreamCUDA(CUcontext dev);
  ~StreamCUDA();

  void sync();
  virtual std::shared_ptr<Event> record() override;
  virtual void waitForEvent(const std::shared_ptr<Event>& e) override;
  virtual void barrier() override;

  /// @brief Capture a reference to @p owner alongside a freshly recorded
  /// event on this stream. The owner is released when the pending-memory
  /// list is reaped (sync, or opportunistic poll at the next enqueue). Used
  /// to keep host memory alive across an async HtoD/DtoH copy when the
  /// caller hands ownership to Ghost via @c HostBytes::adopt.
  void retainHostUntilDone(std::shared_ptr<void> owner);

  /// @brief Drop entries whose events have completed.
  void reapPendingHostMemory();

 protected:
  struct PendingHostMemory {
    cu::ptr<CUevent> event;
    std::shared_ptr<void> owner;
  };

  std::vector<PendingHostMemory> pendingHostMemory;
};

/// @brief Record-and-replay @c CommandBuffer for CUDA, adding native
/// interop via @ref encodeNative.
///
/// CUDA has no native command-buffer concept that maps onto Ghost's
/// recording cb (CUDA graphs are a separate model that most libraries
/// don't use), so the cb still replays its variants directly onto the
/// target stream's @c CUstream at submit time. This subclass exists to
/// add @ref encodeNative on top of the default @ref RecordedCommandBuffer
/// machinery.
class CommandBufferCUDA : public RecordedCommandBuffer {
 public:
  /// @brief Defer a native CUDA encoding step to submit-time replay.
  ///
  /// At replay, @p body is invoked with the target stream's @c CUstream.
  /// Ordering against adjacent Ghost-recorded work is automatic by
  /// CUDA's in-order stream semantics — issue your work onto @p stream
  /// and it slots between the surrounding recorded commands.
  ///
  /// Body contract:
  ///   1. All work must be enqueued on @p stream (or on a stream that
  ///      synchronizes against it before this body returns).
  ///   2. Do not call @c cuStreamSynchronize / @c cuCtxSynchronize.
  ///
  /// Typical use — splicing cuDNN/cuBLAS into a Ghost batch:
  /// @code
  /// cb_cuda->encodeNative([=](CUstream s) {
  ///   cudnnSetStream(handle, s);
  ///   cudnnConvolutionForward(handle, ...);
  /// });
  /// @endcode
  void encodeNative(std::function<void(CUstream stream)> body);

 protected:
  void replayEncodeNative(const EncodeNativeCmd& cmd,
                          const ghost::Stream& stream) override;
};

/// @brief A stream a buffer/image has been used on, with a liveness witness
/// so release paths can tell whether the raw handle is still valid.
struct StreamUse {
  CUstream stream;
  std::weak_ptr<void> alive;
};

/// @brief Owning texture-object handle (see the CUtexObject aliasing note in
/// cu_ptr.h for why the DETAIL parameter is spelled out).
typedef cu::ptr<CUtexObject, cu::detail<cu::GhostCUtexObject>> TexturePtr;

class BufferCUDA : public Buffer {
 public:
  cu::ptr<CUdeviceptr> mem;
  size_t _size;

  BufferCUDA(cu::ptr<CUdeviceptr> mem_, size_t bytes);
  BufferCUDA(const DeviceCUDA& dev, size_t bytes,
             const BufferOptions& opts = {});
  ~BufferCUDA();

  /// @brief Record that this buffer has been used on a stream. The destructor
  /// defers the free until pending work on each recorded stream has
  /// completed, so callers may drop the wrapper immediately after a
  /// fire-and-forget dispatch without synchronizing first. For pool-backed
  /// buffers, a stream's first use also takes a device-side dependency on
  /// the asynchronous allocation's completion.
  virtual void markUsed(const StreamCUDA& s);

#if CUDA_VERSION >= 11020
  /// @brief Mark this buffer as backed by the device's @c CUmemoryPool.
  /// The allocation was stream-ordered on @p allocStream and completes when
  /// @p ready fires. The shared @p pool reference keeps the pool alive until
  /// the destructor has enqueued the stream-ordered free that returns the
  /// memory to it.
  void setPoolBacked(const cu::ptr<CUmemoryPool>& pool, CUstream allocStream,
                     cu::ptr<CUevent>&& ready);
#endif

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

  virtual void copy(const ghost::Encoder& s, HostBytes src, size_t dstOffset,
                    size_t bytes) override;
  virtual void copyTo(const ghost::Encoder& s, HostBytes dst, size_t srcOffset,
                      size_t bytes) const override;

  virtual void fill(const ghost::Encoder& s, size_t offset, size_t size,
                    uint8_t value) override;
  virtual void fill(const ghost::Encoder& s, size_t offset, size_t size,
                    const void* pattern, size_t patternSize) override;

  virtual std::shared_ptr<Buffer> createSubBuffer(
      const std::shared_ptr<Buffer>& self, size_t offset, size_t size) override;

 protected:
  // Streams that have enqueued work referencing this buffer's memory.
  std::vector<StreamUse> _useStreams;
#if CUDA_VERSION >= 11020
  // Pool-backed buffers: shared keepalive for the pool and completion event
  // of the asynchronous allocation (see setPoolBacked).
  cu::ptr<CUmemoryPool> _pool;
  cu::ptr<CUevent> _ready;
#endif
  // Stream a pool-backed allocation was ordered on; null otherwise.
  CUstream _allocStream = nullptr;
};

class SubBufferCUDA : public BufferCUDA {
 public:
  std::shared_ptr<Buffer> _parent;

  SubBufferCUDA(std::shared_ptr<Buffer> parent, cu::ptr<CUdeviceptr> mem_,
                size_t bytes);

  void markUsed(const StreamCUDA& s) override;
};

class MappedBufferCUDA : public BufferCUDA {
 public:
  cu::ptr<void*> ptr;

  MappedBufferCUDA(cu::ptr<void*> mem_);
  MappedBufferCUDA(const DeviceCUDA& dev, size_t bytes,
                   const BufferOptions& opts = {});
  ~MappedBufferCUDA();

  virtual void* map(const ghost::Encoder& s, Access access,
                    bool sync = true) override;
  virtual void unmap(const ghost::Encoder& s) override;
};

class ImageCUDA : public Image {
 public:
  cu::ptr<CUdeviceptr> mem;
  ImageDescription descr;

  ImageCUDA(cu::ptr<CUdeviceptr> mem_, const ImageDescription& descr_,
            const DeviceCUDA* dev = nullptr);
  ImageCUDA(const DeviceCUDA& dev, const ImageDescription& descr_);
  /// @brief Shared image over @p buffer's memory. The image holds a
  /// non-owning view plus a reference that keeps the donor alive; the donor
  /// keeps ownership and frees (with full use-stream ordering — markUsed
  /// propagates to it) once both wrappers are gone.
  ImageCUDA(const DeviceCUDA& dev, const ImageDescription& descr_,
            const std::shared_ptr<Buffer>& buffer);
  /// @brief Shared image over @p image's memory. Same donor semantics as
  /// the buffer overload.
  ImageCUDA(const DeviceCUDA& dev, const ImageDescription& descr_,
            const std::shared_ptr<Image>& image);
  ~ImageCUDA();

  /// @brief Record that this image has been used on a stream. See
  /// @c BufferCUDA::markUsed.
  void markUsed(const StreamCUDA& s);

  /// @brief Return the address of a cached @c CUtexObject for sampling this
  /// image with the given configuration, creating it on first use. The
  /// returned pointer is stable for the texture's lifetime (it points into
  /// the handle's heap node) and is suitable for CUDA kernel parameter
  /// arrays. Cached textures are destroyed when the image is — deferred
  /// behind in-flight work via the device's reap list (see
  /// @c DeviceCUDA::deferTextureRelease).
  CUtexObject* lookupTexture(CUaddress_mode addressMode,
                             CUfilter_mode filterMode, bool normalizedCoords);

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
  virtual void copy(const ghost::Encoder& s, HostBytes src,
                    const BufferLayout& layout) override;
  virtual void copyTo(const ghost::Encoder& s, HostBytes dst,
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

 protected:
  struct CachedTexture {
    CUaddress_mode address;
    CUfilter_mode filter;
    bool normalized;
    TexturePtr tex;
  };

  /// @brief Move the cached texture handles out (for handoff to the
  /// device's deferred-release list), leaving the cache empty.
  std::vector<TexturePtr> takeTextures();

  std::vector<StreamUse> _useStreams;
  std::vector<CachedTexture> _textures;
  // Owning device, used to park cached textures for deferred destruction
  // when the image dies with work in flight. Null for wrapper images
  // created without one; those fall back to a stream drain. Images do not
  // outlive their device (documented contract).
  const DeviceCUDA* _device = nullptr;
  // sharedImage donors: this image's mem is a non-owning view into the
  // donor's allocation; the reference keeps the donor (and so the memory)
  // alive for this image's lifetime.
  std::shared_ptr<Buffer> _donorBuffer;
  std::shared_ptr<Image> _donorImage;
};

// Class to track thread's current context, even if other libraries change it.
// Use one cuCtxPushContext and wait to pop until exiting from the thread (or
// destroying a context).
class CU_CurrentContext {
 protected:
  static thread_local size_t _pushCount;

 public:
  // get context (cuCtxGetCurrent).
  static CUcontext get();
  // set context, pushing on stack if necessary.
  static CUresult set(CUcontext c);
  // cuCtxPushContext (or cuCtxContextCreate) was just called.
  static void pushed();
  static void pop();
};

class DeviceCUDA : public Device {
 public:
  cu::ptr<CUcontext> context;
  cu::ptr<CUstream> queue;
  CUdevice device;
#if CUDA_VERSION >= 11020
  // cu::ptr shares ownership on copy, so pool-backed buffers keep the pool
  // alive across a setMemoryPoolSize() swap while their stream-ordered
  // frees are pending.
  cu::ptr<CUmemoryPool> memPool;
#endif

  struct ComputeCapability {
    int major, minor;
  } computeCapability;

  DeviceCUDA(const SharedContext& share);
  DeviceCUDA(const GpuInfo& info);
  DeviceCUDA(int deviceOrdinal);
  ~DeviceCUDA() override;

  void activate(void** prevOut = nullptr) override;
  void deactivate(void* prev = nullptr) override;

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

  /// @brief Park texture objects for destruction once the work currently
  /// pending on @p streams completes (events are recorded here, on the
  /// calling host thread — never inside a driver callback, which must not
  /// make CUDA API calls). Entries are destroyed by later
  /// @ref reapDeferredTextures passes.
  void deferTextureRelease(std::vector<TexturePtr>&& textures,
                           const std::vector<CUstream>& streams) const;

  /// @brief Destroy parked texture objects whose guarding events have
  /// completed. With @p waitAll, block until every entry is destroyable
  /// (device teardown).
  void reapDeferredTextures(bool waitAll = false) const;

 protected:
  struct PendingTextureRelease {
    std::vector<TexturePtr> textures;
    std::vector<cu::ptr<CUevent>> events;
  };

  mutable std::mutex _texReapMutex;
  mutable std::vector<PendingTextureRelease> _texReap;
};
}  // namespace implementation
}  // namespace ghost

#endif
