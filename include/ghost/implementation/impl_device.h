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

#ifndef GHOST_IMPL_DEVICE_H
#define GHOST_IMPL_DEVICE_H

#include <ghost/attribute.h>
#include <ghost/binary_cache.h>
#include <ghost/gpu_info.h>
#include <ghost/image.h>
#include <ghost/implementation/impl_function.h>
#include <ghost/thread_pool.h>
#include <stdlib.h>

#include <cstdint>
#include <memory>
#include <string>

namespace ghost {

/// @brief Identifiers for queryable device attributes.
///
/// Pass these to Device::getAttribute() or
/// implementation::Device::getAttribute() to retrieve device properties.
enum DeviceAttributeId {
  /// @brief Backend name (e.g., "Metal", "OpenCL", "CUDA").
  kDeviceImplementation,
  /// @brief Device name string.
  kDeviceName,
  /// @brief Device vendor string.
  kDeviceVendor,
  /// @brief Driver or runtime version string.
  kDeviceDriverVersion,
  /// @brief Device family identifier (backend-specific).
  kDeviceFamily,
  /// @brief Number of devices available on this backend.
  kDeviceCount,
  /// @brief Number of compute processors / streaming multiprocessors.
  kDeviceProcessorCount,
  /// @brief Whether the device shares memory with the host (bool).
  kDeviceUnifiedMemory,
  /// @brief Total device global memory in bytes (uint64).
  kDeviceMemory,
  /// @brief Maximum local (shared/threadgroup) memory per work-group in bytes.
  kDeviceLocalMemory,
  /// @brief Maximum threads per work-group.
  kDeviceMaxThreads,
  /// @brief Maximum work-group size per dimension (3-element int array).
  kDeviceMaxWorkSize,
  /// @brief Maximum registers per thread (backend-specific).
  kDeviceMaxRegisters,
  /// @brief Maximum 1D image width.
  kDeviceMaxImageSize1,
  /// @brief Maximum 2D image dimensions (2-element int array).
  kDeviceMaxImageSize2,
  /// @brief Maximum 3D image dimensions (3-element int array).
  kDeviceMaxImageSize3,
  /// @brief Required image row alignment in bytes.
  kDeviceImageAlignment,
  /// @brief Whether integer image filtering is supported (bool).
  kDeviceSupportsImageIntegerFiltering,
  /// @brief Whether float image filtering is supported (bool).
  kDeviceSupportsImageFloatFiltering,
  /// @brief Whether mapped (pinned) buffers are supported (bool).
  kDeviceSupportsMappedBuffer,
  /// @brief Whether program constants / function specialization is supported
  /// (bool).
  kDeviceSupportsProgramConstants,
  /// @brief Whether subgroup (SIMD/warp) operations are supported (bool).
  kDeviceSupportsSubgroup,
  /// @brief Whether subgroup shuffle operations are supported (bool).
  kDeviceSupportsSubgroupShuffle,
  /// @brief Subgroup (SIMD/warp) width in threads.
  kDeviceSubgroupWidth,
  /// @brief Number of compute units / streaming multiprocessors.
  kDeviceMaxComputeUnits,
  /// @brief Minimum allocation alignment in bytes.
  kDeviceMemoryAlignment,
  /// @brief Buffer offset alignment for sub-buffers in bytes.
  kDeviceBufferAlignment,
  /// @brief Maximum single buffer allocation size in bytes.
  kDeviceMaxBufferSize,
  /// @brief Maximum constant buffer / push constant size in bytes.
  kDeviceMaxConstantBufferSize,
  /// @brief Nanoseconds per timestamp tick (float).
  kDeviceTimestampPeriod,
  /// @brief Whether profiling timers are supported (bool).
  kDeviceSupportsProfilingTimer,
  /// @brief Whether hardware cooperative/SIMD-group matrix operations are
  /// supported (bool).
  ///
  /// True when the device can execute cooperative matrix instructions:
  /// - Metal: Apple GPU family 7+ (A14/M1 and later, simdgroup matrix ops)
  /// - CUDA: compute capability >= 7.0 (Volta+, WMMA)
  /// - Vulkan: VK_KHR_cooperative_matrix extension present
  /// - OpenCL/DirectX/CPU: false
  kDeviceSupportsCooperativeMatrix,
};

/// @brief Opaque container for backend-specific context handles, used to share
/// a device context.
///
/// Returned by Device::shareContext() and passed to backend constructors to
/// create a second Device that shares the same GPU context and command queue.
class SharedContext {
 public:
  /// @brief Backend context handle (e.g., cl_context, CUcontext).
  void* context;
  /// @brief Backend queue handle (e.g., cl_command_queue, MTLCommandQueue).
  void* queue;
  /// @brief Backend device handle (e.g., cl_device_id, CUdevice,
  /// id<MTLDevice>).
  void* device;
  /// @brief Backend platform handle (e.g., cl_platform_id), or @c nullptr.
  void* platform;

  /// @brief Construct a SharedContext with optional handles.
  SharedContext(void* context_ = nullptr, void* queue_ = nullptr,
                void* device_ = nullptr, void* platform_ = nullptr)
      : context(context_),
        queue(queue_),
        device(device_),
        platform(platform_) {}
};

class Event;
class Function;
class Library;
class Encoder;
class Stream;
class Buffer;

/// @brief Options for stream creation.
struct StreamOptions {
  bool profiling = false;
  bool forceEventChain = false;
};
class MappedBuffer;
class Image;

namespace implementation {

/// @brief Abstract backend interface for a GPU synchronization event.
///
/// Backend implementations derive from this class to provide event
/// synchronization and timing. Not copyable.
class Event {
 protected:
  Event() {}

  Event(const Event& rhs) = delete;

  virtual ~Event() {}

  Event& operator=(const Event& rhs) = delete;

 public:
  /// @brief Block the calling CPU thread until this event has completed.
  virtual void wait() = 0;

  /// @brief Query whether this event has completed without blocking.
  virtual bool isComplete() const = 0;

  /// @brief Get the absolute timestamp of this event in seconds.
  ///
  /// The default implementation returns 0. Backends with profiling timers
  /// override this method.
  /// @return Timestamp in seconds, or 0 if not supported.
  virtual double timestamp() const;

  /// @brief Measure elapsed time in seconds between this event and another.
  ///
  /// The default implementation returns 0. Backends with profiling timers
  /// override this method.
  /// @param other The later event.
  /// @return Elapsed time in seconds, or 0 if not supported.
  virtual double elapsed(const Event& other) const;
};

/// @brief Abstract backend interface for an operation encoder.
///
/// Base class for Stream and CommandBuffer implementations.
/// Buffer, Image, and Function impls receive this type and downcast
/// to the backend-specific subclass.
class CommandBuffer;

class Encoder {
 protected:
  Encoder() {}

  Encoder(const Encoder& rhs) = delete;

  virtual ~Encoder() {}

  Encoder& operator=(const Encoder& rhs) = delete;

 public:
  /// @brief If this encoder is a CommandBuffer, return it; otherwise nullptr.
  ///
  /// Used by Buffer/Image/Function to dispatch between immediate execution
  /// (Stream) and deferred recording (CommandBuffer).
  virtual CommandBuffer* asCommandBuffer();
};

/// @brief Abstract backend interface for a GPU command stream.
///
/// Backend implementations derive from this class to provide stream
/// synchronization. Not copyable.
class Stream : public Encoder {
 protected:
  Stream() {}

  Stream(const Stream& rhs) = delete;

  Stream& operator=(const Stream& rhs) = delete;

 public:
  virtual void sync() = 0;

  /// @brief Record an event at the current point in the stream.
  ///
  /// The default implementation throws ghost::unsupported_error.
  /// @return A shared pointer to the recorded event.
  virtual std::shared_ptr<Event> record();

  /// @brief Enqueue a GPU-side wait for an event (cross-stream sync).
  ///
  /// The default implementation throws ghost::unsupported_error.
  /// @param e The event to wait for.
  virtual void waitForEvent(const std::shared_ptr<Event>& e);
};

/// @brief Abstract backend interface for a GPU memory buffer.
///
/// Backend implementations derive from this class to provide buffer
/// copy and optional mapping operations. Not copyable.
class Buffer {
 protected:
  Buffer() {}

  Buffer(const Buffer& rhs) = delete;

  virtual ~Buffer() {}

  Buffer& operator=(const Buffer& rhs) = delete;

 public:
  virtual size_t size() const = 0;

  /// @brief Get the base offset for sub-buffers.
  ///
  /// Regular buffers return 0. Sub-buffers return their offset into the parent
  /// buffer. Used by backends (e.g., Metal) where the underlying handle is
  /// shared with the parent.
  virtual size_t baseOffset() const;

  /// @brief Create a sub-buffer view into this buffer.
  ///
  /// The default implementation throws ghost::unsupported_error.
  /// @param self Shared pointer to this buffer (to keep parent alive).
  /// @param offset Byte offset into this buffer.
  /// @param size Size of the sub-buffer in bytes.
  /// @return Shared pointer to the new sub-buffer.
  virtual std::shared_ptr<Buffer> createSubBuffer(
      const std::shared_ptr<Buffer>& self, size_t offset, size_t size);

  virtual void copy(const ghost::Encoder& s, const ghost::Buffer& src,
                    size_t bytes) = 0;
  virtual void copy(const ghost::Encoder& s, const void* src, size_t bytes) = 0;
  virtual void copyTo(const ghost::Encoder& s, void* dst,
                      size_t bytes) const = 0;

  virtual void copy(const ghost::Encoder& s, const ghost::Buffer& src,
                    size_t srcOffset, size_t dstOffset, size_t bytes) = 0;
  virtual void copy(const ghost::Encoder& s, const void* src, size_t dstOffset,
                    size_t bytes) = 0;
  virtual void copyTo(const ghost::Encoder& s, void* dst, size_t srcOffset,
                      size_t bytes) const = 0;

  virtual void fill(const ghost::Encoder& s, size_t offset, size_t size,
                    uint8_t value) = 0;
  virtual void fill(const ghost::Encoder& s, size_t offset, size_t size,
                    const void* pattern, size_t patternSize) = 0;

  /// @brief Map the buffer into host address space.
  ///
  /// The default implementation throws ghost::unsupported_error. Backends
  /// that support mapped buffers override this method.
  /// @throws ghost::unsupported_error if not supported by the backend.
  virtual void* map(const ghost::Encoder& s, Access access, bool sync = true);

  /// @brief Unmap a previously mapped buffer.
  ///
  /// The default implementation throws ghost::unsupported_error.
  /// @throws ghost::unsupported_error if not supported by the backend.
  virtual void unmap(const ghost::Encoder& s);
};

/// @brief Abstract backend interface for a GPU image (texture).
///
/// Backend implementations derive from this class to provide image
/// copy operations. Not copyable.
class Image {
 protected:
  Image() {}

  Image(const Image& rhs) = delete;

  virtual ~Image() {}

  Image& operator=(const Image& rhs) = delete;

 public:
  /// @brief Get the image description this image was allocated with.
  virtual const ImageDescription& description() const = 0;

  virtual void copy(const ghost::Encoder& s, const ghost::Image& src) = 0;
  virtual void copy(const ghost::Encoder& s, const ghost::Buffer& src,
                    const BufferLayout& layout) = 0;
  virtual void copy(const ghost::Encoder& s, const void* src,
                    const BufferLayout& layout) = 0;
  virtual void copyTo(const ghost::Encoder& s, ghost::Buffer& dst,
                      const BufferLayout& layout) const = 0;
  virtual void copyTo(const ghost::Encoder& s, void* dst,
                      const BufferLayout& layout) const = 0;

  virtual void copy(const ghost::Encoder& s, const ghost::Buffer& src,
                    const BufferLayout& layout, const Origin3& imageOrigin);
  virtual void copyTo(const ghost::Encoder& s, ghost::Buffer& dst,
                      const BufferLayout& layout,
                      const Origin3& imageOrigin) const;

  virtual void copy(const ghost::Encoder& s, const ghost::Image& src,
                    const Size3& region, const Origin3& srcOrigin,
                    const Origin3& dstOrigin);
};

/// @brief Abstract backend interface for a GPU device.
///
/// Backend implementations derive from this class to provide library loading,
/// resource allocation, context sharing, and attribute queries. Not copyable.
class Device {
 protected:
  Device() : _poolSize(0) {}

  Device(const Device& rhs) = delete;

  virtual ~Device() {}

  Device& operator=(const Device& rhs) = delete;

 public:
  static BinaryCache& binaryCache();
  virtual ghost::Library loadLibraryFromText(
      const std::string& text,
      const CompilerOptions& options = CompilerOptions(),
      bool retainBinary = false) const = 0;
  virtual ghost::Library loadLibraryFromData(
      const void* data, size_t len,
      const CompilerOptions& options = CompilerOptions(),
      bool retainBinary = false) const = 0;

  /// @brief Load a GPU program from a file path.
  ///
  /// The default implementation reads the file, attempts to load from the
  /// binary cache, and falls back to loadLibraryFromText() or
  /// loadLibraryFromData().
  virtual ghost::Library loadLibraryFromFile(const std::string& filename) const;

  virtual void activate(void** prevOut = nullptr) {
    if (prevOut) *prevOut = nullptr;
  }

  virtual void deactivate(void* prev = nullptr) { (void)prev; }

  virtual SharedContext shareContext() const = 0;
  virtual ghost::Stream createStream(
      const StreamOptions& options = {}) const = 0;

  /// @brief Get the current memory pool size. Default returns the stored pool
  /// size.
  virtual size_t getMemoryPoolSize() const;

  /// @brief Set the memory pool size. Default stores the value for
  /// getMemoryPoolSize().
  virtual void setMemoryPoolSize(size_t bytes);

  /// @brief Allocate page-locked host memory. Default uses malloc().
  virtual void* allocateHostMemory(size_t bytes) const;

  /// @brief Free page-locked host memory. Default uses free().
  virtual void freeHostMemory(void* ptr) const;

  virtual ghost::Buffer allocateBuffer(
      size_t bytes, const BufferOptions& opts = {}) const = 0;
  virtual ghost::MappedBuffer allocateMappedBuffer(
      size_t bytes, const BufferOptions& opts = {}) const = 0;
  virtual ghost::Image allocateImage(const ImageDescription& descr) const = 0;
  virtual ghost::Image sharedImage(const ImageDescription& descr,
                                   ghost::Buffer& buffer) const = 0;
  virtual ghost::Image sharedImage(const ImageDescription& descr,
                                   ghost::Image& image) const = 0;

  virtual Attribute getAttribute(DeviceAttributeId what) const = 0;

  /// @brief Get the thread pool used by this device for CPU-side parallel
  /// dispatch.
  ///
  /// CPU backends return the active pool. GPU backends return @c nullptr by
  /// default; subclasses may override.
  virtual std::shared_ptr<ghost::ThreadPool> threadPool() const {
    return nullptr;
  }

 private:
  size_t _poolSize;
  static BinaryCache _cache;
};
}  // namespace implementation
}  // namespace ghost

#endif
