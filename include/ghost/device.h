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

#ifndef GHOST_DEVICE_H
#define GHOST_DEVICE_H

#include <ghost/encoder.h>
#include <ghost/event.h>
#include <ghost/exception.h>
#include <ghost/function.h>
#include <ghost/image.h>
#include <ghost/implementation/impl_device.h>
#include <ghost/thread_pool.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ghost {

/// @brief A GPU command queue for sequencing kernel dispatches and memory
/// operations.
///
/// Streams represent an ordered sequence of operations on a device. Multiple
/// streams allow concurrent execution of independent work. Obtain streams via
/// Device::createStream() or Device::defaultStream().
class Stream : public Encoder {
 public:
  /// @brief Default-construct a null stream.
  Stream() = default;

  /// @brief Construct from a backend implementation.
  /// @param impl Shared pointer to the backend-specific stream implementation.
  Stream(std::shared_ptr<implementation::Stream> impl);

  /// @brief Block until all operations enqueued on this stream have completed.
  void sync();

  /// @brief Record an event at the current point in this stream.
  ///
  /// The returned event will be signaled once all operations enqueued before
  /// this call have completed on the GPU.
  /// @return An Event that tracks completion of prior operations.
  Event record();

  /// @brief Enqueue a GPU-side wait for an event from another stream.
  ///
  /// Subsequent operations on this stream will not begin until @p e has
  /// completed. This does not block the CPU.
  /// @param e The event to wait for (typically recorded on a different stream).
  void waitForEvent(const Event& e);
};

/// @brief A GPU memory buffer for storing data accessible by kernels.
///
/// Buffers are allocated via Device::allocateBuffer(). Data is transferred
/// between host and device using copy() and copyTo().
class Buffer {
 public:
  /// @brief Default-construct a null buffer.
  Buffer() = default;

  /// @brief Construct from a backend implementation.
  /// @param impl Shared pointer to the backend-specific buffer implementation.
  Buffer(std::shared_ptr<implementation::Buffer> impl);

  /// @brief Check whether this buffer holds a valid implementation.
  explicit operator bool() const { return _impl != nullptr; }

  /// @brief Get the backend implementation (const).
  std::shared_ptr<implementation::Buffer> impl() const { return _impl; }

  /// @brief Get the backend implementation (mutable).
  std::shared_ptr<implementation::Buffer>& impl() { return _impl; }

  /// @brief Get the size of this buffer in bytes.
  size_t size() const;

  /// @brief Copy data from another device buffer into this buffer.
  /// @param s The stream to enqueue the copy on.
  /// @param src Source buffer to copy from.
  /// @param bytes Number of bytes to copy.
  void copy(const Encoder& s, const Buffer& src, size_t bytes);

  /// @brief Copy data from host memory into this buffer.
  /// @param s The stream to enqueue the copy on.
  /// @param src Pointer to host source data.
  /// @param bytes Number of bytes to copy.
  void copy(const Encoder& s, const void* src, size_t bytes);

  /// @brief Copy data from this buffer to host memory.
  /// @param s The stream to enqueue the copy on.
  /// @param[out] dst Pointer to host destination buffer.
  /// @param bytes Number of bytes to copy.
  void copyTo(const Encoder& s, void* dst, size_t bytes) const;

  /// @brief Copy data from another device buffer with offsets.
  void copy(const Encoder& s, const Buffer& src, size_t srcOffset,
            size_t dstOffset, size_t bytes);

  /// @brief Copy data from host memory into this buffer at an offset.
  void copy(const Encoder& s, const void* src, size_t dstOffset, size_t bytes);

  /// @brief Copy data from this buffer at an offset to host memory.
  void copyTo(const Encoder& s, void* dst, size_t srcOffset,
              size_t bytes) const;

  /// @brief Fill a region of this buffer with a byte value.
  void fill(const Encoder& s, size_t offset, size_t size, uint8_t value);

  /// @brief Fill a region of this buffer with a 32-bit value.
  void fill(const Encoder& s, size_t offset, size_t size, uint32_t value);

  /// @brief Fill a region of this buffer with a pattern.
  void fill(const Encoder& s, size_t offset, size_t size, const void* pattern,
            size_t patternSize);

  /// @brief Create a sub-buffer view into this buffer.
  ///
  /// The returned buffer shares memory with this buffer, starting at
  /// @p offset. Writes to either buffer are visible in the other.
  /// @param offset Byte offset into this buffer.
  /// @param size Size of the sub-buffer in bytes.
  /// @return A Buffer viewing the specified region.
  Buffer createSubBuffer(size_t offset, size_t size);

 private:
  std::shared_ptr<implementation::Buffer> _impl;
};

/// @brief A GPU buffer that can be memory-mapped for direct host access.
///
/// Allocated via Device::allocateMappedBuffer(). After mapping, the host
/// can read from or write to the buffer directly. The buffer must be unmapped
/// before it can be used by GPU kernels again.
class MappedBuffer : public Buffer {
 public:
  /// @brief Default-construct a null mapped buffer.
  MappedBuffer() = default;

  /// @brief Construct from a backend implementation.
  /// @param impl Shared pointer to the backend-specific buffer implementation.
  MappedBuffer(std::shared_ptr<implementation::Buffer> impl);

  /// @brief Map the buffer into host address space.
  /// @param s The stream to synchronize with.
  /// @param access The intended access mode (read, write, or read-write).
  /// @param sync If @c true, block until previous GPU operations on this buffer
  /// complete.
  /// @return Pointer to the mapped host memory.
  void* map(const Encoder& s, Access access, bool sync = true);

  /// @brief Unmap the buffer, making it available to GPU kernels again.
  /// @param s The stream to synchronize with.
  void unmap(const Encoder& s);
};

/// @brief A GPU image (texture) supporting 1D, 2D, and 3D formats.
///
/// Images are allocated via Device::allocateImage() or created as shared
/// views via Device::sharedImage(). Data is transferred using copy() and
/// copyTo(). Convenience overloads assume tight-packed buffers at the image's
/// full size; explicit BufferLayout overloads allow custom strides and region
/// sizes.
class Image {
 public:
  /// @brief Default-construct a null image.
  Image() = default;

  /// @brief Construct from a backend implementation.
  /// @param impl Shared pointer to the backend-specific image implementation.
  Image(std::shared_ptr<implementation::Image> impl);

  /// @brief Check whether this image holds a valid implementation.
  explicit operator bool() const { return _impl != nullptr; }

  /// @brief Get the backend implementation (const).
  std::shared_ptr<implementation::Image> impl() const { return _impl; }

  /// @brief Get the backend implementation (mutable).
  std::shared_ptr<implementation::Image>& impl() { return _impl; }

  /// @brief Get the image description this image was allocated with.
  const ImageDescription& description() const;

  /// @name Full-image copies (convenience, tight-packed buffers)
  /// @{

  /// @brief Copy data from another image into this image.
  void copy(const Encoder& s, const Image& src);

  /// @brief Copy data from a buffer into this image (tight packing assumed).
  void copy(const Encoder& s, const Buffer& src);

  /// @brief Copy data from host memory into this image (tight packing assumed).
  void copy(const Encoder& s, const void* src);

  /// @brief Copy image data into a buffer (tight packing assumed).
  void copyTo(const Encoder& s, Buffer& dst) const;

  /// @brief Copy image data to host memory (tight packing assumed).
  void copyTo(const Encoder& s, void* dst) const;

  /// @}
  /// @name Full-image copies with explicit buffer layout
  /// @{

  /// @brief Copy data from a buffer into this image.
  /// @param layout Buffer memory layout (dimensions and strides).
  void copy(const Encoder& s, const Buffer& src, const BufferLayout& layout);

  /// @brief Copy data from host memory into this image.
  /// @param layout Buffer memory layout (dimensions and strides).
  void copy(const Encoder& s, const void* src, const BufferLayout& layout);

  /// @brief Copy image data into a buffer.
  /// @param layout Buffer memory layout (dimensions and strides).
  void copyTo(const Encoder& s, Buffer& dst, const BufferLayout& layout) const;

  /// @brief Copy image data to host memory.
  /// @param layout Buffer memory layout (dimensions and strides).
  void copyTo(const Encoder& s, void* dst, const BufferLayout& layout) const;

  /// @}
  /// @name Subrect copies (buffer <-> image region)
  /// @{

  /// @brief Copy a region from a buffer into this image at the given origin.
  /// @param layout Buffer memory layout (region dimensions and strides).
  /// @param imageOrigin Destination origin within this image.
  void copy(const Encoder& s, const Buffer& src, const BufferLayout& layout,
            const Origin3& imageOrigin);

  /// @brief Copy a region from this image at the given origin into a buffer.
  /// @param layout Buffer memory layout (region dimensions and strides).
  /// @param imageOrigin Source origin within this image.
  void copyTo(const Encoder& s, Buffer& dst, const BufferLayout& layout,
              const Origin3& imageOrigin) const;

  /// @}
  /// @name Image-to-image subrect copy
  /// @{

  /// @brief Copy a region between images with independent source and
  /// destination origins.
  /// @param src Source image.
  /// @param region Size of the region to copy.
  /// @param srcOrigin Origin within the source image.
  /// @param dstOrigin Origin within this (destination) image.
  void copy(const Encoder& s, const Image& src, const Size3& region,
            const Origin3& srcOrigin, const Origin3& dstOrigin);

  /// @}

  /// @brief Create a sampled-image attribute with default sampler settings.
  ///
  /// Returns an Attribute (Type_Image) carrying a SamplerDescription with
  /// defaults (FilterMode::Nearest, AddressMode::Clamp, unnormalized coords).
  /// Use fluent modifiers to customize:
  /// @code
  /// fn(stream, launch, image.sample().linear().wrap());
  /// @endcode
  Attribute sample() const;

 private:
  std::shared_ptr<implementation::Image> _impl;
};

/// @brief A GPU device providing resource allocation, kernel compilation, and
/// stream management.
///
/// Device is the central object in the Ghost API. It represents a connection to
/// a GPU (or CPU fallback) and provides methods to:
/// - Compile and load GPU programs (loadLibraryFromText(),
/// loadLibraryFromFile(), loadLibraryFromData())
/// - Allocate memory (allocateBuffer(), allocateMappedBuffer(),
/// allocateImage())
/// - Create execution streams (createStream(), defaultStream())
/// - Query device capabilities (getAttribute())
///
/// Backend-specific subclasses (e.g., DeviceMetal, DeviceOpenCL) are
/// constructed directly; the Device base class is not instantiated by users.
class Device {
 protected:
  /// @brief Construct from a backend implementation.
  /// @param impl Shared pointer to the backend-specific device implementation.
  Device(std::shared_ptr<implementation::Device> impl);

  /// @brief Set the default stream for this device.
  /// @param stream The stream to use as the default.
  void setDefaultStream(std::shared_ptr<implementation::Stream> stream);

 public:
  /// @brief Some devices may have a concept of being "current" (CUDA). Active
  /// may be useful when interfacing with other libraries or using multiple
  /// devices. Wrap each public entry point of a library that uses Ghost in an
  /// Active scope so the thread's prior context is preserved for the caller.
  /// On non-CUDA backends Active is a no-op.
  class Active {
   private:
    Device& _dev;
    void* _prev;

   public:
    Active(Device& dev) : _dev(dev), _prev(nullptr) {
      // A library entry point should not inherit deferred errors left on this
      // thread by unrelated prior work. Drop any pending error before we
      // touch the device.
      try {
        detail::drainErrors();
      } catch (...) {
      }
      _dev.activate(&_prev);
    }

    ~Active() noexcept {
      try {
        _dev.deactivate(_prev);
      } catch (...) {
      }
      // Don't let an error deferred during this scope poison the caller's
      // next unrelated Ghost call on this thread.
      try {
        detail::drainErrors();
      } catch (...) {
      }
    }
  };

  /// @brief Export the device context for sharing with another Device instance.
  /// @return A SharedContext containing backend-specific handles.
  SharedContext shareContext() const;

  /// @brief Set this as the thread's current device. If @p prevOut is non-null,
  /// the prior backend-specific context handle is written there so a matching
  /// deactivate() can restore it.
  void activate(void** prevOut = nullptr);

  /// @brief Synchronize the device and restore the thread's previous current
  /// device. @p prev is the handle returned by a paired activate() call.
  void deactivate(void* prev = nullptr);

  /// @brief Get the global binary cache instance.
  /// @return Reference to the shared BinaryCache.
  static BinaryCache& binaryCache();

  /// @brief Purge cached compiled binaries older than @p days.
  /// @param days Maximum age in days (default 30).
  void purgeBinaries(int days = 30);

  /// @brief Load a compiled GPU program from a file.
  /// @param filename Path to the source or binary file.
  /// @return The loaded Library.
  Library loadLibraryFromFile(const std::filesystem::path& filename);

  /// @brief Compile a GPU program from source text.
  /// @param text Source code string (e.g., OpenCL C, Metal Shading Language).
  /// @param options Compiler options (backend-specific, default empty).
  /// @param retainBinary If true, retain compiled binary for
  /// Library::getBinary().
  /// @return The compiled Library.
  Library loadLibraryFromText(
      const std::string& text,
      const CompilerOptions& options = CompilerOptions(),
      bool retainBinary = false) const;

  /// @brief Load a GPU program from pre-compiled binary data.
  /// @param data Pointer to the binary data.
  /// @param len Length of the binary data in bytes.
  /// @param options Backend-specific options (default empty).
  /// @param retainBinary If true, retain binary data for Library::getBinary().
  /// @return The loaded Library.
  Library loadLibraryFromData(
      const void* data, size_t len,
      const CompilerOptions& options = CompilerOptions(),
      bool retainBinary = false) const;

  /// @brief Create a new stream for enqueuing operations.
  /// @param options Optional stream creation flags (profiling, event chaining).
  /// @return A new Stream.
  Stream createStream(const StreamOptions& options = {}) const;

  /// @brief Get the default stream for this device.
  /// @return The default Stream.
  Stream defaultStream() const;

  /// @brief Get the current memory pool size.
  /// @return Pool size in bytes, or 0 if pooling is disabled.
  size_t getMemoryPoolSize() const;

  /// @brief Set the memory pool size for sub-allocation.
  ///
  /// When non-zero, backends may use pooled allocation strategies
  /// (e.g., MTLHeap on Metal, cuMemPool on CUDA) or they may reuse freed
  /// buffers for improved performance.
  /// @param bytes Pool size in bytes, or 0 to disable pooling.
  void setMemoryPoolSize(size_t bytes) const;

  /// @brief Allocate page-locked host memory suitable for fast DMA transfers.
  /// @param bytes Number of bytes to allocate.
  /// @return Pointer to the allocated memory.
  void* allocateHostMemory(size_t bytes) const;

  /// @brief Free memory previously allocated with allocateHostMemory().
  /// @param ptr Pointer to the memory to free.
  void freeHostMemory(void* ptr) const;

  /// @brief Allocate a GPU buffer.
  /// @param bytes Size in bytes.
  /// @param opts Allocation options (access mode and lifetime hint). Implicitly
  /// convertible from @c Access for callers that only care about access mode.
  /// @return The allocated Buffer.
  Buffer allocateBuffer(size_t bytes, BufferOptions opts = {}) const;

  /// @brief Allocate a memory-mapped GPU buffer.
  /// @param bytes Size in bytes.
  /// @param opts Allocation options (access mode and lifetime hint).
  /// @return The allocated MappedBuffer.
  MappedBuffer allocateMappedBuffer(size_t bytes,
                                    BufferOptions opts = {}) const;

  /// @brief Allocate a GPU image (texture).
  /// @param descr Image description specifying dimensions, format, and access.
  /// @return The allocated Image.
  Image allocateImage(const ImageDescription& descr) const;

  /// @brief Create an image that shares memory with an existing buffer.
  /// @param descr Image description specifying dimensions and format.
  /// @param buffer The buffer whose memory backs the image.
  /// @return The shared Image.
  Image sharedImage(const ImageDescription& descr, Buffer& buffer) const;

  /// @brief Create an image that shares memory with another image.
  /// @param descr Image description specifying dimensions and format.
  /// @param image The image whose memory backs the new image.
  /// @return The shared Image.
  Image sharedImage(const ImageDescription& descr, Image& image) const;

  /// @brief Query a device attribute.
  /// @param what The attribute to query (e.g., kDeviceName, kDeviceMemory).
  /// @return The attribute value.
  Attribute getAttribute(DeviceAttributeId what) const;

  /// @brief Get the host-side thread pool used by this device.
  ///
  /// CPU backends return the active worker pool. GPU backends return
  /// @c nullptr — GPU parallelism is via kernel dispatch, not host threads.
  /// Use this when you need to run native C++ code in parallel without
  /// going through @c FunctionCPU.
  std::shared_ptr<ThreadPool> threadPool() const;

  /// @brief Get the backend implementation (const).
  std::shared_ptr<implementation::Device> impl() const { return _impl; }

  /// @brief Get the backend implementation (mutable).
  std::shared_ptr<implementation::Device>& impl() { return _impl; }

 private:
  std::shared_ptr<implementation::Device> _impl;
  Stream _stream;
};
/// @brief Enumeration of GPU compute backends.
enum class Backend { CPU, Metal, OpenCL, CUDA, Vulkan, DirectX };

/// @brief Get the human-readable name of a backend.
/// @param backend The backend to name.
/// @return String name (e.g., "Metal", "CUDA").
std::string backendName(Backend backend);

/// @brief List all backends compiled into this build.
/// @return Vector of available Backend values.
std::vector<Backend> availableBackends();

/// @brief Create a device for a specific backend.
/// @param backend The backend to use.
/// @return A Device instance, or nullptr if the backend is not compiled in
///         or no suitable hardware is available.
std::unique_ptr<Device> createDevice(Backend backend);

/// @brief Create a device using the best available GPU backend.
///
/// Tries backends in platform-preferred order:
/// - macOS: Metal, OpenCL, Vulkan
/// - Windows: CUDA, DirectX, OpenCL, Vulkan
/// - Linux: CUDA, OpenCL, Vulkan
///
/// @param allowCPU If true, falls back to CPU when no GPU backend succeeds.
/// @return A Device instance, or nullptr if no suitable backend is found.
std::unique_ptr<Device> createDevice(bool allowCPU = false);

/// @brief Enumerate all available GPU devices across all compiled backends.
/// @return Vector of GpuInfo descriptors (may be empty).
std::vector<GpuInfo> enumerateDevices();

/// @brief Pick the best available GPU using a default heuristic.
///
/// Prefers discrete GPUs over integrated, then selects by highest VRAM.
/// @param backend If set, only consider devices from this backend.
/// @return The best device descriptor, or std::nullopt if none available.
std::optional<GpuInfo> preferredDevice(
    std::optional<Backend> backend = std::nullopt);

/// @brief Pick the best GPU from a pre-enumerated list.
///
/// Prefers discrete GPUs over integrated, then selects by highest VRAM.
/// @param devices The list to choose from.
/// @return The best device descriptor, or std::nullopt if the list is empty.
std::optional<GpuInfo> preferredDevice(const std::vector<GpuInfo>& devices);

}  // namespace ghost

#endif