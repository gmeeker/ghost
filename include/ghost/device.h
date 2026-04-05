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

#include <ghost/event.h>
#include <ghost/function.h>
#include <ghost/image.h>
#include <ghost/implementation/impl_device.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace ghost {

/// @brief A GPU command queue for sequencing kernel dispatches and memory
/// operations.
///
/// Streams represent an ordered sequence of operations on a device. Multiple
/// streams allow concurrent execution of independent work. Obtain streams via
/// Device::createStream() or Device::defaultStream().
class Stream {
 public:
  /// @brief Default-construct a null stream.
  Stream() = default;

  /// @brief Construct from a backend implementation.
  /// @param impl Shared pointer to the backend-specific stream implementation.
  Stream(std::shared_ptr<implementation::Stream> impl);

  /// @brief Check whether this stream holds a valid implementation.
  explicit operator bool() const { return _impl != nullptr; }

  /// @brief Get the backend implementation (const).
  std::shared_ptr<implementation::Stream> impl() const { return _impl; }

  /// @brief Get the backend implementation (mutable).
  std::shared_ptr<implementation::Stream>& impl() { return _impl; }

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

 private:
  std::shared_ptr<implementation::Stream> _impl;
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
  void copy(const Stream& s, const Buffer& src, size_t bytes);

  /// @brief Copy data from host memory into this buffer.
  /// @param s The stream to enqueue the copy on.
  /// @param src Pointer to host source data.
  /// @param bytes Number of bytes to copy.
  void copy(const Stream& s, const void* src, size_t bytes);

  /// @brief Copy data from this buffer to host memory.
  /// @param s The stream to enqueue the copy on.
  /// @param[out] dst Pointer to host destination buffer.
  /// @param bytes Number of bytes to copy.
  void copyTo(const Stream& s, void* dst, size_t bytes) const;

  /// @brief Copy data from another device buffer with offsets.
  void copy(const Stream& s, const Buffer& src, size_t srcOffset,
            size_t dstOffset, size_t bytes);

  /// @brief Copy data from host memory into this buffer at an offset.
  void copy(const Stream& s, const void* src, size_t dstOffset, size_t bytes);

  /// @brief Copy data from this buffer at an offset to host memory.
  void copyTo(const Stream& s, void* dst, size_t srcOffset, size_t bytes) const;

  /// @brief Fill a region of this buffer with a byte value.
  void fill(const Stream& s, size_t offset, size_t size, uint8_t value);

  /// @brief Fill a region of this buffer with a 32-bit value.
  void fill(const Stream& s, size_t offset, size_t size, uint32_t value);

  /// @brief Fill a region of this buffer with a pattern.
  void fill(const Stream& s, size_t offset, size_t size, const void* pattern,
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
  void* map(const Stream& s, Access access, bool sync = true);

  /// @brief Unmap the buffer, making it available to GPU kernels again.
  /// @param s The stream to synchronize with.
  void unmap(const Stream& s);
};

/// @brief A GPU image (texture) supporting 1D, 2D, and 3D formats.
///
/// Images are allocated via Device::allocateImage() or created as shared
/// views via Device::sharedImage(). Data is transferred using copy() and
/// copyTo() with an ImageDescription specifying the pixel format and layout.
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

  /// @brief Copy data from another image into this image.
  /// @param s The stream to enqueue the copy on.
  /// @param src Source image to copy from.
  void copy(const Stream& s, const Image& src);

  /// @brief Copy data from a buffer into this image.
  /// @param s The stream to enqueue the copy on.
  /// @param src Source buffer containing pixel data.
  /// @param descr Image description specifying dimensions and format.
  void copy(const Stream& s, const Buffer& src, const ImageDescription& descr);

  /// @brief Copy data from host memory into this image.
  /// @param s The stream to enqueue the copy on.
  /// @param src Pointer to host source pixel data.
  /// @param descr Image description specifying dimensions and format.
  void copy(const Stream& s, const void* src, const ImageDescription& descr);

  /// @brief Copy image data into a buffer.
  /// @param s The stream to enqueue the copy on.
  /// @param[out] dst Destination buffer.
  /// @param descr Image description specifying dimensions and format.
  void copyTo(const Stream& s, Buffer& dst,
              const ImageDescription& descr) const;

  /// @brief Copy image data to host memory.
  /// @param s The stream to enqueue the copy on.
  /// @param[out] dst Pointer to host destination buffer.
  /// @param descr Image description specifying dimensions and format.
  void copyTo(const Stream& s, void* dst, const ImageDescription& descr) const;

  /// @brief Copy a region from a buffer into this image at the given origin.
  /// @param s The stream to enqueue the copy on.
  /// @param src Source buffer containing pixel data.
  /// @param descr Image description specifying the region dimensions and
  /// format.
  /// @param imageOrigin Destination origin within this image (x, y, z).
  void copy(const Stream& s, const Buffer& src, const ImageDescription& descr,
            const Size3& imageOrigin);

  /// @brief Copy a region from this image at the given origin into a buffer.
  /// @param s The stream to enqueue the copy on.
  /// @param[out] dst Destination buffer.
  /// @param descr Image description specifying the region dimensions and
  /// format.
  /// @param imageOrigin Source origin within this image (x, y, z).
  void copyTo(const Stream& s, Buffer& dst, const ImageDescription& descr,
              const Size3& imageOrigin) const;

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
  /// @brief Export the device context for sharing with another Device instance.
  /// @return A SharedContext containing backend-specific handles.
  SharedContext shareContext() const;

  /// @brief Get the global binary cache instance.
  /// @return Reference to the shared BinaryCache.
  static BinaryCache& binaryCache();

  /// @brief Purge cached compiled binaries older than @p days.
  /// @param days Maximum age in days (default 30).
  void purgeBinaries(int days = 30);

  /// @brief Load a compiled GPU program from a file.
  /// @param filename Path to the source or binary file.
  /// @return The loaded Library.
  Library loadLibraryFromFile(const std::string& filename);

  /// @brief Compile a GPU program from source text.
  /// @param text Source code string (e.g., OpenCL C, Metal Shading Language).
  /// @param options Compiler options (backend-specific, default empty).
  /// @param retainBinary If true, retain compiled binary for
  /// Library::getBinary().
  /// @return The compiled Library.
  Library loadLibraryFromText(const std::string& text,
                              const std::string& options = "",
                              bool retainBinary = false) const;

  /// @brief Load a GPU program from pre-compiled binary data.
  /// @param data Pointer to the binary data.
  /// @param len Length of the binary data in bytes.
  /// @param options Backend-specific options (default empty).
  /// @param retainBinary If true, retain binary data for Library::getBinary().
  /// @return The loaded Library.
  Library loadLibraryFromData(const void* data, size_t len,
                              const std::string& options = "",
                              bool retainBinary = false) const;

  /// @brief Create a new stream for enqueuing operations.
  /// @return A new Stream.
  Stream createStream() const;

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
  /// @param access Access mode (default: read-write).
  /// @return The allocated Buffer.
  Buffer allocateBuffer(size_t bytes, Access access = Access_ReadWrite) const;

  /// @brief Allocate a memory-mapped GPU buffer.
  /// @param bytes Size in bytes.
  /// @param access Access mode (default: read-write).
  /// @return The allocated MappedBuffer.
  MappedBuffer allocateMappedBuffer(size_t bytes,
                                    Access access = Access_ReadWrite) const;

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

  /// @brief Get the backend implementation (const).
  std::shared_ptr<implementation::Device> impl() const { return _impl; }

  /// @brief Get the backend implementation (mutable).
  std::shared_ptr<implementation::Device>& impl() { return _impl; }

 private:
  std::shared_ptr<implementation::Device> _impl;
  Stream _stream;
};
}  // namespace ghost

#endif