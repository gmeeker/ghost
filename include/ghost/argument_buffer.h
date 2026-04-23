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

#ifndef GHOST_ARGUMENT_BUFFER_H
#define GHOST_ARGUMENT_BUFFER_H

#include <ghost/device.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace ghost {

/// @brief A host-side buffer for packing kernel parameters into a struct.
///
/// ArgumentBuffer allows grouping multiple scalar kernel arguments into a
/// single packed struct. It supports two modes of passing data to kernels:
///
/// **Struct mode** (default, no upload): The packed data is passed directly
/// as a kernel argument by value. Suitable for small parameter structs.
/// @code
/// ghost::ArgumentBuffer ab;
/// ab.set(0, 1.0f);
/// ab.set(4, 42);
/// fn(stream, launchArgs, ab);  // passed as struct by value
/// @endcode
///
/// **Buffer mode** (after upload): The data is copied to a GPU buffer and
/// the buffer is bound as a kernel argument. Required for large structs or
/// when the kernel expects a pointer to device memory.
/// @code
/// ghost::ArgumentBuffer ab;
/// ab.set(0, 1.0f);
/// ab.set(4, 42);
/// ab.upload(device, stream);
/// fn(stream, launchArgs, ab);  // passed as buffer pointer
/// @endcode
class ArgumentBuffer {
 public:
  /// @brief Construct an empty argument buffer.
  ArgumentBuffer();

  /// @brief Reset the buffer, clearing all packed data and GPU buffer.
  void reset();

  /// @brief Get the current packed data size in bytes.
  size_t size() const;

  /// @brief Get a pointer to the packed host data.
  const void* data() const;

  /// @brief Set a scalar value at a byte offset in the packed data.
  /// @tparam T Scalar type (float, int, etc.).
  /// @param offset Byte offset into the packed struct.
  /// @param value The value to write.
  template <typename T>
  void set(size_t offset, const T& value) {
    ensureSize(offset + sizeof(T));
    memcpy(_data.data() + offset, &value, sizeof(T));
  }

  /// @brief Upload the packed data to a GPU buffer.
  ///
  /// After calling this, the argument buffer will be passed as a GPU buffer
  /// argument rather than a by-value struct. Allocates a device buffer on
  /// first call (or if the data has grown), then copies the host data.
  /// @param device The device to allocate the buffer on.
  /// @param stream The stream to enqueue the copy on.
  void upload(const Device& device, const Encoder& stream);

  /// @brief Check whether this argument buffer should be passed as a struct.
  ///
  /// Returns @c true if upload() has not been called (or was reset),
  /// meaning the host data should be passed directly as a kernel argument.
  /// Returns @c false if upload() has been called, meaning the GPU buffer
  /// should be bound instead.
  bool isStruct() const;

  /// @brief Get the underlying GPU buffer implementation.
  ///
  /// Valid only after upload() has been called.
  /// @return Shared pointer to the buffer implementation, or nullptr.
  std::shared_ptr<implementation::Buffer> bufferImpl() const;

 private:
  void ensureSize(size_t minSize);

  std::vector<uint8_t> _data;
  Buffer _gpuBuffer;
};

}  // namespace ghost

#endif
