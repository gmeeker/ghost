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

#ifndef GHOST_IMAGE_H
#define GHOST_IMAGE_H

#include <stdint.h>
#include <stdlib.h>

namespace ghost {

/// @brief A 2-component value (x, y).
/// @tparam T The component type.
template <typename T>
class Value2 {
 public:
  T x, y;

  Value2(T x_, T y_) : x(x_), y(y_) {}
};

/// @brief A 3-component value (x, y, z).
/// @tparam T The component type.
template <typename T>
class Value3 {
 public:
  T x, y, z;

  Value3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
};

/// @brief 2D integer point.
typedef Value2<int> Point2;
/// @brief 2D size (width, height).
typedef Value2<size_t> Size2;
/// @brief 3D integer point.
typedef Value3<int> Point3;
/// @brief 3D size (width, height, depth).
typedef Value3<size_t> Size3;
/// @brief Row and slice strides in bytes.
typedef Value2<int32_t> Stride2;

/// @brief Pixel data types for image elements.
enum DataType {
  /// @brief Unsigned 8-bit integer.
  DataType_UInt8,
  /// @brief Signed 8-bit integer.
  DataType_Int8,
  /// @brief Unsigned 16-bit integer.
  DataType_UInt16,
  /// @brief Signed 16-bit integer.
  DataType_Int16,
  /// @brief 16-bit floating point (half).
  DataType_Float16,
  /// @brief 32-bit floating point.
  DataType_Float,
  /// @brief 64-bit floating point.
  DataType_Double
};

/// @brief Pixel channel ordering, packed as 2-bit indices per channel.
typedef uint32_t PixelOrder;

/// @name Predefined pixel orderings
/// @{
enum {
  /// @brief RGBA ordering: R=0, G=1, B=2, A=3.
  PixelOrder_RGBA = ((0 << 6) + (1 << 4) + (2 << 2) + (3 << 0)),
  /// @brief ARGB ordering: A=0, R=1, G=2, B=3.
  PixelOrder_ARGB = ((1 << 6) + (2 << 4) + (3 << 2) + (0 << 0)),
  /// @brief ABGR ordering: A=0, B=1, G=2, R=3.
  PixelOrder_ABGR = ((3 << 6) + (2 << 4) + (1 << 2) + (0 << 0)),
  /// @brief BGRA ordering: B=0, G=1, R=2, A=3.
  PixelOrder_BGRA = ((2 << 6) + (1 << 4) + (0 << 2) + (3 << 0))
};

/// @}

/// @brief Buffer and image access modes.
enum Access {
  /// @brief Read-only access.
  Access_ReadOnly,
  /// @brief Write-only access.
  Access_WriteOnly,
  /// @brief Read-write access.
  Access_ReadWrite
};

/// @brief Descriptor for a 1D, 2D, or 3D GPU image.
///
/// Specifies the dimensions, pixel format, memory layout, and access mode
/// for image allocation and copy operations. Set the z component of @c size
/// to 1 for 2D images, or both y and z to 1 for 1D images.
class ImageDescription {
 public:
  /// @brief Image dimensions (width, height, depth).
  Size3 size;
  /// @brief Number of channels per pixel.
  size_t channels;
  /// @brief Channel ordering within each pixel.
  PixelOrder order;
  /// @brief Data type of each channel element.
  DataType type;
  /// @brief Row stride (x) and slice stride (y) in bytes.
  Stride2 stride;
  /// @brief Intended access mode for the image.
  Access access;

  /// @brief Construct an image description.
  /// @param size_ Image dimensions (width, height, depth).
  /// @param order_ Pixel channel ordering.
  /// @param type_ Data type of each channel element.
  /// @param stride_ Row and slice strides in bytes.
  /// @param access_ Access mode (default: read-write).
  ImageDescription(Size3 size_, PixelOrder order_, DataType type_,
                   Stride2 stride_, Access access_ = Access_ReadWrite)
      : size(size_),
        order(order_),
        type(type_),
        stride(stride_),
        access(access_) {}

  /// @brief Compute the total data size of the image in bytes.
  /// @return Total byte count for the image data.
  size_t dataSize() const;

  /// @brief Compute the size of a single pixel in bytes.
  /// @return Byte count per pixel (channels * element size).
  size_t pixelSize() const;
};
}  // namespace ghost

#endif
