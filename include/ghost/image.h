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
/// @brief 3D origin (x, y, z) for image region offsets.
typedef Value3<size_t> Origin3;
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

/// @brief Texture filter mode for image sampling.
///
/// Controls how texels are read when sampling coordinates fall between texel
/// centers. On CUDA this maps to @c CUfilter_mode when creating texture
/// objects. On OpenCL and Metal, samplers are declared kernel-side so this
/// is informational. On Vulkan and DirectX it will map to the corresponding
/// sampler descriptor fields.
enum class FilterMode {
  /// @brief Return the nearest texel (point sampling).
  Nearest,
  /// @brief Linearly interpolate between neighboring texels.
  Linear
};

/// @brief Texture address mode for out-of-range coordinates.
///
/// Controls what happens when sampling coordinates fall outside [0, 1]
/// (normalized) or [0, dim) (unnormalized).
enum class AddressMode {
  /// @brief Clamp coordinates to the valid range.
  Clamp,
  /// @brief Wrap (repeat) coordinates.
  Wrap,
  /// @brief Mirror coordinates at boundaries.
  Mirror
};

/// @brief Describes how an image should be sampled when bound as a kernel
/// argument.
///
/// Carries filter mode, address mode, and coordinate normalization. This is
/// attached to an Attribute via @c Image::sample() and its fluent modifiers
/// rather than to the ImageDescription, because the same image may be sampled
/// differently in different kernel invocations.
struct SamplerDescription {
  /// @brief Texel filter mode (default: nearest / point sampling).
  FilterMode filter = FilterMode::Nearest;
  /// @brief Address mode for out-of-range coordinates (default: clamp).
  AddressMode address = AddressMode::Clamp;
  /// @brief Whether coordinates are normalized to [0, 1] (default: false).
  bool normalizedCoords = false;
};

/// @brief Buffer and image access modes.
enum class Access {
  /// @brief Read-only access.
  ReadOnly,
  /// @brief Write-only access.
  WriteOnly,
  /// @brief Read-write access.
  ReadWrite
};

/// @brief Allocation tier hint for buffers.
///
/// Describes the intended lifetime and host/device residency of an allocation.
/// Backends use this to select an appropriate memory pool or storage mode.
/// Orthogonal to @c Access, which describes what the kernel will do with the
/// buffer.
enum class AllocHint {
  /// @brief Backend chooses. Preserves current behavior.
  Default,
  /// @brief Long-lived device-local memory (e.g. model weights uploaded once).
  Persistent,
  /// @brief Short-lived, prefer a pooled / recycled allocation.
  Transient,
  /// @brief Host-visible staging memory for CPU<->GPU transfers. The staging
  /// direction is inferred from @c Access: ReadOnly implies host-write /
  /// kernel-read (upload), WriteOnly implies kernel-write / host-read
  /// (readback).
  Staging,
};

/// @brief Options controlling buffer allocation.
///
/// Combines kernel-side access mode with lifetime / residency hint. Implicitly
/// constructible from an @c Access value so that callers that only care about
/// access mode can continue to write @c allocateBuffer(n, Access::ReadOnly).
struct BufferOptions {
  /// @brief Kernel-side access mode.
  Access access = Access::ReadWrite;
  /// @brief Lifetime / residency hint.
  AllocHint hint = AllocHint::Default;

  constexpr BufferOptions() = default;

  /* implicit */ constexpr BufferOptions(Access a) : access(a) {}

  constexpr BufferOptions(Access a, AllocHint h) : access(a), hint(h) {}

  /* implicit */ constexpr BufferOptions(AllocHint h) : hint(h) {}
};

/// @brief Describes the memory layout of a linear buffer for image copy
/// operations.
///
/// Used when copying between buffers and images to specify the region
/// dimensions and row/slice strides. A stride of 0 indicates tight packing (no
/// padding).
class BufferLayout {
 public:
  /// @brief Region dimensions (width, height, depth).
  Size3 size;
  /// @brief Row stride (x) and slice stride (y) in bytes. Zero means tight
  /// packing.
  Stride2 stride;

  /// @brief Construct a buffer layout.
  /// @param size_ Region dimensions (width, height, depth).
  /// @param stride_ Row and slice strides in bytes (default: tight packing).
  BufferLayout(Size3 size_, Stride2 stride_ = Stride2(0, 0))
      : size(size_), stride(stride_) {}
};

/// @brief Descriptor for a 1D, 2D, or 3D GPU image.
///
/// Specifies the dimensions, pixel format, memory layout, and access mode
/// for image allocation and copy operations. Set the z component of @c size
/// to 1 for 2D images, or both y and z to 1 for 1D images.
///
/// Inherits BufferLayout (size + stride) so it can be passed directly to
/// copy methods that accept a BufferLayout.
class ImageDescription : public BufferLayout {
 public:
  /// @brief Number of channels per pixel.
  size_t channels;
  /// @brief Channel ordering within each pixel.
  PixelOrder order;
  /// @brief Data type of each channel element.
  DataType type;
  /// @brief Intended access mode for the image.
  Access access;

  /// @brief Construct an image description.
  /// @param size_ Image dimensions (width, height, depth).
  /// @param order_ Pixel channel ordering.
  /// @param type_ Data type of each channel element.
  /// @param stride_ Row and slice strides in bytes.
  /// @param access_ Access mode (default: read-write).
  ImageDescription(Size3 size_, PixelOrder order_, DataType type_,
                   Stride2 stride_, Access access_ = Access::ReadWrite)
      : BufferLayout(size_, stride_),
        channels(4),
        order(order_),
        type(type_),
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
