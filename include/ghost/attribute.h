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

#ifndef GHOST_ATTRIBUTE_H
#define GHOST_ATTRIBUTE_H

#include <ghost/image.h>
#include <stdint.h>

#include <memory>
#include <optional>
#include <string>

namespace ghost {
class ArgumentBuffer;
class Buffer;
class Image;

namespace implementation {
class Buffer;
class Image;
}  // namespace implementation

/// @brief Type-safe tagged union for passing kernel arguments and querying
/// device/function metadata.
///
/// An Attribute can hold one of several types: string, float, integer, boolean,
/// buffer, image, argument buffer, or a local memory size. Numeric types
/// support up to 4-element vectors, stored in both 32-bit and 64-bit
/// representations for convenient access. Kernel arguments have limited
/// support for 64-bit types we don't define different data sizes.
///
/// Attributes are used in two contexts:
/// - As kernel arguments passed to Function::operator() and
///   Function::execute().
/// - As return values from Device::getAttribute() and Function::getAttribute().
///
/// Buffer/Image/ArgumentBuffer attributes hold strong references to the
/// underlying backend implementations, so they remain valid even if the
/// caller's wrapper objects go out of scope before the kernel actually
/// dispatches. This is required for deferred execution paths like
/// CommandBuffer where the recorded Attribute may outlive the wrapper.
class Attribute {
 public:
  /// @brief The type tag identifying which value the Attribute holds.
  enum Type {
    /// @brief No value set.
    Type_Unknown,
    /// @brief String value.
    Type_String,
    /// @brief Floating-point value(s) (up to 4 elements).
    Type_Float,
    /// @brief Signed integer value(s) (up to 4 elements).
    Type_Int,
    /// @brief Unsigned integer value(s) (up to 4 elements).
    Type_UInt,
    /// @brief Boolean value(s) (up to 4 elements).
    Type_Bool,
    /// @brief Reference to a Buffer (held via shared_ptr to its impl).
    Type_Buffer,
    /// @brief Reference to an Image (held via shared_ptr to its impl).
    Type_Image,
    /// @brief Local (shared) memory allocation size in bytes.
    Type_LocalMem,
    /// @brief Snapshot of an ArgumentBuffer (host data + GPU buffer ref).
    Type_ArgumentBuffer
  };

 private:
  Type _type;
  size_t _count;

  union {
    float f[4];
    int32_t i[4];
    uint32_t u[4];
    bool b[4];
  } _u;

  union {
    double f[4];
    int64_t i[4];
    uint64_t u[4];
    bool b[4];
  } _u64;

  std::string _s;

  // Strong references kept alive for the Attribute's lifetime. These are
  // what make Buffer/Image/ArgumentBuffer Attributes safe to outlive the
  // user's wrapper objects (e.g., when recording into a CommandBuffer and
  // submitting later).
  std::shared_ptr<implementation::Buffer> _bufferImpl;
  std::shared_ptr<implementation::Image> _imageImpl;
  std::shared_ptr<ArgumentBuffer> _argBuffer;
  std::optional<SamplerDescription> _sampler;

  template <typename S, typename T>
  void setT(const S* v, S v0, S* s, T* t, size_t num) {
    _count = num;
    size_t idx;
    for (idx = 0; idx < num; idx++) {
      s[idx] = v[idx];
    }
    for (; idx < 4; idx++) {
      s[idx] = v0;
    }
    for (idx = 0; idx < num; idx++) {
      t[idx] = (T)v[idx];
    }
    for (; idx < 4; idx++) {
      t[idx] = (T)v0;
    }
  }

 public:
  /// @brief Construct an empty attribute with Type_Unknown.
  Attribute();

  ~Attribute();
  Attribute(const Attribute&);
  Attribute(Attribute&&) noexcept;
  Attribute& operator=(const Attribute&);
  Attribute& operator=(Attribute&&) noexcept;

  /// @name String constructors
  /// @{
  Attribute(char* s);
  Attribute(const char* s);
  Attribute(const std::string& s);
  /// @}

  /// @name Buffer / Image / ArgumentBuffer constructors
  ///
  /// Each constructor captures a strong reference to the underlying backend
  /// object so the Attribute remains valid for the duration of any deferred
  /// dispatch.
  /// @{
  Attribute(Buffer* b);
  Attribute(Buffer& b);
  Attribute(Image* i);
  Attribute(Image& i);
  /// @brief Construct an image attribute with an explicit sampler description.
  ///
  /// Used by @c Image::sample() to create a sampled image attribute.
  Attribute(Image& i, const SamplerDescription& sampler);
  Attribute(ArgumentBuffer* ab);
  Attribute(ArgumentBuffer& ab);

  /// @}

  /// @name Scalar and vector numeric constructors
  /// Construct from 1 to 4 numeric values (float, double, int32_t, uint32_t,
  /// int64_t, uint64_t, or bool). The type is inferred from the argument type.
  /// @{
  template <typename T>
  Attribute(T v) : _type(Type_Unknown), _count(0) {
    set(&v, 1);
  }

  template <typename T>
  Attribute(T v0, T v1) : _type(Type_Unknown), _count(0) {
    T v[] = {v0, v1};
    set(v, 2);
  }

  template <typename T>
  Attribute(T v0, T v1, T v2) : _type(Type_Unknown), _count(0) {
    T v[] = {v0, v1, v2};
    set(v, 3);
  }

  template <typename T>
  Attribute(T v0, T v1, T v2, T v3) : _type(Type_Unknown), _count(0) {
    T v[] = {v0, v1, v2, v3};
    set(v, 4);
  }

  /// @brief Construct from an array of @p num values.
  /// @tparam T Numeric type (float, double, int32_t, uint32_t, int64_t,
  /// uint64_t, or bool).
  /// @param v Pointer to the array of values.
  /// @param num Number of elements (1–4).
  template <typename T>
  Attribute(const T* v, size_t num) : _type(Type_Unknown), _count(0) {
    set(v, num);
  }

  /// @}

  /// @name Typed setters
  /// Set the attribute value from an array. Each overload sets the type tag
  /// and stores the values in both 32-bit and 64-bit representations.
  /// @{
  void set(const float* v, size_t num) {
    _type = Type_Float;
    setT(v, 0.f, _u.f, _u64.f, num);
  }

  void set(const double* v, size_t num) {
    _type = Type_Float;
    setT(v, 0.0, _u64.f, _u.f, num);
  }

  void set(const int32_t* v, size_t num) {
    _type = Type_Int;
    setT(v, (int32_t)0, _u.i, _u64.i, num);
  }

  void set(const uint32_t* v, size_t num) {
    _type = Type_UInt;
    setT(v, (uint32_t)0, _u.u, _u64.u, num);
  }

  void set(const int64_t* v, size_t num) {
    _type = Type_Int;
    setT(v, (int64_t)0, _u64.i, _u.i, num);
  }

  void set(const uint64_t* v, size_t num) {
    _type = Type_UInt;
    setT(v, (uint64_t)0, _u64.u, _u.u, num);
  }

  void set(const bool* v, size_t num) {
    _type = Type_Bool;
    setT(v, false, _u.b, _u64.b, num);
  }

  /// @}

  /// @brief Set this attribute to represent a local memory allocation.
  /// @param bytes Size of local (shared) memory in bytes.
  /// @return Reference to this attribute for chaining.
  Attribute& localMem(uint32_t bytes) {
    _type = Type_LocalMem;
    _count = 1;
    _u.u[0] = bytes;
    return *this;
  }

  /// @brief Check whether the attribute holds a value.
  /// @return @c true if the type is not Type_Unknown.
  bool valid() const { return _type != Type_Unknown; }

  /// @brief Get the type tag.
  Type type() const { return _type; }

  /// @brief Get the number of elements (1–4 for numeric types).
  size_t count() const { return _count; }

  /// @name Value accessors
  /// Retrieve the stored value in the requested type. The caller must ensure
  /// the attribute's type matches the accessor used.
  /// @{
  const std::string& asString() const { return _s; }

  const float asFloat() const { return _u.f[0]; }

  const float* floatArray() const { return _u.f; }

  const double asDouble() const { return _u64.f[0]; }

  const double* doubleArray() const { return _u64.f; }

  const int32_t asInt() const { return _u.i[0]; }

  const int32_t* intArray() const { return _u.i; }

  const uint32_t asUInt() const { return _u.u[0]; }

  const uint32_t* uintArray() const { return _u.u; }

  const int64_t asInt64() const { return _u64.i[0]; }

  const int64_t* int64Array() const { return _u64.i; }

  const uint64_t asUInt64() const { return _u64.u[0]; }

  const uint64_t* uint64Array() const { return _u64.u; }

  const bool asBool() const { return _u.b[0]; }

  const bool* boolArray() const { return _u.b; }

  /// @brief Strong reference to the underlying buffer implementation.
  ///
  /// Valid for the lifetime of this Attribute. Backends should use this
  /// instead of dereferencing the user's wrapper, which may have already
  /// been destroyed in deferred execution paths.
  const std::shared_ptr<implementation::Buffer>& bufferImpl() const {
    return _bufferImpl;
  }

  /// @brief Strong reference to the underlying image implementation.
  const std::shared_ptr<implementation::Image>& imageImpl() const {
    return _imageImpl;
  }

  /// @brief Snapshot of the ArgumentBuffer captured at construction time.
  ///
  /// Returns a pointer to a heap-allocated copy of the user's
  /// ArgumentBuffer. The host-side data is snapshotted; the GPU buffer
  /// (if any) shares its impl with the original via shared_ptr.
  ArgumentBuffer* argumentBuffer() const { return _argBuffer.get(); }

  /// @brief Sampler description attached to an image attribute.
  ///
  /// Present only when the attribute was created via @c Image::sample().
  /// Backends that create host-side sampler/texture objects (CUDA, Vulkan,
  /// DirectX) use this to configure filtering and addressing. Backends
  /// where samplers are declared kernel-side (OpenCL, Metal) may ignore it.
  const std::optional<SamplerDescription>& sampler() const { return _sampler; }

  /// @}

  /// @name Sampler fluent modifiers
  ///
  /// These methods modify the sampler description on an image attribute
  /// created via @c Image::sample(). They return @c *this for chaining:
  /// @code
  /// fn(stream, launch, image.sample().linear().clamp());
  /// @endcode
  /// @{

  /// @brief Set the filter mode to linear interpolation.
  Attribute& linear() {
    if (_sampler) _sampler->filter = FilterMode::Linear;
    return *this;
  }

  /// @brief Set the filter mode to nearest (point) sampling.
  Attribute& nearest() {
    if (_sampler) _sampler->filter = FilterMode::Nearest;
    return *this;
  }

  /// @brief Set the address mode to clamp.
  Attribute& clamp() {
    if (_sampler) _sampler->address = AddressMode::Clamp;
    return *this;
  }

  /// @brief Set the address mode to wrap (repeat).
  Attribute& wrap() {
    if (_sampler) _sampler->address = AddressMode::Wrap;
    return *this;
  }

  /// @brief Set the address mode to mirror.
  Attribute& mirror() {
    if (_sampler) _sampler->address = AddressMode::Mirror;
    return *this;
  }

  /// @brief Enable or disable normalized coordinates.
  Attribute& normalized(bool enable = true) {
    if (_sampler) _sampler->normalizedCoords = enable;
    return *this;
  }

  /// @}
};

}  // namespace ghost

#endif
