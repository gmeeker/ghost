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

#include <stdint.h>

#include <string>

namespace ghost {
class ArgumentBuffer;
class Buffer;
class Image;

/// @brief Type-safe tagged union for passing kernel arguments and querying
/// device/function metadata.
///
/// An Attribute can hold one of several types: string, float, integer, boolean,
/// buffer pointer, image pointer, or a local memory size. Numeric types support
/// up to 4-element vectors, stored in both 32-bit and 64-bit representations
/// for convenient access.
///
/// Attributes are used in two contexts:
/// - As kernel arguments passed to Function::operator() and
/// Function::execute().
/// - As return values from Device::getAttribute() and Function::getAttribute().
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
    /// @brief Integer value(s) (up to 4 elements, signed or unsigned).
    Type_Int,
    /// @brief Boolean value(s) (up to 4 elements).
    Type_Bool,
    /// @brief Pointer to a Buffer.
    Type_Buffer,
    /// @brief Pointer to an Image.
    Type_Image,
    /// @brief Local (shared) memory allocation size in bytes.
    Type_LocalMem,
    /// @brief Pointer to an ArgumentBuffer.
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
    ArgumentBuffer* argumentBuffer;
    Buffer* buffer;
    Image* image;
  } _u;

  union {
    double f[4];
    int64_t i[4];
    uint64_t u[4];
    bool b[4];
  } _u64;

  std::string _s;

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
  Attribute() : _type(Type_Unknown), _count(0) {}

  /// @name String constructors
  /// @{
  Attribute(char* s) : _type(Type_String), _count(1), _s(s) {}

  Attribute(const char* s) : _type(Type_String), _count(1), _s(s) {}

  Attribute(const std::string& s) : _type(Type_String), _count(1), _s(s) {}

  /// @}

  /// @name Buffer and Image constructors
  /// @{
  Attribute(Buffer* b) : _type(Type_Buffer), _count(1) { _u.buffer = b; }

  Attribute(Buffer& b) : _type(Type_Buffer), _count(1) { _u.buffer = &b; }

  Attribute(Image* i) : _type(Type_Image), _count(1) { _u.image = i; }

  Attribute(Image& i) : _type(Type_Image), _count(1) { _u.image = &i; }

  Attribute(ArgumentBuffer* ab) : _type(Type_ArgumentBuffer), _count(1) {
    _u.argumentBuffer = ab;
  }

  Attribute(ArgumentBuffer& ab) : _type(Type_ArgumentBuffer), _count(1) {
    _u.argumentBuffer = &ab;
  }

  /// @}

  /// @name Scalar and vector numeric constructors
  /// Construct from 1 to 4 numeric values (float, double, int32_t, uint32_t,
  /// int64_t, uint64_t, or bool). The type is inferred from the argument type.
  /// @{
  template <typename T>
  Attribute(T v) {
    set(&v, 1);
  }

  template <typename T>
  Attribute(T v0, T v1) {
    T v[] = {v0, v1};
    set(v, 2);
  }

  template <typename T>
  Attribute(T v0, T v1, T v2) {
    T v[] = {v0, v1, v2};
    set(v, 3);
  }

  template <typename T>
  Attribute(T v0, T v1, T v2, T v3) {
    T v[] = {v0, v1, v2, v3};
    set(v, 4);
  }

  /// @brief Construct from an array of @p num values.
  /// @tparam T Numeric type (float, double, int32_t, uint32_t, int64_t,
  /// uint64_t, or bool).
  /// @param v Pointer to the array of values.
  /// @param num Number of elements (1–4).
  template <typename T>
  Attribute(const T* v, size_t num) {
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
    _type = Type_Int;
    setT(v, (uint32_t)0, _u.u, _u64.u, num);
  }

  void set(const int64_t* v, size_t num) {
    _type = Type_Int;
    setT(v, (int64_t)0, _u64.i, _u.i, num);
  }

  void set(const uint64_t* v, size_t num) {
    _type = Type_Int;
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

  Buffer* asBuffer() const { return _u.buffer; }

  Image* asImage() const { return _u.image; }

  ArgumentBuffer* asArgumentBuffer() const { return _u.argumentBuffer; }

  /// @}
};

}  // namespace ghost

#endif
