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
class Buffer;
class Image;

class Attribute {
 public:
  enum Type {
    Type_Unknown,
    Type_String,
    Type_Float,
    Type_Int,
    Type_Bool,
    Type_Buffer,
    Type_Image,
    Type_LocalMem
  };

 private:
  Type _type;
  size_t _count;

  union {
    float f[4];
    int32_t i[4];
    uint32_t u[4];
    bool b[4];
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
  Attribute() : _type(Type_Unknown), _count(0) {}

  Attribute(const char* s) : _type(Type_String), _count(1), _s(s) {}

  Attribute(const std::string& s) : _type(Type_String), _count(1), _s(s) {}

  Attribute(Buffer* b) : _type(Type_Buffer), _count(1) { _u.buffer = b; }

  Attribute(Buffer& b) : _type(Type_Buffer), _count(1) { _u.buffer = &b; }

  Attribute(Image* i) : _type(Type_Image), _count(1) { _u.image = i; }

  Attribute(Image& i) : _type(Type_Image), _count(1) { _u.image = &i; }

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

  template <typename T>
  Attribute(const T* v, size_t num) {
    set(v, num);
  }

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

  Attribute& localMem(uint32_t bytes) {
    _type = Type_LocalMem;
    _count = 1;
    _u.u[0] = bytes;
    return *this;
  }

  bool valid() const { return _type != Type_Unknown; }

  Type type() const { return _type; }

  size_t count() const { return _count; }

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
};

}  // namespace ghost

#endif
