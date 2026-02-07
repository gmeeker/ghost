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

  std::string _s;

 public:
  Attribute() : _type(Type_Unknown), _count(0) {}

  Attribute(const std::string& s) : _type(Type_String), _count(1), _s(s) {}

  Attribute(float f) : _type(Type_Float), _count(1) {
    _u.f[0] = f;
    _u.f[1] = 0.f;
    _u.f[2] = 0.f;
    _u.f[3] = 0.f;
  }

  Attribute(float f0, float f1) : _type(Type_Float), _count(2) {
    _u.f[0] = f0;
    _u.f[1] = f1;
    _u.f[2] = 0.f;
    _u.f[3] = 0.f;
  }

  Attribute(float f0, float f1, float f2) : _type(Type_Float), _count(3) {
    _u.f[0] = f0;
    _u.f[0] = f1;
    _u.f[2] = f2;
    _u.f[3] = 0.f;
  }

  Attribute(float f0, float f1, float f2, float f3)
      : _type(Type_Float), _count(4) {
    _u.f[0] = f0;
    _u.f[0] = f1;
    _u.f[2] = f2;
    _u.f[3] = f3;
  }

  Attribute(const float* f, size_t num) : _type(Type_Float), _count(num) {
    size_t idx;
    for (idx = 0; idx < num; idx++) {
      _u.f[idx] = f[idx];
    }
    for (; idx < 4; idx++) {
      _u.f[idx] = 0.f;
    }
  }

  Attribute(int32_t i) : _type(Type_Int), _count(1) {
    _u.i[0] = i;
    _u.i[1] = 0;
    _u.i[2] = 0;
    _u.i[3] = 0;
  }

  Attribute(int32_t i0, int32_t i1) : _type(Type_Int), _count(2) {
    _u.i[0] = i0;
    _u.i[1] = i1;
    _u.i[2] = 0;
    _u.i[3] = 0;
  }

  Attribute(int32_t i0, int32_t i1, int32_t i2) : _type(Type_Int), _count(3) {
    _u.i[0] = i0;
    _u.i[0] = i1;
    _u.i[2] = i2;
    _u.i[3] = 0;
  }

  Attribute(int32_t i0, int32_t i1, int32_t i2, int32_t i3)
      : _type(Type_Int), _count(4) {
    _u.i[0] = i0;
    _u.i[0] = i1;
    _u.i[2] = i2;
    _u.i[3] = i3;
  }

  Attribute(const int32_t* i, size_t num) : _type(Type_Int), _count(num) {
    size_t idx;
    for (idx = 0; idx < num; idx++) {
      _u.i[idx] = i[idx];
    }
    for (; idx < 4; idx++) {
      _u.i[idx] = 0;
    }
  }

  Attribute(uint32_t i) : _type(Type_Int), _count(1) {
    _u.u[0] = i;
    _u.u[1] = 0;
    _u.u[2] = 0;
    _u.u[3] = 0;
  }

  Attribute(uint32_t i0, uint32_t i1) : _type(Type_Int), _count(2) {
    _u.u[0] = i0;
    _u.u[1] = i1;
    _u.u[2] = 0;
    _u.u[3] = 0;
  }

  Attribute(uint32_t i0, uint32_t i1, uint32_t i2)
      : _type(Type_Int), _count(3) {
    _u.u[0] = i0;
    _u.u[0] = i1;
    _u.u[2] = i2;
    _u.u[3] = 0;
  }

  Attribute(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3)
      : _type(Type_Int), _count(4) {
    _u.u[0] = i0;
    _u.u[0] = i1;
    _u.u[2] = i2;
    _u.u[3] = i3;
  }

  Attribute(const uint32_t* i, size_t num) : _type(Type_Int), _count(num) {
    size_t idx;
    for (idx = 0; idx < num; idx++) {
      _u.u[idx] = i[idx];
    }
    for (; idx < 4; idx++) {
      _u.u[idx] = 0;
    }
  }

  Attribute(bool i) : _type(Type_Bool), _count(1) {
    _u.b[0] = i;
    _u.b[1] = false;
    _u.b[2] = false;
    _u.b[3] = false;
  }

  Attribute(bool i0, bool i1) : _type(Type_Bool), _count(2) {
    _u.b[0] = i0;
    _u.b[1] = i1;
    _u.b[2] = false;
    _u.b[3] = false;
  }

  Attribute(bool i0, bool i1, bool i2) : _type(Type_Bool), _count(3) {
    _u.b[0] = i0;
    _u.b[0] = i1;
    _u.b[2] = i2;
    _u.b[3] = false;
  }

  Attribute(bool i0, bool i1, bool i2, bool i3) : _type(Type_Bool), _count(4) {
    _u.b[0] = i0;
    _u.b[0] = i1;
    _u.b[2] = i2;
    _u.b[3] = i3;
  }

  Attribute(const bool* i, size_t num) : _type(Type_Bool), _count(num) {
    size_t idx;
    for (idx = 0; idx < num; idx++) {
      _u.b[idx] = i[idx];
    }
    for (; idx < 4; idx++) {
      _u.b[idx] = false;
    }
  }

  Attribute(Buffer* b) : _type(Type_Buffer), _count(1) { _u.buffer = b; }

  Attribute(Buffer& b) : _type(Type_Buffer), _count(1) { _u.buffer = &b; }

  Attribute(Image* i) : _type(Type_Image), _count(1) { _u.image = i; }

  Attribute(Image& i) : _type(Type_Image), _count(1) { _u.image = &i; }

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

  const int32_t asInt() const { return _u.i[0]; }

  const int32_t* intArray() const { return _u.i; }

  const uint32_t asUInt() const { return _u.u[0]; }

  const uint32_t* uintArray() const { return _u.u; }

  const bool asBool() const { return _u.b[0]; }

  const bool* boolArray() const { return _u.b; }

  Buffer* asBuffer() const { return _u.buffer; }

  Image* asImage() const { return _u.image; }
};

}  // namespace ghost

#endif
