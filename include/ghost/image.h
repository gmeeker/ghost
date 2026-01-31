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
template <typename T>
class Value2 {
 public:
  T x, y;

  Value2(T x_, T y_) : x(x_), y(y_) {}
};

template <typename T>
class Value3 {
 public:
  T x, y, z;

  Value3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
};

typedef Value2<int> Point2;
typedef Value2<size_t> Size2;
typedef Value3<int> Point3;
typedef Value3<size_t> Size3;
typedef Value2<int32_t> Stride2;

enum DataType {
  DataType_UInt8,
  DataType_Int8,
  DataType_UInt16,
  DataType_Int16,
  DataType_Float16,
  DataType_Float,
  DataType_Double
};

typedef uint32_t PixelOrder;

enum {
  PixelOrder_RGBA = ((0 << 6) + (1 << 4) + (2 << 2) + (3 << 0)),
  PixelOrder_ARGB = ((1 << 6) + (2 << 4) + (3 << 2) + (0 << 0)),
  PixelOrder_ABGR = ((3 << 6) + (2 << 4) + (1 << 2) + (0 << 0)),
  PixelOrder_BGRA = ((2 << 6) + (1 << 4) + (0 << 2) + (3 << 0))
};

enum Access { Access_ReadOnly, Access_WriteOnly, Access_ReadWrite };

class ImageDescription {
 public:
  Size3 size;
  size_t channels;
  PixelOrder order;
  DataType type;
  Stride2 stride;
  Access access;

  ImageDescription(Size3 size_, PixelOrder order_, DataType type_,
                   Stride2 stride_, Access access_ = Access_ReadWrite)
      : size(size_),
        order(order_),
        type(type_),
        stride(stride_),
        access(access_) {}

  size_t dataSize();
  size_t pixelSize();
};
}  // namespace ghost

#endif
