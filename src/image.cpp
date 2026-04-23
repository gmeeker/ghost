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

#include <ghost/image.h>

namespace ghost {
static size_t elementSize(DataType type) {
  switch (type) {
    case DataType_UInt8:
    case DataType_Int8:
      return 1;
    case DataType_UInt16:
    case DataType_Int16:
    case DataType_Float16:
      return 2;
    case DataType_Float:
      return sizeof(float);
    case DataType_Double:
      return sizeof(double);
  }
  return 0;
}

size_t ImageDescription::pixelSize() const {
  return elementSize(type) * channels;
}

size_t ImageDescription::dataSize() const {
  size_t rowBytes = stride.x > 0 ? (size_t)stride.x : size.x * pixelSize();
  size_t height = size.y > 0 ? size.y : 1;
  size_t depth = size.z > 0 ? size.z : 1;
  if (stride.y > 0) {
    return (size_t)stride.y * depth;
  }
  return rowBytes * height * depth;
}
}  // namespace ghost
