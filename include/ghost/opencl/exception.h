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

#ifndef GHOST_OPENCL_EXCEPTION_H
#define GHOST_OPENCL_EXCEPTION_H

#include <ghost/device.h>
#include <stdint.h>

#include <exception>

namespace ghost {
namespace opencl {
class runtime_error : public std::runtime_error {
 private:
  int32_t _err;

 public:
  runtime_error(int err);
  int32_t error() const noexcept;
  static const char* errorString(int32_t err);
};

inline void checkError(int32_t err) {
  if (err) throw runtime_error(err);
}
}  // namespace opencl
}  // namespace ghost

#endif