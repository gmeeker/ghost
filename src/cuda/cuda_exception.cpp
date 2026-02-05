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

#if WITH_CUDA

#include <cuda.h>
#include <ghost/cuda/exception.h>

namespace ghost {
namespace cu {
runtime_error::runtime_error(int err)
    : std::runtime_error(errorString(err)), _err(err) {}

int32_t runtime_error::error() const noexcept { return _err; }

const char* runtime_error::errorString(int32_t err) {
  const char* str;
  if (err == CUDA_SUCCESS) return nullptr;
  if (cuGetErrorString((CUresult)err, &str) == CUDA_SUCCESS) return str;
  return "Unknown error";
}
}  // namespace cu
}  // namespace ghost

#endif
