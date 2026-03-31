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

#ifndef GHOST_CPU_DEVICE_H
#define GHOST_CPU_DEVICE_H

#include <ghost/cpu/impl_function.h>
#include <ghost/device.h>
#include <ghost/gpu_info.h>

#include <string>
#include <utility>
#include <vector>

namespace ghost {
class DeviceCPU : public Device {
 public:
  DeviceCPU(const SharedContext& share = SharedContext());
  DeviceCPU(const GpuInfo& info);

  static std::vector<GpuInfo> enumerateDevices();

  /// @brief Create a library from inline C++ function pointers.
  ///
  /// This allows registering native C++ functions as CPU kernels without
  /// building a separate shared library. Each function must match the
  /// FunctionCPU::Type signature:
  ///   void (*)(size_t i, size_t n, const std::vector<Attribute>& args)
  /// @param functions Vector of (name, function_pointer) pairs.
  /// @return The Library containing the registered functions.
  Library loadLibraryFromFunctions(
      const std::vector<
          std::pair<std::string, implementation::FunctionCPU::Type>>&
          functions);
};
}  // namespace ghost

#endif