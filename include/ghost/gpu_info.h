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

#ifndef GHOST_GPU_INFO_H
#define GHOST_GPU_INFO_H

#include <cstdint>
#include <string>

namespace ghost {

/// @brief Lightweight descriptor for an available GPU device.
///
/// Populated by backend-specific `enumerateDevices()` without creating a
/// context. Pass to a backend Device constructor to create a device targeting
/// a specific GPU.
struct GpuInfo {
  /// @brief Human-readable device name.
  std::string name;

  /// @brief Device vendor string.
  std::string vendor;

  /// @brief Backend implementation name (e.g., "Metal", "OpenCL", "CUDA",
  /// "CPU").
  std::string implementation;

  /// @brief Device global memory in bytes.
  uint64_t memory = 0;

  /// @brief Whether the device shares memory with the host.
  bool unifiedMemory = false;

  /// @brief Backend-local device index (ordinal).
  ///
  /// For CUDA this is the device ordinal passed to cuDeviceGet. For OpenCL it
  /// encodes the platform and device index. For Metal it is the index into the
  /// array returned by MTLCopyAllDevices. For CPU it is always 0.
  int index = 0;
};

}  // namespace ghost

#endif
