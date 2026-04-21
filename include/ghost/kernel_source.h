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

#ifndef GHOST_KERNEL_SOURCE_H
#define GHOST_KERNEL_SOURCE_H

#include <ghost/device.h>
#include <ghost/function.h>

#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ghost {

/// @brief Stores kernel source or binary data and produces compiled Functions
/// on demand, caching variants by named constants.
///
/// Constructed from either source text or pre-compiled binary data. Each
/// unique set of named constants yields a separately compiled/specialized
/// artifact, cached in memory. Disk caching of compiled binaries is handled
/// transparently by Device::binaryCache().
///
/// **Text mode** (OpenCL C, Metal Shading Language, CUDA via NVRTC):
/// - Metal: compiles the base library once, then creates specialized
///   function variants via named function constants (no recompilation).
/// - CUDA NVRTC: compiles once, extracts the binary, then per variant
///   loads from the cached binary and applies setGlobals() (avoids
///   repeated expensive NVRTC compilation).
/// - OpenCL: compiles a separate library per variant with constants
///   translated to @c -D preprocessor defines.
///
/// Pass @c useDefines=true to the text constructor to force all backends
/// to use @c -D defines instead of specialization or setGlobals.
///
/// **Binary mode** (Metal metallib, CUDA fatbin/PTX, Vulkan SPIR-V):
/// - Metal: loads once, specializes via named function constants.
/// - Vulkan: loads once, specializes via positional specialization
///   constants (constants are passed in declaration order).
/// - CUDA: loads a separate CUmodule per variant via
///   loadLibraryFromData() + setGlobals() (safe for concurrent dispatch
///   since each variant has its own @c __constant__ memory).
///
/// Thread-safe: concurrent getFunction() calls are safe. Cache reads take
/// a shared lock; compilation takes an exclusive lock.
///
/// @code
/// // Text mode (OpenCL)
/// ghost::KernelSource src(opencl_source);
/// auto fn = src.getFunction(device, "average_pool", {
///     {"POOL_H", Attribute(2)},
///     {"POOL_W", Attribute(2)},
/// });
/// fn(args, stream)(outBuf, inBuf);
///
/// // Binary mode (CUDA fatbin)
/// ghost::KernelSource cuda_src(fatbin_data, fatbin_size);
/// auto fn = cuda_src.getFunction(device, "average_pool", {
///     {"POOL_H", Attribute(2)},
///     {"POOL_W", Attribute(2)},
/// });
/// @endcode
class KernelSource {
 public:
  /// @brief Construct from kernel source text.
  ///
  /// By default, backends that support specialization (Metal, Vulkan) will
  /// compile once and create specialized variants. Pass @p useDefines = true
  /// to force all backends to compile per variant with @c -D preprocessor
  /// defines instead. This is useful when the kernel uses @c #ifdef guards
  /// rather than function constants / specialization constants.
  ///
  /// @param text Kernel source code (OpenCL C, Metal Shading Language, etc.).
  /// @param baseOptions Base compiler options applied to every compilation.
  /// @param useDefines If true, always use @c -D defines instead of
  ///   specialization, even on backends that support function constants.
  KernelSource(const std::string& text, const CompilerOptions& baseOptions = {},
               bool useDefines = false);

  /// @brief Construct from pre-compiled binary data.
  /// @param data Pointer to binary data (metallib, fatbin, SPIR-V, etc.).
  /// @param len Length of binary data in bytes.
  /// @param baseOptions Base options (default empty).
  KernelSource(const void* data, size_t len,
               const CompilerOptions& baseOptions = {});

  /// @brief Get a compiled Function for a set of named, typed constants.
  ///
  /// On first call for a given constants combination, compiles (or
  /// specializes) and caches the result. Subsequent calls with the same
  /// constants return the cached Function.
  ///
  /// @param device The device to compile for.
  /// @param functionName The kernel function name.
  /// @param constants Named constant values. On Metal, resolved by name
  ///   via MTLFunctionConstantValues. On OpenCL/CUDA NVRTC (text mode),
  ///   translated to @c -D preprocessor defines. On CUDA (binary mode),
  ///   passed to Library::setGlobals().
  /// @return The compiled Function.
  Function getFunction(
      Device& device, const std::string& functionName,
      const std::vector<std::pair<std::string, Attribute>>& constants = {});

  /// @brief Get a compiled Function with positional specialization constants.
  ///
  /// Constants are identified by position rather than name. This is useful
  /// for Metal metallib and Vulkan SPIR-V paths where function /
  /// specialization constants are declared by index.
  ///
  /// On Metal/Vulkan, passed directly to
  /// Library::lookupSpecializedFunction(name, positional). On other
  /// backends, not supported (throws ghost::unsupported_error) since
  /// @c -D defines and setGlobals() require names.
  ///
  /// @param device The device to compile for.
  /// @param functionName The kernel function name.
  /// @param constants Positional constant values.
  /// @return The compiled Function.
  Function getSpecializedFunction(Device& device,
                                  const std::string& functionName,
                                  const std::vector<Attribute>& constants);

 private:
  enum class Mode { Text, Binary };

  /// Build a deterministic cache key from function name and named constants.
  static std::string makeKey(
      const std::string& functionName,
      const std::vector<std::pair<std::string, Attribute>>& constants);

  /// Build a deterministic cache key from function name and positional
  /// constants.
  static std::string makeKey(const std::string& functionName,
                             const std::vector<Attribute>& constants);

  /// Convert named Attribute constants to CompilerOptions defines.
  static void constantsToDefines(
      const std::vector<std::pair<std::string, Attribute>>& constants,
      CompilerOptions& opts);

  /// Strip names from constants, keeping only values in order.
  static std::vector<Attribute> constantsToPositional(
      const std::vector<std::pair<std::string, Attribute>>& constants);

  Function getFunctionFromText(
      Device& device, const std::string& functionName,
      const std::vector<std::pair<std::string, Attribute>>& constants);

  Function getFunctionFromBinary(
      Device& device, const std::string& functionName,
      const std::vector<std::pair<std::string, Attribute>>& constants);

  Mode _mode;
  bool _forceDefines;
  std::string _text;
  std::vector<uint8_t> _binaryData;
  CompilerOptions _baseOptions;

  mutable std::shared_mutex _mutex;
  // Base library for backends that support specialization (Metal, Vulkan).
  Library _baseLibrary;
  bool _hasBaseLibrary = false;
  // Resolved on first call per mode.
  bool _capabilityChecked = false;
  bool _useSpecialization = false;
  // Cached Functions keyed by (functionName + serialized constants).
  std::unordered_map<std::string, Function> _cache;
};

}  // namespace ghost

#endif
