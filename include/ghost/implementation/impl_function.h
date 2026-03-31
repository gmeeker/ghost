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

#ifndef GHOST_IMPL_FUNCTION_H
#define GHOST_IMPL_FUNCTION_H

#include <ghost/attribute.h>
#include <ghost/exception.h>
#include <stdlib.h>

#include <memory>
#include <string>
#include <vector>

namespace ghost {

/// @brief Identifiers for queryable function (kernel) attributes.
///
/// Pass these to Function::getAttribute() or
/// implementation::Function::getAttribute() to retrieve kernel properties after
/// compilation.
enum FunctionAttributeId {
  /// @brief Static local memory used by the kernel in bytes.
  kFunctionLocalMemory,
  /// @brief Maximum local memory available to the kernel in bytes.
  kFunctionMaxLocalMemory,
  /// @brief SIMD/warp width for this kernel.
  kFunctionThreadWidth,
  /// @brief Maximum threads per work-group for this kernel.
  kFunctionMaxThreads,
  /// @brief Required work-group size (3-element int array), or zeros if none.
  kFunctionRequiredWorkSize,
  /// @brief Preferred work-group size multiple.
  kFunctionPreferredWorkMultiple,
  /// @brief Number of registers used per thread (CUDA).
  kFunctionNumRegisters,
  /// @brief Private memory per work-item in bytes.
  kFunctionPrivateMemory,
};

class Function;
class Library;
class Stream;
class LaunchArgs;

namespace implementation {

class Buffer;

/// @brief Abstract backend interface for a compiled GPU kernel function.
///
/// Backend implementations derive from this class to provide kernel execution
/// and attribute queries. Not copyable. The variadic operator() converts
/// arguments to a vector of Attribute and delegates to execute().
class Function {
 public:
  class Arg {};

  Function() {}

  Function(const Function& rhs) = delete;

  virtual ~Function() {}

  Function& operator=(const Function& rhs) = delete;

  /// @brief Execute the kernel on a stream with the given arguments.
  /// @param s The stream to enqueue the kernel on.
  /// @param launchArgs Global and local work size configuration.
  /// @param args Kernel arguments as a vector of Attribute.
  virtual void execute(const ghost::Stream& s, const LaunchArgs& launchArgs,
                       const std::vector<Attribute>& args) = 0;

  /// @brief Execute the kernel with workgroup counts read from a GPU buffer.
  ///
  /// The default implementation syncs the stream, reads 3x uint32_t from
  /// the indirect buffer, and calls execute(). Backends with native indirect
  /// dispatch (Metal) override this for zero-CPU-roundtrip dispatch.
  ///
  /// @param s The stream to enqueue on.
  /// @param indirectBuffer Buffer containing 3x uint32_t workgroup counts.
  /// @param indirectOffset Byte offset into the indirect buffer.
  /// @param args Kernel arguments.
  virtual void executeIndirect(const ghost::Stream& s,
                               const std::shared_ptr<Buffer>& indirectBuffer,
                               size_t indirectOffset,
                               const std::vector<Attribute>& args);

  virtual Attribute getAttribute(FunctionAttributeId what) const = 0;

  /// @brief Helper to build an Attribute vector from variadic arguments.
  /// @{
  static void addArgs(std::vector<Attribute>&) {}

  template <typename ARG>
  static void addArgs(std::vector<Attribute>& args, ARG&& head) {
    args.push_back(std::forward<ARG>(head));
  }

  template <typename ARG, typename... ARGS>
  static void addArgs(std::vector<Attribute>& args, ARG&& head,
                      ARGS&&... tail) {
    args.push_back(std::forward<ARG>(head));
    addArgs(args, std::forward<ARGS>(tail)...);
  }

  /// @}

  /// @brief Dispatch the kernel, converting variadic arguments to Attribute
  /// vector.
  template <typename... ARGS>
  void operator()(const ghost::Stream& s, const LaunchArgs& launchArgs,
                  ARGS&&... tail) {
    std::vector<Attribute> args;
    addArgs(args, std::forward<ARGS>(tail)...);
    execute(s, launchArgs, args);
  }
};

/// @brief Abstract backend interface for a compiled GPU program (library).
///
/// Backend implementations derive from this class to provide function lookup
/// and optional specialization. Not copyable.
class Library {
 public:
  Library() {}

  explicit Library(bool retainBinary) : _retainBinary(retainBinary) {}

  Library(const Library& rhs) = delete;

  virtual ~Library() {}

  Library& operator=(const Library& rhs) = delete;

  /// @brief Whether this library retains compiled binary data for getBinary().
  bool retainBinary() const { return _retainBinary; }

 private:
  bool _retainBinary = false;

 public:
  virtual ghost::Function lookupFunction(const std::string& name) const = 0;

  /// @brief Create a specialized function variant with compile-time constant
  /// values.
  ///
  /// The default implementation throws ghost::unsupported_error. Backends
  /// that support function constants (e.g., Metal) override this method.
  /// @param name The kernel function name.
  /// @param args Specialization constant values.
  /// @return The specialized Function.
  /// @throws ghost::unsupported_error if not supported by the backend.
  virtual ghost::Function specializeFunction(
      const std::string& name, const std::vector<Attribute>& args) const;

  /// @brief Retrieve the compiled binary data from this library.
  ///
  /// Returns the backend-specific compiled binary (e.g., cubin for CUDA,
  /// metallib for Metal, device binary for OpenCL, SPIR-V for Vulkan,
  /// DXIL for DirectX). Returns empty vector if unsupported.
  /// @return A vector of bytes containing the compiled binary.
  virtual std::vector<uint8_t> getBinary() const;
};
}  // namespace implementation
}  // namespace ghost

#endif