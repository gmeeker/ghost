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

#ifndef GHOST_FUNCTION_H
#define GHOST_FUNCTION_H

#include <ghost/implementation/impl_function.h>

#include <memory>
#include <string>

namespace ghost {

/// @brief Kernel dispatch configuration specifying global and local work sizes.
///
/// Supports 1D, 2D, and 3D dispatches. Use the fluent API to set sizes:
/// @code
/// LaunchArgs args;
/// args.global_size(1024).local_size(256);           // 1D
/// args.global_size(512, 512).local_size(16, 16);    // 2D
/// @endcode
///
/// If local_size is not explicitly set, the backend chooses a default.
class LaunchArgs {
 private:
  uint32_t _dims;
  uint32_t _global_size[3];
  uint32_t _local_size[3];
  bool _local_defined;
  bool _cooperative;

 public:
  /// @brief Get the number of dimensions (1, 2, or 3).
  uint32_t dims() const { return _dims; }

  /// @brief Get the global work size array (3 elements, unused dims are 1).
  const uint32_t* global_size() const { return _global_size; }

  /// @brief Get the local work size array (3 elements, unused dims are 1).
  const uint32_t* local_size() const { return _local_size; }

  /// @brief Check whether local_size was explicitly set.
  bool is_local_defined() const { return _local_defined; }

  /// @brief Check whether cooperative launch is enabled.
  bool is_cooperative() const { return _cooperative; }

  /// @brief Compute the number of work groups in dimension @p i.
  /// @param i Dimension index (0, 1, or 2).
  /// @return Ceiling division of global_size[i] by local_size[i].
  size_t count(uint32_t i) const {
    return size_t((global_size()[i] + local_size()[i] - 1) / local_size()[i]);
  }

  /// @brief Compute the total number of work groups across all dimensions.
  size_t count() const {
    size_t v = 1;
    for (uint32_t i = 0; i < dims(); i++) {
      v *= count(i);
    }
    return v;
  }

  /// @brief Construct with default values (0 dimensions, all sizes 1).
  LaunchArgs() : _dims(0), _local_defined(false), _cooperative(false) {
    _global_size[0] = 1;
    _global_size[1] = 1;
    _global_size[2] = 1;
    _local_size[0] = 1;
    _local_size[1] = 1;
    _local_size[2] = 1;
  }

  /// @name Global work size setters
  /// Set the global work size for 1D, 2D, or 3D dispatch. Returns @c *this for
  /// chaining.
  /// @{
  LaunchArgs& global_size(uint32_t v0) {
    _dims = 1;
    _global_size[0] = v0;
    return *this;
  }

  LaunchArgs& global_size(uint32_t v0, uint32_t v1) {
    _dims = 2;
    _global_size[0] = v0;
    _global_size[1] = v1;
    return *this;
  }

  LaunchArgs& global_size(uint32_t v0, uint32_t v1, uint32_t v2) {
    _dims = 3;
    _global_size[0] = v0;
    _global_size[1] = v1;
    _global_size[2] = v2;
    return *this;
  }

  /// @}

  /// @name Local work size setters
  /// Set the local (work-group) size for 1D, 2D, or 3D dispatch.
  /// Returns @c *this for chaining.
  /// @{
  LaunchArgs& local_size(uint32_t v0) {
    _dims = 1;
    _local_size[0] = v0;
    _local_defined = true;
    return *this;
  }

  LaunchArgs& local_size(uint32_t v0, uint32_t v1) {
    _dims = 2;
    _local_size[0] = v0;
    _local_size[1] = v1;
    _local_defined = true;
    return *this;
  }

  LaunchArgs& local_size(uint32_t v0, uint32_t v1, uint32_t v2) {
    _dims = 3;
    _local_size[0] = v0;
    _local_size[1] = v1;
    _local_size[2] = v2;
    _local_defined = true;
    return *this;
  }

  /// @}

  /// @brief Enable or disable cooperative launch mode.
  ///
  /// When enabled, the CUDA backend uses @c cuLaunchCooperativeKernel instead
  /// of @c cuLaunchKernel, enabling cooperative groups functionality. Other
  /// backends ignore this flag.
  /// @param enable @c true to enable cooperative launch (default @c true).
  /// @return @c *this for chaining.
  LaunchArgs& cooperative(bool enable = true) {
    _cooperative = enable;
    return *this;
  }
};

/// @brief A compiled GPU kernel function that can be dispatched on a stream.
///
/// Functions are obtained from a Library via lookupFunction(). They are
/// invoked using operator() with a Stream, LaunchArgs, and kernel arguments:
/// @code
/// ghost::Function fn = library.lookupFunction("myKernel");
/// fn(stream, LaunchArgs().global_size(1024), buffer, 42.0f);
/// @endcode
class Function {
 public:
  /// @brief Construct from a backend implementation.
  /// @param impl Shared pointer to the backend-specific function
  /// implementation.
  Function(std::shared_ptr<implementation::Function> impl);

  /// @brief Get the backend implementation (const).
  std::shared_ptr<implementation::Function> impl() const { return _impl; }

  /// @brief Get the backend implementation (mutable).
  std::shared_ptr<implementation::Function>& impl() { return _impl; }

  /// @brief Dispatch the kernel on a stream with the given launch configuration
  /// and arguments.
  /// @tparam ARGS Variadic argument types, each convertible to Attribute.
  /// @param s The stream to enqueue the kernel on.
  /// @param launchArgs Global and local work size configuration.
  /// @param args Kernel arguments (buffers, images, scalars, local memory).
  template <typename... ARGS>
  void operator()(const Stream& s, const LaunchArgs& launchArgs,
                  ARGS&&... args) {
    (*_impl)(s, launchArgs, std::forward<ARGS>(args)...);
  }

  /// @brief Dispatch the kernel with a pre-built argument vector.
  ///
  /// Use this overload when kernel arguments are assembled dynamically
  /// (e.g., from descriptor set bindings or push constants).
  /// @param s The stream to enqueue the kernel on.
  /// @param launchArgs Global and local work size configuration.
  /// @param args Kernel arguments as a vector of Attribute.
  void execute(const Stream& s, const LaunchArgs& launchArgs,
               const std::vector<Attribute>& args);

  /// @brief Query a function attribute.
  /// @param what The attribute to query (e.g., kFunctionMaxThreads).
  /// @return The attribute value.
  Attribute getAttribute(FunctionAttributeId what) const;

 private:
  std::shared_ptr<implementation::Function> _impl;
};

/// @brief A compiled GPU program containing one or more kernel functions.
///
/// Libraries are created by Device::loadLibraryFromText(),
/// loadLibraryFromData(), or loadLibraryFromFile(). Use lookupFunction() to
/// obtain individual kernels, or lookupSpecializedFunction() to create function
/// specializations with compile-time constant values (supported on Metal).
class Library {
 public:
  /// @brief Construct from a backend implementation.
  /// @param impl Shared pointer to the backend-specific library implementation.
  Library(std::shared_ptr<implementation::Library> impl);

  /// @brief Look up a kernel function by name.
  /// @param name The kernel function name.
  /// @return The Function object.
  /// @throws std::runtime_error if the function is not found.
  Function lookupFunction(const std::string& name) const;

  /// @brief Look up a kernel function by name with additional backend-specific
  /// arguments.
  /// @tparam ARGS Additional argument types forwarded to the backend.
  /// @param name The kernel function name.
  /// @param args Backend-specific arguments.
  /// @return The Function object.
  template <typename... ARGS>
  Function lookupFunction(const std::string& name, ARGS&&... args) {
    return _impl->lookupFunction(name, std::forward<ARGS>(args)...);
  }

  /// @brief Look up a specialized function with compile-time constant values.
  ///
  /// Creates a function variant where certain parameters are baked in as
  /// constants, enabling backend-specific optimizations (e.g., Metal function
  /// constants).
  /// @tparam ARGS Specialization constant types, each convertible to Attribute.
  /// @param name The kernel function name.
  /// @param tail Specialization constant values.
  /// @return The specialized Function object.
  template <typename... ARGS>
  Function lookupSpecializedFunction(const std::string& name, ARGS&&... tail) {
    std::vector<Attribute> args;
    implementation::Function::addArgs(args, std::forward<ARGS>(tail)...);
    return _impl->specializeFunction(name, args);
  }

  /// @brief Look up a specialized function with a vector of constant values.
  ///
  /// Non-template overload for dynamic argument counts.
  /// @param name The kernel function name.
  /// @param args Specialization constant values.
  /// @return The specialized Function object.
  Function lookupSpecializedFunction(const std::string& name,
                                     const std::vector<Attribute>& args) {
    return _impl->specializeFunction(name, args);
  }

  /// @brief Retrieve the compiled binary data from this library.
  ///
  /// Returns the backend-specific compiled binary (e.g., cubin for CUDA,
  /// metallib for Metal, device binary for OpenCL, SPIR-V for Vulkan,
  /// DXIL for DirectX). Returns empty vector if unsupported.
  /// @return A vector of bytes containing the compiled binary.
  std::vector<uint8_t> getBinary() const;

 protected:
  /// @brief Get the backend implementation (const).
  std::shared_ptr<implementation::Library> impl() const { return _impl; }

  /// @brief Get the backend implementation (mutable).
  std::shared_ptr<implementation::Library>& impl() { return _impl; }

 private:
  std::shared_ptr<implementation::Library> _impl;
};
}  // namespace ghost

#endif