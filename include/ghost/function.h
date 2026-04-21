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
///
/// Sizes are stored as @c size_t so very large 1D dispatches (>2^32 elements)
/// are expressible. Backends that take 32-bit dispatch dimensions natively
/// (CUDA, Vulkan, DirectX) narrow at the dispatch boundary and throw
/// @c std::overflow_error if a value would be truncated.
class LaunchArgs {
 private:
  size_t _dims;
  size_t _global_size[3];
  size_t _local_size[3];
  uint32_t _requiredSubgroupSize;
  bool _local_defined;
  bool _cooperative;

 public:
  /// @brief Get the number of dimensions (1, 2, or 3).
  size_t dims() const { return _dims; }

  /// @brief Get the global work size array (3 elements, unused dims are 1).
  const size_t* global_size() const { return _global_size; }

  /// @brief Get the local work size array (3 elements, unused dims are 1).
  const size_t* local_size() const { return _local_size; }

  /// @brief Check whether local_size was explicitly set.
  bool is_local_defined() const { return _local_defined; }

  /// @brief Check whether cooperative launch is enabled.
  bool is_cooperative() const { return _cooperative; }

  /// @brief Required subgroup size, or 0 if not set.
  uint32_t requiredSubgroupSize() const { return _requiredSubgroupSize; }

  /// @brief Compute the number of work groups in dimension @p i.
  /// @param i Dimension index (0, 1, or 2).
  /// @return Ceiling division of global_size[i] by local_size[i].
  size_t count(size_t i) const {
    return (global_size()[i] + local_size()[i] - 1) / local_size()[i];
  }

  /// @brief Compute the total number of work groups across all dimensions.
  size_t count() const {
    size_t v = 1;
    for (size_t i = 0; i < dims(); i++) {
      v *= count(i);
    }
    return v;
  }

  /// @brief Construct with default values (0 dimensions, all sizes 1).
  LaunchArgs()
      : _dims(0),
        _requiredSubgroupSize(0),
        _local_defined(false),
        _cooperative(false) {
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
  LaunchArgs& global_size(size_t v0) {
    _dims = 1;
    _global_size[0] = v0;
    return *this;
  }

  LaunchArgs& global_size(size_t v0, size_t v1) {
    _dims = 2;
    _global_size[0] = v0;
    _global_size[1] = v1;
    return *this;
  }

  LaunchArgs& global_size(size_t v0, size_t v1, size_t v2) {
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
  LaunchArgs& local_size(size_t v0) {
    _dims = 1;
    _local_size[0] = v0;
    _local_defined = true;
    return *this;
  }

  LaunchArgs& local_size(size_t v0, size_t v1) {
    _dims = 2;
    _local_size[0] = v0;
    _local_size[1] = v1;
    _local_defined = true;
    return *this;
  }

  LaunchArgs& local_size(size_t v0, size_t v1, size_t v2) {
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

  /// @brief Require the kernel to dispatch with a specific subgroup size.
  ///
  /// Backends validate the request against what the compiled pipeline can
  /// actually do:
  /// - Vulkan: requires @c VK_EXT_subgroup_size_control; throws
  ///   @c ghost::unsupported_error if the extension is not available.
  /// - Metal / CUDA / OpenCL / DirectX / CPU: validated against the kernel's
  ///   actual subgroup width; throws @c std::invalid_argument on mismatch.
  ///
  /// Pass 0 to clear the requirement.
  /// @param n Required subgroup size, or 0 for no requirement.
  /// @return @c *this for chaining.
  LaunchArgs& requireSubgroupSize(uint32_t n) {
    _requiredSubgroupSize = n;
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
///
/// A Function holds a strong reference to the Library it was looked up from,
/// so the underlying compiled module (CUmodule, cl_program, VkShaderModule,
/// dlopened .so, etc.) stays alive for as long as any Function from it is
/// in use, even if the user drops the Library wrapper.
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

  /// @brief Bind launch configuration and encoder, returning a callable.
  ///
  /// The returned BoundFunction is called with kernel arguments to dispatch:
  /// @code
  /// fn(LaunchArgs().global_size(1024), stream)(buffer, 42.0f);
  /// fn(LaunchArgs().global_size(1024), cmdBuf)(buffer, 42.0f);
  /// @endcode
  class BoundFunction {
   public:
    BoundFunction(std::shared_ptr<implementation::Function> impl,
                  const LaunchArgs& launchArgs, const Encoder& encoder);

    template <typename... ARGS>
    void operator()(ARGS&&... args) {
      std::vector<Attribute> attrArgs;
      implementation::Function::addArgs(attrArgs, std::forward<ARGS>(args)...);
      dispatch(attrArgs);
    }

   private:
    void dispatch(const std::vector<Attribute>& args);

    std::shared_ptr<implementation::Function> _impl;
    LaunchArgs _launchArgs;
    const Encoder& _encoder;
  };

  /// @brief Bind this kernel to a launch configuration and encoder.
  /// @param launchArgs Global and local work size configuration.
  /// @param s The encoder (Stream or CommandBuffer) to target.
  /// @return A BoundFunction that dispatches when called with kernel arguments.
  BoundFunction operator()(const LaunchArgs& launchArgs, const Encoder& s) {
    return BoundFunction(_impl, launchArgs, s);
  }

  /// @brief Dispatch the kernel with a pre-built argument vector.
  ///
  /// Use this overload when kernel arguments are assembled dynamically
  /// (e.g., from descriptor set bindings or push constants).
  /// @param s The encoder to dispatch on.
  /// @param launchArgs Global and local work size configuration.
  /// @param args Kernel arguments as a vector of Attribute.
  void execute(const Encoder& s, const LaunchArgs& launchArgs,
               const std::vector<Attribute>& args);

  /// @brief Query a function attribute.
  /// @param what The attribute to query (e.g., kFunctionMaxThreads).
  /// @return The attribute value.
  Attribute getAttribute(FunctionAttributeId what) const;

  /// @brief The subgroup (warp / SIMD-group / wavefront) width this kernel
  /// will use when dispatched.
  ///
  /// This may differ from the device-wide subgroup width: Metal and Vulkan
  /// can lock a specific size at pipeline creation, and that locked value is
  /// what the kernel actually sees. Returns 1 for backends without subgroup
  /// execution (CPU).
  uint32_t preferredSubgroupSize() const;

 private:
  friend class Library;
  std::shared_ptr<implementation::Function> _impl;
  // Strong reference to the Library this function was looked up from.
  // Without this, dropping the Library wrapper would unload the underlying
  // compiled module while functions from it are still in use.
  std::shared_ptr<implementation::Library> _parent;
};

/// @brief A compiled GPU program containing one or more kernel functions.
///
/// Libraries are created by Device::loadLibraryFromText(),
/// loadLibraryFromData(), or loadLibraryFromFile(). Use lookupFunction() to
/// obtain individual kernels, lookupSpecializedFunction() to create function
/// specializations with compile-time constant values (Metal, Vulkan), or
/// setGlobals() to set named global constants (CUDA, OpenCL).
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
    Function fn = _impl->lookupFunction(name, std::forward<ARGS>(args)...);
    fn._parent = _impl;
    return fn;
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
    Function fn = _impl->specializeFunction(name, args);
    fn._parent = _impl;
    return fn;
  }

  /// @brief Look up a specialized function with a vector of constant values.
  ///
  /// Non-template overload for dynamic argument counts.
  /// @param name The kernel function name.
  /// @param args Specialization constant values (positional).
  /// @return The specialized Function object.
  Function lookupSpecializedFunction(const std::string& name,
                                     const std::vector<Attribute>& args) {
    Function fn = _impl->specializeFunction(name, args);
    fn._parent = _impl;
    return fn;
  }

  /// @brief Look up a specialized function with named constant values.
  ///
  /// Constants are identified by name rather than position. On Metal, this
  /// uses MTLFunctionConstantValues::setConstantValue:type:withName: so the
  /// caller does not need to know the positional index of each constant.
  ///
  /// @param name The kernel function name.
  /// @param constants Named constant values.
  /// @return The specialized Function object.
  /// @throws ghost::unsupported_error if the backend does not support named
  ///   specialization constants.
  Function lookupSpecializedFunction(
      const std::string& name,
      const std::vector<std::pair<std::string, Attribute>>& constants) {
    Function fn = _impl->specializeFunctionNamed(name, constants);
    fn._parent = _impl;
    return fn;
  }

  /// @brief Set named global constants on this library.
  ///
  /// The semantics are backend-specific:
  /// - CUDA: writes to __constant__ device globals via cuModuleGetGlobal +
  ///   cuMemcpyHtoD.
  /// - OpenCL: recompiles from source with -D defines (only if loaded from
  ///   source text; throws unsupported_error for binary/SPIR-V).
  ///
  /// Previously looked-up functions may be invalidated by this call.
  ///
  /// @param globals Name/value pairs where names match kernel global variable
  ///   names or preprocessor define names.
  /// @throws ghost::unsupported_error if not supported by the backend.
  void setGlobals(
      const std::vector<std::pair<std::string, Attribute>>& globals);

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