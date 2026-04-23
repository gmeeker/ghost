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

#ifndef GHOST_DIRECTX_REFLECT_H
#define GHOST_DIRECTX_REFLECT_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ghost {
namespace dx {

/// @brief HLSL register class for a reflected resource.
///
/// Mirrors the four HLSL register letters: u (UAV), t (SRV), b (CBV),
/// s (sampler). Used both as a tag on individual resources and as the
/// canonical sort order for matching positional kernel arguments — the
/// order is u < t < b < s, then by register slot, mirroring the typical
/// "outputs, inputs, params" declaration convention in compute kernels.
enum class RegisterClass : uint8_t {
  UAV = 0,
  SRV = 1,
  CBV = 2,
  Sampler = 3,
};

/// @brief Sub-classification of a reflected DXIL resource.
///
/// Distinguishes structured/raw/typed resources because D3D12 root
/// parameters use different binding APIs for each. In phase 1 we only
/// handle the buffer-typed variants; sampled images need descriptor
/// tables and are deferred.
enum class ResourceShape : uint8_t {
  CBV,                    // root constants or root CBV
  StructuredOrRawBuffer,  // root SRV / root UAV (no descriptor table needed)
  TypedBuffer,            // requires descriptor table — phase 2
  Texture,                // requires descriptor table — phase 2
  Sampler,                // requires descriptor table — phase 2
};

/// @brief A single resource binding discovered by reflecting a DXIL container.
struct ReflectedResource {
  RegisterClass regClass;
  ResourceShape shape;
  uint32_t space;       ///< HLSL register space (default 0)
  uint32_t lowerBound;  ///< First register slot in the range
  uint32_t upperBound;  ///< Last register slot (inclusive); == lowerBound for
                        ///< single-resource declarations
};

/// @brief Whole-shader reflection result.
struct ReflectedDxilShader {
  /// @brief Resources sorted by HLSL register class then (space, lowerBound).
  ///
  /// Sort key is @c (regClass, space, lowerBound). Within each class,
  /// resources are in ascending register slot order. The class ordering is
  /// u → t → b → s so positional argument matching matches typical HLSL
  /// declaration order in compute kernels.
  std::vector<ReflectedResource> resources;
};

/// @brief Reflect a DXIL container by parsing its PSV0 part.
///
/// Looks at the DXBC container header, scans the part directory for
/// the @c PSV0 part, and walks the resource bind info table inside it.
/// PSV0 has been emitted by every shipping dxc release we care about
/// (1.5+) and is the cheapest reflection surface that has the data we
/// need — full DXIL bitcode parsing or @c IDxcContainerReflection would
/// be heavier and pull in @c dxcompiler.dll at runtime.
///
/// @param data Pointer to the DXIL container bytes (starts with @c 'DXBC').
/// @param sizeBytes Size of the container in bytes.
/// @param[out] out Reflected resources, sorted as documented on
///   @c ReflectedDxilShader::resources.
/// @return @c true on success, @c false if the container is malformed,
///   has no PSV0 part, or the PSV0 part has an unrecognized layout.
bool reflectDxilContainer(const void* data, size_t sizeBytes,
                          ReflectedDxilShader& out);

}  // namespace dx
}  // namespace ghost

#endif
