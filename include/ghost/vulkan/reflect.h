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

#ifndef GHOST_VULKAN_REFLECT_H
#define GHOST_VULKAN_REFLECT_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ghost {
namespace vk {

/// @brief Backend-neutral classification of a SPIR-V resource binding.
enum class ResourceKind {
  StorageBuffer,  ///< SSBO (HLSL u#/RWStructuredBuffer or t#/StructuredBuffer)
  UniformBuffer,  ///< UBO (HLSL b#/cbuffer when not push-constant-decorated)
  StorageImage,   ///< Storage image (HLSL u# RWTexture, GLSL imageND)
  SampledImage,   ///< Sampled image (HLSL t# Texture, GLSL textureND)
  Sampler,        ///< Standalone sampler (HLSL s#/SamplerState)
  CombinedImageSampler,  ///< GLSL combined sampler2D etc.
};

/// @brief A descriptor binding discovered by reflection.
///
/// One per @c OpVariable in the relevant storage classes. The @c set and
/// @c binding fields are read directly from the SPIR-V decorations; the
/// @c kind is inferred from the storage class plus the type pointed at.
struct ReflectedBinding {
  ResourceKind kind;
  uint32_t set;
  uint32_t binding;
  /// @brief @c true if HLSL SRV (StructuredBuffer) — decorated NonWritable.
  /// Currently informational only; Vulkan binds either way as STORAGE_BUFFER.
  bool readOnly;
};

/// @brief A push-constant block discovered by reflection.
///
/// SPIR-V allows at most one push-constant block per stage. @c size is
/// derived from the largest member offset plus its size and rounded to the
/// next 4-byte boundary, matching what @c vkCmdPushConstants expects.
struct ReflectedPushConstants {
  bool present = false;
  uint32_t size = 0;
};

/// @brief Output of @c reflectSpirv — what the shader needs from the host.
struct ReflectedShader {
  std::vector<ReflectedBinding> bindings;  ///< sorted by (set, binding)
  ReflectedPushConstants pushConstants;
};

/// @brief Walk a SPIR-V binary and produce a backend-neutral binding layout.
///
/// Handles both modern @c StorageBuffer storage class and legacy
/// @c Uniform-with-BufferBlock-decoration encodings (the latter is what
/// glslangValidator produces without @c --target-env vulkan1.3, and what
/// dxc produces when targeting vulkan1.0). Skips any opcode it doesn't
/// recognize, so this is safe against future SPIR-V revisions.
///
/// @param data Pointer to SPIR-V words. Must be 4-byte aligned.
/// @param sizeBytes Size of the SPIR-V binary in bytes (must be a multiple
///   of 4).
/// @param[out] out The reflected layout. Bindings are sorted by
///   @c (set, binding).
/// @return @c true on success, @c false if the input doesn't look like
///   SPIR-V (wrong magic) or is malformed.
bool reflectSpirv(const void* data, size_t sizeBytes, ReflectedShader& out);

}  // namespace vk
}  // namespace ghost

#endif
