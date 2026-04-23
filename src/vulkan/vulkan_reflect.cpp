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

#if WITH_VULKAN

#include <ghost/vulkan/reflect.h>

#include <algorithm>
#include <unordered_map>

namespace ghost {
namespace vk {

namespace {

// Subset of the SPIR-V spec we care about. The numeric values come from
// SPIR-V 1.6 (https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html)
// and are stable across SPIR-V revisions.

constexpr uint32_t kSpirvMagic = 0x07230203;

// Storage classes
constexpr uint32_t kStorageClassUniformConstant = 0;
constexpr uint32_t kStorageClassUniform = 2;
constexpr uint32_t kStorageClassPushConstant = 9;
constexpr uint32_t kStorageClassStorageBuffer = 12;

// Decorations
constexpr uint32_t kDecorationBlock = 2;
constexpr uint32_t kDecorationBufferBlock = 3;  // legacy SSBO marker
constexpr uint32_t kDecorationNonWritable = 24;
constexpr uint32_t kDecorationBinding = 33;
constexpr uint32_t kDecorationDescriptorSet = 34;
constexpr uint32_t kDecorationOffset = 35;

// Opcodes
constexpr uint16_t kOpTypeVoid = 19;
constexpr uint16_t kOpTypeBool = 20;
constexpr uint16_t kOpTypeInt = 21;
constexpr uint16_t kOpTypeFloat = 22;
constexpr uint16_t kOpTypeVector = 23;
constexpr uint16_t kOpTypeMatrix = 24;
constexpr uint16_t kOpTypeImage = 25;
constexpr uint16_t kOpTypeSampler = 26;
constexpr uint16_t kOpTypeSampledImage = 27;
constexpr uint16_t kOpTypeArray = 28;
constexpr uint16_t kOpTypeRuntimeArray = 29;
constexpr uint16_t kOpTypeStruct = 30;
constexpr uint16_t kOpTypePointer = 32;
constexpr uint16_t kOpVariable = 59;
constexpr uint16_t kOpDecorate = 71;
constexpr uint16_t kOpMemberDecorate = 72;
constexpr uint16_t kOpConstant = 43;

// Per-id information collected during the walk.
struct TypeInfo {
  enum Kind {
    Unknown,
    Scalar,  // int / float / bool
    Vector,
    Matrix,
    Struct,
    Pointer,
    Array,         // statically-sized array
    RuntimeArray,  // unbounded
    Image,
    Sampler,
    SampledImage,
  };

  Kind kind = Unknown;
  uint32_t componentSize = 0;  // for scalars: bytes; for composites: 0
  uint32_t elementCount = 0;   // for vector/matrix/array
  uint32_t innerType = 0;      // for vector/matrix/array/pointer/runtimearray
  uint32_t storageClass = 0;   // for pointer
  // for struct
  std::vector<uint32_t> memberTypes;
  bool decoratedBlock = false;
  bool decoratedBufferBlock = false;
  // largest "member offset + member size" seen — used to compute push
  // constant block size when no array stride/sized members are present.
  uint32_t structSize = 0;
};

struct VarInfo {
  uint32_t pointerType = 0;
  uint32_t storageClass = 0;
  bool hasDescriptorSet = false;
  uint32_t descriptorSet = 0;
  bool hasBinding = false;
  uint32_t binding = 0;
};

struct Walker {
  const uint32_t* words;
  size_t wordCount;

  std::unordered_map<uint32_t, TypeInfo> types;
  std::unordered_map<uint32_t, VarInfo> vars;
  std::unordered_map<uint32_t, uint32_t> constants;  // id → uint value

  // Per-id "is decorated NonWritable" — applies to variables (HLSL SRV) or
  // to all members of the struct (also HLSL SRV).
  std::unordered_map<uint32_t, bool> nonWritableVars;
  std::unordered_map<uint32_t, bool> nonWritableStructs;

  TypeInfo& type(uint32_t id) { return types[id]; }

  VarInfo& var(uint32_t id) { return vars[id]; }
};

// Round @p v up to the next multiple of 4.
uint32_t roundUp4(uint32_t v) { return (v + 3u) & ~3u; }

// Compute the byte size of a struct used as a push-constant block.
//
// SPIR-V emitters always tag members of such a struct with @c Offset
// decorations. We approximate the total size as @c (max(offset) + size of
// last member's type), rounded up to 4. For nested types we recurse into
// vectors/matrices/arrays.
uint32_t typeSizeBytes(Walker& w, uint32_t typeId);

uint32_t structSizeBytes(Walker& w, const TypeInfo& s) {
  // s.structSize was populated as we observed OpMemberDecorate Offset
  // entries. Bump by the trailing member size.
  if (s.memberTypes.empty()) return 0;
  uint32_t lastMember = s.memberTypes.back();
  return roundUp4(s.structSize + typeSizeBytes(w, lastMember));
}

uint32_t typeSizeBytes(Walker& w, uint32_t typeId) {
  auto it = w.types.find(typeId);
  if (it == w.types.end()) return 0;
  const TypeInfo& t = it->second;
  switch (t.kind) {
    case TypeInfo::Scalar:
      return t.componentSize;
    case TypeInfo::Vector:
      return t.componentSize * t.elementCount;  // Vector ctor stores both
    case TypeInfo::Matrix:
      // matrix = elementCount columns of vectors of innerType
      return typeSizeBytes(w, t.innerType) * t.elementCount;
    case TypeInfo::Array:
      return typeSizeBytes(w, t.innerType) * t.elementCount;
    case TypeInfo::Struct:
      return structSizeBytes(w, t);
    default:
      return 0;
  }
}

ResourceKind resolveResourceKind(Walker& w, const VarInfo& v,
                                 const TypeInfo* pointee, bool& outReadOnly) {
  outReadOnly = false;
  switch (v.storageClass) {
    case kStorageClassStorageBuffer:
      // Modern SSBO encoding. Always a Block-decorated struct.
      return ResourceKind::StorageBuffer;
    case kStorageClassUniform:
      // Either UBO (struct decorated Block) or legacy SSBO (struct
      // decorated BufferBlock). dxc emits the former by default; older
      // glslangValidator may emit the latter.
      if (pointee && pointee->kind == TypeInfo::Struct &&
          pointee->decoratedBufferBlock) {
        return ResourceKind::StorageBuffer;
      }
      return ResourceKind::UniformBuffer;
    case kStorageClassUniformConstant: {
      // Image / sampler / sampled image — pointee is one of those types
      // (possibly through a runtime array, which we deliberately don't
      // unwrap for descriptor arrays in phase 1).
      if (!pointee) return ResourceKind::SampledImage;  // arbitrary default
      switch (pointee->kind) {
        case TypeInfo::Image:
          return ResourceKind::StorageImage;
        case TypeInfo::Sampler:
          return ResourceKind::Sampler;
        case TypeInfo::SampledImage:
          return ResourceKind::CombinedImageSampler;
        default:
          return ResourceKind::SampledImage;
      }
    }
    default:
      return ResourceKind::StorageBuffer;
  }
}

}  // namespace

bool reflectSpirv(const void* data, size_t sizeBytes, ReflectedShader& out) {
  out.bindings.clear();
  out.pushConstants = {};

  if (!data || (sizeBytes % 4) != 0 || sizeBytes < 5 * sizeof(uint32_t)) {
    return false;
  }
  Walker w;
  w.words = static_cast<const uint32_t*>(data);
  w.wordCount = sizeBytes / sizeof(uint32_t);

  if (w.words[0] != kSpirvMagic) return false;
  // words[1]=version, [2]=generator, [3]=bound, [4]=schema
  size_t i = 5;
  while (i < w.wordCount) {
    uint32_t header = w.words[i];
    uint16_t wordCount = (uint16_t)(header >> 16);
    uint16_t opcode = (uint16_t)(header & 0xFFFFu);
    if (wordCount == 0 || i + wordCount > w.wordCount) return false;
    const uint32_t* operands = w.words + i + 1;
    uint16_t numOperands = (uint16_t)(wordCount - 1);
    switch (opcode) {
      case kOpDecorate: {
        if (numOperands < 2) break;
        uint32_t targetId = operands[0];
        uint32_t decoration = operands[1];
        if (decoration == kDecorationDescriptorSet && numOperands >= 3) {
          auto& v = w.var(targetId);
          v.hasDescriptorSet = true;
          v.descriptorSet = operands[2];
        } else if (decoration == kDecorationBinding && numOperands >= 3) {
          auto& v = w.var(targetId);
          v.hasBinding = true;
          v.binding = operands[2];
        } else if (decoration == kDecorationBlock) {
          w.type(targetId).decoratedBlock = true;
        } else if (decoration == kDecorationBufferBlock) {
          w.type(targetId).decoratedBufferBlock = true;
        } else if (decoration == kDecorationNonWritable) {
          w.nonWritableVars[targetId] = true;
        }
        break;
      }
      case kOpMemberDecorate: {
        if (numOperands < 3) break;
        uint32_t structId = operands[0];
        // operands[1] = member index
        uint32_t decoration = operands[2];
        if (decoration == kDecorationOffset && numOperands >= 4) {
          uint32_t off = operands[3];
          auto& t = w.type(structId);
          if (off > t.structSize) t.structSize = off;
        } else if (decoration == kDecorationNonWritable) {
          // If every member of a struct is decorated NonWritable, treat
          // the whole struct as read-only. Tracking once is sufficient
          // for our use — even one NonWritable member is a strong signal
          // this came from an HLSL StructuredBuffer SRV.
          w.nonWritableStructs[structId] = true;
        }
        break;
      }
      case kOpConstant: {
        // result_type, result_id, literal0[, literal1...]
        if (numOperands < 3) break;
        // We only care about uint literals used by OpTypeArray below.
        uint32_t resultId = operands[1];
        w.constants[resultId] = operands[2];
        break;
      }
      case kOpTypeVoid:
      case kOpTypeBool: {
        if (numOperands < 1) break;
        auto& t = w.type(operands[0]);
        t.kind = TypeInfo::Scalar;
        t.componentSize = (opcode == kOpTypeBool) ? 4 : 0;
        break;
      }
      case kOpTypeInt:
      case kOpTypeFloat: {
        if (numOperands < 2) break;
        auto& t = w.type(operands[0]);
        t.kind = TypeInfo::Scalar;
        t.componentSize = operands[1] / 8u;
        break;
      }
      case kOpTypeVector: {
        if (numOperands < 3) break;
        auto& t = w.type(operands[0]);
        t.kind = TypeInfo::Vector;
        t.innerType = operands[1];
        t.elementCount = operands[2];
        // Cache the component size up front so typeSizeBytes() doesn't have
        // to recurse for the very common float/int vector cases.
        auto inner = w.types.find(operands[1]);
        if (inner != w.types.end())
          t.componentSize = inner->second.componentSize;
        break;
      }
      case kOpTypeMatrix: {
        if (numOperands < 3) break;
        auto& t = w.type(operands[0]);
        t.kind = TypeInfo::Matrix;
        t.innerType = operands[1];
        t.elementCount = operands[2];
        break;
      }
      case kOpTypeArray: {
        if (numOperands < 3) break;
        auto& t = w.type(operands[0]);
        t.kind = TypeInfo::Array;
        t.innerType = operands[1];
        // operands[2] is a constant id, not the count itself.
        auto cit = w.constants.find(operands[2]);
        t.elementCount = (cit != w.constants.end()) ? cit->second : 0;
        break;
      }
      case kOpTypeRuntimeArray: {
        if (numOperands < 2) break;
        auto& t = w.type(operands[0]);
        t.kind = TypeInfo::RuntimeArray;
        t.innerType = operands[1];
        break;
      }
      case kOpTypeStruct: {
        if (numOperands < 1) break;
        auto& t = w.type(operands[0]);
        t.kind = TypeInfo::Struct;
        t.memberTypes.assign(operands + 1, operands + numOperands);
        break;
      }
      case kOpTypePointer: {
        if (numOperands < 3) break;
        auto& t = w.type(operands[0]);
        t.kind = TypeInfo::Pointer;
        t.storageClass = operands[1];
        t.innerType = operands[2];
        break;
      }
      case kOpTypeImage: {
        if (numOperands < 1) break;
        w.type(operands[0]).kind = TypeInfo::Image;
        break;
      }
      case kOpTypeSampler: {
        if (numOperands < 1) break;
        w.type(operands[0]).kind = TypeInfo::Sampler;
        break;
      }
      case kOpTypeSampledImage: {
        if (numOperands < 1) break;
        w.type(operands[0]).kind = TypeInfo::SampledImage;
        break;
      }
      case kOpVariable: {
        if (numOperands < 3) break;
        uint32_t resultType = operands[0];
        uint32_t resultId = operands[1];
        uint32_t storageClass = operands[2];
        auto& v = w.var(resultId);
        v.pointerType = resultType;
        v.storageClass = storageClass;
        break;
      }
      default:
        break;
    }
    i += wordCount;
  }

  // Walk the variables we collected and emit ReflectedBindings for the
  // ones in storage classes we care about. Push-constant variables go to
  // out.pushConstants instead.
  for (auto& kv : w.vars) {
    const VarInfo& v = kv.second;
    if (v.storageClass == kStorageClassPushConstant) {
      // Pointer → struct, struct's members carry the offsets.
      auto pIt = w.types.find(v.pointerType);
      if (pIt == w.types.end() || pIt->second.kind != TypeInfo::Pointer)
        continue;
      auto sIt = w.types.find(pIt->second.innerType);
      if (sIt == w.types.end() || sIt->second.kind != TypeInfo::Struct)
        continue;
      uint32_t bytes = structSizeBytes(w, sIt->second);
      if (bytes > out.pushConstants.size) out.pushConstants.size = bytes;
      out.pushConstants.present = true;
      continue;
    }

    if (!v.hasDescriptorSet || !v.hasBinding) continue;
    if (v.storageClass != kStorageClassUniform &&
        v.storageClass != kStorageClassStorageBuffer &&
        v.storageClass != kStorageClassUniformConstant) {
      continue;
    }

    // Resolve pointee type for the kind switch.
    const TypeInfo* pointee = nullptr;
    auto pIt = w.types.find(v.pointerType);
    if (pIt != w.types.end() && pIt->second.kind == TypeInfo::Pointer) {
      auto innerIt = w.types.find(pIt->second.innerType);
      if (innerIt != w.types.end()) pointee = &innerIt->second;
    }

    bool ro = false;
    ResourceKind kind = resolveResourceKind(w, v, pointee, ro);
    if (w.nonWritableVars.count(kv.first)) ro = true;
    if (pointee && pointee->kind == TypeInfo::Struct &&
        w.nonWritableStructs.count(pIt->second.innerType)) {
      ro = true;
    }

    ReflectedBinding b;
    b.kind = kind;
    b.set = v.descriptorSet;
    b.binding = v.binding;
    b.readOnly = ro;
    out.bindings.push_back(b);
  }

  std::sort(out.bindings.begin(), out.bindings.end(),
            [](const ReflectedBinding& a, const ReflectedBinding& b) {
              if (a.set != b.set) return a.set < b.set;
              return a.binding < b.binding;
            });
  return true;
}

}  // namespace vk
}  // namespace ghost

#endif  // WITH_VULKAN
