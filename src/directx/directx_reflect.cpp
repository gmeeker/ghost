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

#if WITH_DIRECTX

#include <ghost/directx/reflect.h>

#include <algorithm>
#include <cstring>

namespace ghost {
namespace dx {

namespace {

// DXIL container layout. See
// https://github.com/microsoft/DirectXShaderCompiler/blob/main/include/dxc/DxilContainer/DxilContainer.h
//
// struct DxilContainerHeader {
//   uint32_t  HeaderFourCC;       // 'DXBC' = 0x43425844
//   uint8_t   HashDigest[16];
//   uint16_t  MajorVersion;       // 1
//   uint16_t  MinorVersion;       // 0
//   uint32_t  ContainerSizeInBytes;
//   uint32_t  PartCount;
//   // followed by uint32_t PartOffset[PartCount]
// };
//
// struct DxilPartHeader {
//   uint32_t  PartFourCC;
//   uint32_t  PartSize;            // bytes following this header
// };
//
// PSV0 part body (from DxilPipelineStateValidation.h):
//   uint32_t                 PSVRuntimeInfoSize;
//   PSVRuntimeInfo<N>        Info;             // size = PSVRuntimeInfoSize
//   uint32_t                 ResourceCount;
//   if (ResourceCount > 0):
//     uint32_t               PSVResourceBindInfoSize;
//     PSVResourceBindInfo<N> Resources[ResourceCount];
//   ... per-stage signature info, not relevant for compute ...
//
// PSVResourceBindInfo0 (16 bytes; older shaders):
//   uint32_t ResType;       // PSVResourceType
//   uint32_t Space;
//   uint32_t LowerBound;
//   uint32_t UpperBound;
//
// PSVResourceBindInfo1 (24 bytes; SM 6.6+):
//   PSVResourceBindInfo0 ...;
//   uint32_t ResKind;       // PSVResourceKind
//   uint32_t ResFlags;
//
// PSVResourceType enum (from DxilPipelineStateValidation.h):
//   0 Invalid, 1 Sampler, 2 CBV,
//   3 SRVTyped, 4 SRVRaw, 5 SRVStructured,
//   6 UAVTyped, 7 UAVRaw, 8 UAVStructured,
//   9 UAVTypedWithCounter, 10 UAVRawWithCounter, 11 UAVStructuredWithCounter

constexpr uint32_t kFourCC_DXBC = 0x43425844;  // 'DXBC' little-endian
constexpr uint32_t kFourCC_PSV0 = 0x30565350;  // 'PSV0' little-endian

constexpr uint32_t kPSVResType_Invalid = 0;
constexpr uint32_t kPSVResType_Sampler = 1;
constexpr uint32_t kPSVResType_CBV = 2;
constexpr uint32_t kPSVResType_SRVTyped = 3;
constexpr uint32_t kPSVResType_SRVRaw = 4;
constexpr uint32_t kPSVResType_SRVStructured = 5;
constexpr uint32_t kPSVResType_UAVTyped = 6;
constexpr uint32_t kPSVResType_UAVRaw = 7;
constexpr uint32_t kPSVResType_UAVStructured = 8;
constexpr uint32_t kPSVResType_UAVTypedWithCounter = 9;
constexpr uint32_t kPSVResType_UAVRawWithCounter = 10;
constexpr uint32_t kPSVResType_UAVStructuredWithCounter = 11;

// Read a little-endian uint32 from an unaligned pointer.
inline uint32_t rd32(const uint8_t* p) {
  uint32_t v;
  memcpy(&v, p, sizeof(uint32_t));
  return v;
}

bool classifyResource(uint32_t resType, RegisterClass& outClass,
                      ResourceShape& outShape) {
  switch (resType) {
    case kPSVResType_CBV:
      outClass = RegisterClass::CBV;
      outShape = ResourceShape::CBV;
      return true;
    case kPSVResType_SRVRaw:
    case kPSVResType_SRVStructured:
      outClass = RegisterClass::SRV;
      outShape = ResourceShape::StructuredOrRawBuffer;
      return true;
    case kPSVResType_SRVTyped:
      outClass = RegisterClass::SRV;
      outShape = ResourceShape::TypedBuffer;
      return true;
    case kPSVResType_UAVRaw:
    case kPSVResType_UAVStructured:
    case kPSVResType_UAVRawWithCounter:
    case kPSVResType_UAVStructuredWithCounter:
      outClass = RegisterClass::UAV;
      outShape = ResourceShape::StructuredOrRawBuffer;
      return true;
    case kPSVResType_UAVTyped:
    case kPSVResType_UAVTypedWithCounter:
      outClass = RegisterClass::UAV;
      outShape = ResourceShape::TypedBuffer;
      return true;
    case kPSVResType_Sampler:
      outClass = RegisterClass::Sampler;
      outShape = ResourceShape::Sampler;
      return true;
    case kPSVResType_Invalid:
    default:
      return false;
  }
}

bool parsePSV0(const uint8_t* part, size_t partSize, ReflectedDxilShader& out) {
  // PSV0 minimum: 4 bytes for runtime-info-size, runtime info, 4 bytes for
  // resource count.
  if (partSize < 8) return false;
  size_t off = 0;
  uint32_t runtimeInfoSize = rd32(part + off);
  off += 4;
  if (runtimeInfoSize > partSize - off) return false;
  off += runtimeInfoSize;
  if (off + 4 > partSize) return false;
  uint32_t resCount = rd32(part + off);
  off += 4;
  if (resCount == 0) return true;  // shader uses no resources — also valid
  if (off + 4 > partSize) return false;
  uint32_t bindInfoSize = rd32(part + off);
  off += 4;
  // Sanity: PSVResourceBindInfo0 is 16 bytes, v1 is 24. Anything smaller
  // than 16 means we're looking at something we don't recognize.
  if (bindInfoSize < 16) return false;
  if ((uint64_t)bindInfoSize * resCount > partSize - off) return false;

  out.resources.reserve(resCount);
  for (uint32_t i = 0; i < resCount; i++) {
    const uint8_t* entry = part + off + (size_t)i * bindInfoSize;
    uint32_t resType = rd32(entry + 0);
    uint32_t space = rd32(entry + 4);
    uint32_t lower = rd32(entry + 8);
    uint32_t upper = rd32(entry + 12);

    ReflectedResource r;
    if (!classifyResource(resType, r.regClass, r.shape)) continue;
    r.space = space;
    r.lowerBound = lower;
    r.upperBound = upper;
    out.resources.push_back(r);
  }

  std::sort(out.resources.begin(), out.resources.end(),
            [](const ReflectedResource& a, const ReflectedResource& b) {
              if (a.regClass != b.regClass)
                return (uint8_t)a.regClass < (uint8_t)b.regClass;
              if (a.space != b.space) return a.space < b.space;
              return a.lowerBound < b.lowerBound;
            });
  return true;
}

}  // namespace

bool reflectDxilContainer(const void* data, size_t sizeBytes,
                          ReflectedDxilShader& out) {
  out.resources.clear();
  if (!data || sizeBytes < 32) return false;
  const uint8_t* base = static_cast<const uint8_t*>(data);

  // Container header (32 bytes minimum, plus 4*partCount).
  uint32_t fourcc = rd32(base + 0);
  if (fourcc != kFourCC_DXBC) return false;
  uint32_t containerSize = rd32(base + 24);
  uint32_t partCount = rd32(base + 28);
  if (containerSize > sizeBytes) return false;
  if ((uint64_t)32 + (uint64_t)4 * partCount > containerSize) return false;

  for (uint32_t i = 0; i < partCount; i++) {
    uint32_t partOffset = rd32(base + 32 + 4 * i);
    if (partOffset + 8 > containerSize) return false;
    uint32_t partFourCC = rd32(base + partOffset);
    uint32_t partSize = rd32(base + partOffset + 4);
    if ((uint64_t)partOffset + 8 + partSize > containerSize) return false;
    if (partFourCC != kFourCC_PSV0) continue;
    return parsePSV0(base + partOffset + 8, partSize, out);
  }
  return false;
}

}  // namespace dx
}  // namespace ghost

#endif  // WITH_DIRECTX
