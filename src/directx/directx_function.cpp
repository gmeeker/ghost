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

#include <ghost/argument_buffer.h>
#include <ghost/directx/exception.h>
#include <ghost/directx/impl_device.h>
#include <ghost/directx/impl_function.h>
#include <ghost/directx/reflect.h>
#include <ghost/exception.h>

#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>

namespace ghost {
namespace implementation {
using namespace dx;

namespace {

// Append a scalar/struct Attribute's host bytes onto a vector. Returns
// false if the Attribute type is not packable into a constant block.
bool appendScalarBytes(const Attribute& a, std::vector<uint8_t>& out) {
  size_t before = out.size();
  switch (a.type()) {
    case Attribute::Type_Float: {
      size_t sz = sizeof(float) * a.count();
      out.resize(before + sz);
      memcpy(out.data() + before, a.floatArray(), sz);
      return true;
    }
    case Attribute::Type_Int: {
      size_t sz = sizeof(int32_t) * a.count();
      out.resize(before + sz);
      memcpy(out.data() + before, a.intArray(), sz);
      return true;
    }
    case Attribute::Type_UInt: {
      size_t sz = sizeof(uint32_t) * a.count();
      out.resize(before + sz);
      memcpy(out.data() + before, a.uintArray(), sz);
      return true;
    }
    case Attribute::Type_Bool: {
      for (size_t i = 0; i < a.count(); i++) {
        uint32_t v = a.boolArray()[i] ? 1u : 0u;
        size_t pos = out.size();
        out.resize(pos + sizeof(uint32_t));
        memcpy(out.data() + pos, &v, sizeof(uint32_t));
      }
      return true;
    }
    case Attribute::Type_ArgumentBuffer: {
      auto* ab = a.argumentBuffer();
      if (ab && ab->isStruct()) {
        out.resize(before + ab->size());
        memcpy(out.data() + before, ab->data(), ab->size());
        return true;
      }
      return false;
    }
    default:
      return false;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// FunctionDirectX
// ---------------------------------------------------------------------------

FunctionDirectX::FunctionDirectX(const DeviceDirectX& dev,
                                 ComPtr<ID3D12PipelineState> pso,
                                 ComPtr<ID3D12RootSignature> rootSig,
                                 std::vector<RootSlot> slots)
    : _dev(dev), _pso(pso), _rootSignature(rootSig), _slots(std::move(slots)) {}

FunctionDirectX::~FunctionDirectX() {}

void FunctionDirectX::execute(const ghost::Encoder& s,
                              const LaunchArgs& launchArgs,
                              const std::vector<Attribute>& args) {
  auto& stream = *static_cast<StreamDirectX*>(s.impl().get());

  stream.begin();
  auto* cmdList = stream.commandList.Get();

  stream.bindDescriptorHeaps();
  cmdList->SetPipelineState(_pso.Get());
  cmdList->SetComputeRootSignature(_rootSignature.Get());

  // Walk slots and args together. Slots are in declaration order
  // (UAV → SRV → CBV → Sampler) so the user passes outputs first, then
  // inputs, then constants, then samplers — matching typical HLSL
  // declaration order.
  size_t argIdx = 0;
  auto skipIgnorable = [&]() {
    while (argIdx < args.size()) {
      const Attribute& a = args[argIdx];
      if (a.type() == Attribute::Type_LocalMem ||
          a.type() == Attribute::Type_Unknown) {
        argIdx++;
        continue;
      }
      break;
    }
  };
  auto peekArg = [&]() -> const Attribute* {
    skipIgnorable();
    if (argIdx < args.size()) return &args[argIdx];
    return nullptr;
  };
  auto nextArg = [&]() -> const Attribute* {
    const Attribute* a = peekArg();
    if (a) argIdx++;
    return a;
  };

  // Carries an image.sample() description forward so a following Sampler
  // slot can be auto-filled without an explicit argument.
  std::optional<SamplerDescription> carriedSampler;

  for (auto& slot : _slots) {
    // TableSampler consumes an explicit sampler arg if present, else
    // inherits from the most recent image.sample() — doesn't require an
    // arg, so handle it before the nextArg() check.
    if (slot.kind == RootSlot::TableSampler) {
      SamplerDescription sd;
      bool have = false;
      if (const Attribute* peek = peekArg()) {
        if (peek->type() == Attribute::Type_Sampler && peek->sampler()) {
          sd = *peek->sampler();
          argIdx++;
          have = true;
        }
      }
      if (!have && carriedSampler) {
        sd = *carriedSampler;
        have = true;
      }
      if (!have) {
        throw std::invalid_argument(
            "FunctionDirectX: sampler binding has no matching sampler "
            "argument (pass ghost::sampler() or Image::sample())");
      }

      auto toFilter = [](FilterMode f) {
        return f == FilterMode::Linear ? D3D12_FILTER_MIN_MAG_MIP_LINEAR
                                       : D3D12_FILTER_MIN_MAG_MIP_POINT;
      };
      auto toAddress = [](AddressMode a) {
        switch (a) {
          case AddressMode::Clamp:
            return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
          case AddressMode::Wrap:
            return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
          case AddressMode::Mirror:
            return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
        }
        return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
      };
      D3D12_SAMPLER_DESC sDesc = {};
      sDesc.Filter = toFilter(sd.filter);
      sDesc.AddressU = toAddress(sd.address);
      sDesc.AddressV = toAddress(sd.address);
      sDesc.AddressW = toAddress(sd.address);
      sDesc.MinLOD = 0.0f;
      sDesc.MaxLOD = D3D12_FLOAT32_MAX;
      sDesc.MaxAnisotropy = 1;
      sDesc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;

      D3D12_CPU_DESCRIPTOR_HANDLE cpu;
      D3D12_GPU_DESCRIPTOR_HANDLE gpu;
      stream.allocSamplerSlots(1, cpu, gpu);
      _dev.device->CreateSampler(&sDesc, cpu);
      cmdList->SetComputeRootDescriptorTable(slot.rootParamIndex, gpu);
      carriedSampler.reset();
      continue;
    }

    const Attribute* arg = nextArg();
    if (!arg) {
      throw std::invalid_argument(
          "FunctionDirectX: not enough arguments for shader bindings");
    }

    // Any non-texture-SRV slot invalidates the carried sampler.
    if (slot.kind != RootSlot::TableSRVTex) {
      carriedSampler.reset();
    }

    switch (slot.kind) {
      case RootSlot::TableSampler:
        // handled above
        break;
      case RootSlot::TableSRVTex:
      case RootSlot::TableUAVTex: {
        if (arg->type() != Attribute::Type_Image) {
          throw std::invalid_argument(
              "FunctionDirectX: texture binding requires an Image argument");
        }
        auto* dxImg = static_cast<ImageDirectX*>(arg->imageImpl().get());
        bool isUAV = slot.kind == RootSlot::TableUAVTex;
        dxImg->transitionTo(
            cmdList, isUAV ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                           : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        D3D12_CPU_DESCRIPTOR_HANDLE cpu;
        D3D12_GPU_DESCRIPTOR_HANDLE gpu;
        stream.allocSrvSlots(1, cpu, gpu);

        const auto& d = dxImg->description();
        DXGI_FORMAT format = _dev.getImageFormat(d);
        if (isUAV) {
          D3D12_UNORDERED_ACCESS_VIEW_DESC v = {};
          v.Format = format;
          if (d.size.z > 1) {
            v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            v.Texture3D.WSize = (UINT)d.size.z;
          } else if (d.size.y > 1) {
            v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
          } else {
            v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
          }
          _dev.device->CreateUnorderedAccessView(dxImg->resource.Get(), nullptr,
                                                 &v, cpu);
        } else {
          D3D12_SHADER_RESOURCE_VIEW_DESC v = {};
          v.Format = format;
          v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
          if (d.size.z > 1) {
            v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            v.Texture3D.MipLevels = 1;
          } else if (d.size.y > 1) {
            v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            v.Texture2D.MipLevels = 1;
          } else {
            v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
            v.Texture1D.MipLevels = 1;
          }
          _dev.device->CreateShaderResourceView(dxImg->resource.Get(), &v, cpu);
        }
        cmdList->SetComputeRootDescriptorTable(slot.rootParamIndex, gpu);

        // For SRV textures created via image.sample(), carry the sampler
        // forward so a following TableSampler slot can inherit it.
        if (!isUAV && arg->sampler()) {
          carriedSampler = *arg->sampler();
        }
        break;
      }
      case RootSlot::RootSRV:
      case RootSlot::RootUAV: {
        std::shared_ptr<implementation::Buffer> bufImpl;
        if (arg->type() == Attribute::Type_Buffer) {
          bufImpl = arg->bufferImpl();
        } else if (arg->type() == Attribute::Type_ArgumentBuffer &&
                   arg->argumentBuffer() &&
                   !arg->argumentBuffer()->isStruct()) {
          bufImpl = arg->argumentBuffer()->bufferImpl();
        } else {
          throw std::invalid_argument(
              "FunctionDirectX: SRV/UAV binding requires a Buffer argument");
        }
        auto* dxBuf = static_cast<BufferDirectX*>(bufImpl.get());
        D3D12_RESOURCE_STATES targetState =
            (slot.kind == RootSlot::RootUAV)
                ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        dxBuf->transitionTo(cmdList, targetState);
        D3D12_GPU_VIRTUAL_ADDRESS gva =
            dxBuf->resource->GetGPUVirtualAddress() + dxBuf->baseOffset();
        if (slot.kind == RootSlot::RootUAV) {
          cmdList->SetComputeRootUnorderedAccessView(slot.rootParamIndex, gva);
        } else {
          cmdList->SetComputeRootShaderResourceView(slot.rootParamIndex, gva);
        }
        break;
      }
      case RootSlot::RootCBV: {
        // Pack the user's scalar/struct data into a transient upload buffer
        // and bind it as a root CBV. The buffer is queued on the stream's
        // staging list so it survives until the dispatch completes.
        std::vector<uint8_t> bytes;
        if (!appendScalarBytes(*arg, bytes)) {
          throw std::invalid_argument(
              "FunctionDirectX: CBV binding requires a scalar/struct or "
              "ArgumentBuffer argument");
        }
        // CBVs in D3D12 must be 256-byte aligned in size at allocation time.
        size_t allocSize = (bytes.size() + 255u) & ~size_t{255u};
        if (allocSize == 0) allocSize = 256;

        auto upload = _dev.createCommittedBuffer(
            allocSize, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ);

        void* mapped = nullptr;
        D3D12_RANGE readRange = {0, 0};
        checkHR(upload->Map(0, &readRange, &mapped));
        memcpy(mapped, bytes.data(), bytes.size());
        if (allocSize > bytes.size()) {
          memset(static_cast<uint8_t*>(mapped) + bytes.size(), 0,
                 allocSize - bytes.size());
        }
        upload->Unmap(0, nullptr);

        cmdList->SetComputeRootConstantBufferView(
            slot.rootParamIndex, upload->GetGPUVirtualAddress());
        stream.pendingStaging.push_back({upload});
        break;
      }
    }
  }

  if (launchArgs.requiredSubgroupSize() != 0) {
    uint32_t actual = preferredSubgroupSize();
    if (launchArgs.requiredSubgroupSize() != actual) {
      throw std::invalid_argument(
          "DirectX: requiredSubgroupSize (" +
          std::to_string(launchArgs.requiredSubgroupSize()) +
          ") does not match wave size (" + std::to_string(actual) + ")");
    }
  }

  // Dispatch compute work
  UINT gx = launchArgs.dims() >= 1
                ? (UINT)narrowDim(launchArgs.count(0), "global_size[0] / count")
                : 1;
  UINT gy = launchArgs.dims() >= 2
                ? (UINT)narrowDim(launchArgs.count(1), "global_size[1] / count")
                : 1;
  UINT gz = launchArgs.dims() >= 3
                ? (UINT)narrowDim(launchArgs.count(2), "global_size[2] / count")
                : 1;
  cmdList->Dispatch(gx, gy, gz);

  // UAV barrier for compute shader writes
  D3D12_RESOURCE_BARRIER uavBarrier = {};
  uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  uavBarrier.UAV.pResource = nullptr;  // Barrier on all UAVs
  cmdList->ResourceBarrier(1, &uavBarrier);
}

Attribute FunctionDirectX::getAttribute(FunctionAttributeId what) const {
  switch (what) {
    case kFunctionMaxThreads:
      return Attribute((int32_t)1024);  // D3D12 max threads per group
    case kFunctionRequiredWorkSize:
      return Attribute((int32_t)0, (int32_t)0, (int32_t)0);
    case kFunctionPreferredWorkMultiple:
      return Attribute((int32_t)preferredSubgroupSize());
    case kFunctionThreadWidth:
      return Attribute((int32_t)preferredSubgroupSize());
    default:
      return Attribute();
  }
}

uint32_t FunctionDirectX::preferredSubgroupSize() const {
  D3D12_FEATURE_DATA_D3D12_OPTIONS1 opts1 = {};
  if (SUCCEEDED(_dev.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1,
                                                 &opts1, sizeof(opts1))) &&
      opts1.WaveLaneCountMin > 0) {
    return opts1.WaveLaneCountMin;
  }
  return 32;
}

// ---------------------------------------------------------------------------
// LibraryDirectX
// ---------------------------------------------------------------------------

LibraryDirectX::LibraryDirectX(const DeviceDirectX& dev) : _dev(dev) {}

LibraryDirectX::~LibraryDirectX() {}

void LibraryDirectX::loadFromCache(const void* data, size_t length,
                                   const CompilerOptions& options) {
  auto& cache = Device::binaryCache();
  if (!cache.isEnabled()) return;

  std::vector<std::vector<unsigned char>> binaries;
  std::vector<size_t> sizes;
  if (cache.loadBinaries(binaries, sizes, _dev, data, length, options)) {
    _bytecode = std::move(binaries[0]);
  }
}

void LibraryDirectX::saveToCache(const void* data, size_t length,
                                 const CompilerOptions& options) const {
  auto& cache = Device::binaryCache();
  if (!cache.isEnabled()) return;

  std::vector<unsigned char*> binaries = {
      const_cast<unsigned char*>(_bytecode.data())};
  std::vector<size_t> sizes = {_bytecode.size()};
  cache.saveBinaries(_dev, binaries, sizes, data, length, options);
}

void LibraryDirectX::loadFromData(const void* data, size_t len,
                                  const CompilerOptions& options) {
  // Try cache first
  loadFromCache(data, len, options);
  if (_bytecode.empty()) {
    _bytecode.resize(len);
    memcpy(_bytecode.data(), data, len);
    saveToCache(data, len, options);
  }

  // Parse PSV0 reflection from the DXIL container so lookupFunction can
  // build a matching root signature.
  ghost::dx::reflectDxilContainer(_bytecode.data(), _bytecode.size(),
                                  _reflection);
}

ghost::Function LibraryDirectX::lookupFunction(const std::string& name) const {
  // Build a root signature whose parameters are derived from the reflected
  // resource list. The slot ordering used here mirrors what
  // FunctionDirectX::execute consumes positionally: UAV first, then SRV,
  // then CBV, then Sampler — matching how HLSL compute kernels typically
  // declare "outputs, inputs, params, samplers".
  //
  // Buffers use root SRV/UAV/CBV descriptors. Textures and samplers need
  // descriptor tables because D3D12 can't bind them directly as root
  // descriptors. Each texture/sampler gets its own 1-entry table so the
  // positional argument matcher can walk slots one-by-one regardless of
  // how HLSL numbered the registers.
  std::vector<D3D12_ROOT_PARAMETER> rootParams;
  std::vector<FunctionDirectX::RootSlot> slots;
  // Ranges must outlive rootParams, since D3D12_ROOT_DESCRIPTOR_TABLE stores
  // a raw pointer into this vector.
  std::vector<std::unique_ptr<D3D12_DESCRIPTOR_RANGE>> ranges;
  rootParams.reserve(_reflection.resources.size());
  slots.reserve(_reflection.resources.size());
  ranges.reserve(_reflection.resources.size());

  auto makeTableRange = [&](D3D12_DESCRIPTOR_RANGE_TYPE type, uint32_t reg,
                            uint32_t space, uint32_t count) {
    auto r = std::make_unique<D3D12_DESCRIPTOR_RANGE>();
    r->RangeType = type;
    r->NumDescriptors = count;
    r->BaseShaderRegister = reg;
    r->RegisterSpace = space;
    r->OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    ranges.push_back(std::move(r));
    return ranges.back().get();
  };

  for (auto& r : _reflection.resources) {
    D3D12_ROOT_PARAMETER p = {};
    p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    FunctionDirectX::RootSlot slot;
    slot.rootParamIndex = (uint32_t)rootParams.size();

    const uint32_t rangeCount = r.upperBound - r.lowerBound + 1;

    switch (r.regClass) {
      case ghost::dx::RegisterClass::UAV:
        if (r.shape == ghost::dx::ResourceShape::StructuredOrRawBuffer) {
          p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
          p.Descriptor.ShaderRegister = r.lowerBound;
          p.Descriptor.RegisterSpace = r.space;
          slot.kind = FunctionDirectX::RootSlot::RootUAV;
        } else {
          // Typed UAV → treated as a texture/RWBuffer — descriptor table.
          p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
          p.DescriptorTable.NumDescriptorRanges = 1;
          p.DescriptorTable.pDescriptorRanges =
              makeTableRange(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, r.lowerBound,
                             r.space, rangeCount);
          slot.kind = FunctionDirectX::RootSlot::TableUAVTex;
        }
        break;
      case ghost::dx::RegisterClass::SRV:
        if (r.shape == ghost::dx::ResourceShape::StructuredOrRawBuffer) {
          p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
          p.Descriptor.ShaderRegister = r.lowerBound;
          p.Descriptor.RegisterSpace = r.space;
          slot.kind = FunctionDirectX::RootSlot::RootSRV;
        } else {
          p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
          p.DescriptorTable.NumDescriptorRanges = 1;
          p.DescriptorTable.pDescriptorRanges =
              makeTableRange(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, r.lowerBound,
                             r.space, rangeCount);
          slot.kind = FunctionDirectX::RootSlot::TableSRVTex;
        }
        break;
      case ghost::dx::RegisterClass::CBV:
        p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        p.Descriptor.ShaderRegister = r.lowerBound;
        p.Descriptor.RegisterSpace = r.space;
        slot.kind = FunctionDirectX::RootSlot::RootCBV;
        break;
      case ghost::dx::RegisterClass::Sampler:
        p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p.DescriptorTable.NumDescriptorRanges = 1;
        p.DescriptorTable.pDescriptorRanges =
            makeTableRange(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, r.lowerBound,
                           r.space, rangeCount);
        slot.kind = FunctionDirectX::RootSlot::TableSampler;
        break;
    }

    rootParams.push_back(p);
    slots.push_back(slot);
  }

  D3D12_ROOT_SIGNATURE_DESC rsDesc = {};
  rsDesc.NumParameters = (UINT)rootParams.size();
  rsDesc.pParameters = rootParams.empty() ? nullptr : rootParams.data();
  rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

  ComPtr<ID3DBlob> serialized;
  ComPtr<ID3DBlob> error;
  checkHR(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1,
                                      &serialized, &error));

  ComPtr<ID3D12RootSignature> rootSig;
  checkHR(_dev.device->CreateRootSignature(0, serialized->GetBufferPointer(),
                                           serialized->GetBufferSize(),
                                           IID_PPV_ARGS(&rootSig)));

  // Create compute pipeline state
  D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
  psoDesc.pRootSignature = rootSig.Get();
  psoDesc.CS.pShaderBytecode = _bytecode.data();
  psoDesc.CS.BytecodeLength = _bytecode.size();

  ComPtr<ID3D12PipelineState> pso;
  checkHR(
      _dev.device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pso)));

  return ghost::Function(
      std::make_shared<FunctionDirectX>(_dev, pso, rootSig, std::move(slots)));
}

std::vector<uint8_t> LibraryDirectX::getBinary() const { return _bytecode; }

}  // namespace implementation
}  // namespace ghost

#endif
