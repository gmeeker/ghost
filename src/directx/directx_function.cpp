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

#include <cstring>

namespace ghost {
namespace implementation {
using namespace dx;

// ---------------------------------------------------------------------------
// FunctionDirectX
// ---------------------------------------------------------------------------

FunctionDirectX::FunctionDirectX(const DeviceDirectX& dev,
                                 ComPtr<ID3D12PipelineState> pso,
                                 ComPtr<ID3D12RootSignature> rootSig,
                                 uint32_t numDescriptors)
    : _dev(dev),
      _pso(pso),
      _rootSignature(rootSig),
      _numDescriptors(numDescriptors) {}

FunctionDirectX::~FunctionDirectX() {}

void FunctionDirectX::execute(const ghost::Stream& s,
                              const LaunchArgs& launchArgs,
                              const std::vector<Attribute>& args) {
  auto& stream = *static_cast<StreamDirectX*>(s.impl().get());

  stream.begin();
  auto* cmdList = stream.commandList.Get();

  cmdList->SetPipelineState(_pso.Get());
  cmdList->SetComputeRootSignature(_rootSignature.Get());

  // Bind arguments as root parameters:
  // Root parameter 0: root constants (scalar data)
  // Root parameters 1..N: UAV descriptors for buffers/images
  uint32_t rootIdx = 0;

  // Collect scalar data for root constants
  std::vector<uint8_t> constants;
  for (auto& arg : args) {
    switch (arg.type()) {
      case Attribute::Type_Float: {
        size_t sz = sizeof(float) * arg.count();
        size_t off = constants.size();
        constants.resize(off + sz);
        memcpy(constants.data() + off, arg.floatArray(), sz);
        break;
      }
      case Attribute::Type_Int: {
        size_t sz = sizeof(int32_t) * arg.count();
        size_t off = constants.size();
        constants.resize(off + sz);
        memcpy(constants.data() + off, arg.intArray(), sz);
        break;
      }
      case Attribute::Type_Bool: {
        for (size_t i = 0; i < arg.count(); i++) {
          uint32_t v = arg.boolArray()[i] ? 1 : 0;
          size_t off = constants.size();
          constants.resize(off + sizeof(uint32_t));
          memcpy(constants.data() + off, &v, sizeof(uint32_t));
        }
        break;
      }
      case Attribute::Type_ArgumentBuffer: {
        auto* ab = arg.asArgumentBuffer();
        if (ab->isStruct()) {
          size_t sz = ab->size();
          size_t off = constants.size();
          constants.resize(off + sz);
          memcpy(constants.data() + off, ab->data(), sz);
        }
        break;
      }
      default:
        break;
    }
  }

  // Set root constants (pad to DWORD alignment)
  while (constants.size() % 4 != 0) constants.push_back(0);
  if (!constants.empty()) {
    cmdList->SetComputeRoot32BitConstants(rootIdx, (UINT)(constants.size() / 4),
                                          constants.data(), 0);
  }
  rootIdx++;

  // Bind buffers and images as UAV descriptors
  for (auto& arg : args) {
    switch (arg.type()) {
      case Attribute::Type_Buffer: {
        auto* buf = arg.asBuffer();
        auto* dxBuf = static_cast<BufferDirectX*>(buf->impl().get());
        dxBuf->transitionTo(cmdList, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmdList->SetComputeRootUnorderedAccessView(
            rootIdx++, dxBuf->resource->GetGPUVirtualAddress());
        break;
      }
      case Attribute::Type_Image: {
        // Images require a descriptor heap for UAV binding.
        // For simplicity, use a GPU virtual address if the resource supports
        // it. Full implementation would use a shader-visible descriptor heap.
        auto* img = arg.asImage();
        auto* dxImg = static_cast<ImageDirectX*>(img->impl().get());
        dxImg->transitionTo(cmdList, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        // Note: Textures cannot use root UAV directly; they require descriptor
        // tables. A full implementation would create a shader-visible
        // descriptor heap entry here. For buffers treated as raw/structured, we
        // can use root UAV.
        rootIdx++;
        break;
      }
      case Attribute::Type_ArgumentBuffer: {
        auto* ab = arg.asArgumentBuffer();
        if (!ab->isStruct()) {
          auto bufImpl = ab->bufferImpl();
          auto* dxBuf = static_cast<BufferDirectX*>(bufImpl.get());
          dxBuf->transitionTo(cmdList, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
          cmdList->SetComputeRootUnorderedAccessView(
              rootIdx++, dxBuf->resource->GetGPUVirtualAddress());
        }
        break;
      }
      default:
        break;
    }
  }

  // Dispatch compute work
  cmdList->Dispatch(launchArgs.dims() >= 1 ? (UINT)launchArgs.count(0) : 1,
                    launchArgs.dims() >= 2 ? (UINT)launchArgs.count(1) : 1,
                    launchArgs.dims() >= 3 ? (UINT)launchArgs.count(2) : 1);

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
      return Attribute((int32_t)32);  // Typical warp/wavefront size
    case kFunctionThreadWidth:
      return Attribute((int32_t)32);
    default:
      return Attribute();
  }
}

// ---------------------------------------------------------------------------
// LibraryDirectX
// ---------------------------------------------------------------------------

LibraryDirectX::LibraryDirectX(const DeviceDirectX& dev) : _dev(dev) {}

LibraryDirectX::~LibraryDirectX() {}

void LibraryDirectX::loadFromCache(const void* data, size_t length,
                                   const std::string& options) {
  auto& cache = Device::binaryCache();
  if (!cache.isEnabled()) return;

  Digest d;
  Device::binaryCache().makeDigest(d, _dev, 0, data, length, options);

  size_t binaryLen = 0;
  if (cache.loadBinaries(d, nullptr, binaryLen)) {
    _bytecode.resize(binaryLen);
    cache.loadBinaries(d, _bytecode.data(), binaryLen);
  }
}

void LibraryDirectX::saveToCache(const void* data, size_t length,
                                 const std::string& options) const {
  auto& cache = Device::binaryCache();
  if (!cache.isEnabled()) return;

  Digest d;
  Device::binaryCache().makeDigest(d, _dev, 0, data, length, options);
  cache.saveBinaries(d, data, length);
}

void LibraryDirectX::loadFromData(const void* data, size_t len,
                                  const std::string& options) {
  // Try cache first
  loadFromCache(data, len, options);
  if (!_bytecode.empty()) return;

  // Store DXIL/CSO bytecode
  _bytecode.resize(len);
  memcpy(_bytecode.data(), data, len);

  saveToCache(data, len, options);
}

ghost::Function LibraryDirectX::lookupFunction(const std::string& name) const {
  // Count expected bindings from the bytecode.
  // A full implementation would parse DXIL reflection to determine root
  // signature requirements. For now, create a generic root signature with:
  // - Root parameter 0: root constants (32 DWORDs max = 128 bytes)
  // - Root parameters 1-16: UAV descriptors

  // Create root signature
  D3D12_ROOT_PARAMETER rootParams[17] = {};
  uint32_t numParams = 0;

  // Root constants (parameter 0)
  rootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  rootParams[0].Constants.ShaderRegister = 0;
  rootParams[0].Constants.RegisterSpace = 0;
  rootParams[0].Constants.Num32BitValues = 32;
  rootParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  numParams++;

  // UAV descriptors (parameters 1-16)
  for (uint32_t i = 0; i < 16; i++) {
    rootParams[numParams].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    rootParams[numParams].Descriptor.ShaderRegister = i;
    rootParams[numParams].Descriptor.RegisterSpace = 0;
    rootParams[numParams].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    numParams++;
  }

  D3D12_ROOT_SIGNATURE_DESC rsDesc = {};
  rsDesc.NumParameters = numParams;
  rsDesc.pParameters = rootParams;
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
      std::make_shared<FunctionDirectX>(_dev, pso, rootSig, numParams));
}

}  // namespace implementation
}  // namespace ghost

#endif
