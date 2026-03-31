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

#ifndef GHOST_DIRECTX_IMPL_FUNCTION_H
#define GHOST_DIRECTX_IMPL_FUNCTION_H

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d12.h>
#include <ghost/implementation/impl_function.h>
#include <wrl/client.h>

#include <string>
#include <vector>

namespace ghost {
namespace implementation {
class DeviceDirectX;

class FunctionDirectX : public Function {
 public:
  FunctionDirectX(const DeviceDirectX& dev,
                  Microsoft::WRL::ComPtr<ID3D12PipelineState> pso,
                  Microsoft::WRL::ComPtr<ID3D12RootSignature> rootSig,
                  uint32_t numDescriptors);
  ~FunctionDirectX();

  virtual void execute(const ghost::Stream& s, const LaunchArgs& launchArgs,
                       const std::vector<Attribute>& args) override;

  virtual Attribute getAttribute(FunctionAttributeId what) const override;

 private:
  const DeviceDirectX& _dev;
  Microsoft::WRL::ComPtr<ID3D12PipelineState> _pso;
  Microsoft::WRL::ComPtr<ID3D12RootSignature> _rootSignature;
  uint32_t _numDescriptors;
};

class LibraryDirectX : public Library {
 public:
  LibraryDirectX(const DeviceDirectX& dev);
  ~LibraryDirectX();

  void loadFromData(const void* data, size_t len, const std::string& options);
  virtual ghost::Function lookupFunction(
      const std::string& name) const override;
  virtual std::vector<uint8_t> getBinary() const override;

 private:
  void loadFromCache(const void* data, size_t length,
                     const std::string& options);
  void saveToCache(const void* data, size_t length,
                   const std::string& options) const;

  const DeviceDirectX& _dev;
  std::vector<uint8_t> _bytecode;
};
}  // namespace implementation
}  // namespace ghost

#endif
