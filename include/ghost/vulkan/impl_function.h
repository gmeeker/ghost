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

#ifndef GHOST_VULKAN_IMPL_FUNCTION_H
#define GHOST_VULKAN_IMPL_FUNCTION_H

#include <ghost/implementation/impl_function.h>
#include <ghost/vulkan/ptr.h>

#include <string>
#include <vector>

namespace ghost {
namespace implementation {
class DeviceVulkan;

class FunctionVulkan : public Function {
 public:
  FunctionVulkan(const DeviceVulkan& dev, VkShaderModule module,
                 const std::string& entryPoint);
  FunctionVulkan(const DeviceVulkan& dev, VkShaderModule module,
                 const std::string& entryPoint,
                 const std::vector<Attribute>& specConstants);

  virtual void execute(const ghost::Stream& s, const LaunchArgs& launchArgs,
                       const std::vector<Attribute>& args) override;

  virtual Attribute getAttribute(FunctionAttributeId what) const override;

  virtual uint32_t preferredSubgroupSize() const override;

 private:
  void createPipeline(const std::vector<Attribute>& args);
  void buildSpecializationData(const std::vector<Attribute>& specConstants);

  const DeviceVulkan& _dev;
  // Borrowed from the parent LibraryVulkan; not owned here.
  VkShaderModule _module;
  std::string _entryPoint;

  vk::ptr<VkDescriptorSetLayout> _descriptorSetLayout;
  vk::ptr<VkPipelineLayout> _pipelineLayout;
  vk::ptr<VkPipeline> _pipeline;
  bool _pipelineCreated;
  uint32_t _numBuffers;
  uint32_t _numImages;
  uint32_t _pushConstantSize;

  std::vector<uint8_t> _specData;
  std::vector<VkSpecializationMapEntry> _specEntries;
};

class LibraryVulkan : public Library {
 public:
  LibraryVulkan(const DeviceVulkan& dev, bool retainBinary = false);

  void loadFromData(const void* data, size_t len,
                    const CompilerOptions& options);
  virtual ghost::Function lookupFunction(
      const std::string& name) const override;
  virtual ghost::Function specializeFunction(
      const std::string& name,
      const std::vector<Attribute>& args) const override;
  virtual std::vector<uint8_t> getBinary() const override;

 private:
  void loadFromCache(const void* data, size_t length,
                     const CompilerOptions& options);
  void saveToCache(const void* data, size_t length,
                   const CompilerOptions& options) const;

  const DeviceVulkan& _dev;
  vk::ptr<VkShaderModule> _module;
  std::vector<uint8_t> _spirvData;
};
}  // namespace implementation
}  // namespace ghost

#endif
