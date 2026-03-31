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

#include <ghost/argument_buffer.h>
#include <ghost/vulkan/exception.h>
#include <ghost/vulkan/impl_device.h>
#include <ghost/vulkan/impl_function.h>

#include <cstring>

namespace ghost {
namespace implementation {
using namespace vk;

// ---------------------------------------------------------------------------
// FunctionVulkan
// ---------------------------------------------------------------------------

FunctionVulkan::FunctionVulkan(const DeviceVulkan& dev, VkShaderModule module,
                               const std::string& entryPoint)
    : _dev(dev),
      _module(module),
      _entryPoint(entryPoint),
      _descriptorSetLayout(VK_NULL_HANDLE),
      _pipelineLayout(VK_NULL_HANDLE),
      _pipeline(VK_NULL_HANDLE),
      _pipelineCreated(false),
      _numBuffers(0),
      _numImages(0),
      _pushConstantSize(0) {}

FunctionVulkan::~FunctionVulkan() {
  if (_pipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(_dev.device, _pipeline, nullptr);
  if (_pipelineLayout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(_dev.device, _pipelineLayout, nullptr);
  if (_descriptorSetLayout != VK_NULL_HANDLE)
    vkDestroyDescriptorSetLayout(_dev.device, _descriptorSetLayout, nullptr);
}

void FunctionVulkan::createPipeline(const std::vector<Attribute>& args) {
  // Count argument types to determine layout
  _numBuffers = 0;
  _numImages = 0;
  _pushConstantSize = 0;

  for (auto& arg : args) {
    switch (arg.type()) {
      case Attribute::Type_Buffer:
        _numBuffers++;
        break;
      case Attribute::Type_Image:
        _numImages++;
        break;
      case Attribute::Type_Float:
        _pushConstantSize += (uint32_t)(sizeof(float) * arg.count());
        break;
      case Attribute::Type_Int:
        _pushConstantSize += (uint32_t)(sizeof(int32_t) * arg.count());
        break;
      case Attribute::Type_Bool:
        _pushConstantSize += (uint32_t)(sizeof(uint32_t) * arg.count());
        break;
      case Attribute::Type_ArgumentBuffer: {
        auto* ab = arg.asArgumentBuffer();
        if (ab->isStruct()) {
          _pushConstantSize += (uint32_t)ab->size();
        } else {
          _numBuffers++;
        }
        break;
      }
      case Attribute::Type_LocalMem:
        // Vulkan shared memory is declared in the shader
        break;
      default:
        break;
    }
  }

  // Align push constant size to 4 bytes
  _pushConstantSize = (_pushConstantSize + 3) & ~3u;

  // Create descriptor set layout
  std::vector<VkDescriptorSetLayoutBinding> bindings;
  uint32_t bindingIdx = 0;

  for (uint32_t i = 0; i < _numBuffers; i++) {
    VkDescriptorSetLayoutBinding binding = {};
    binding.binding = bindingIdx++;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings.push_back(binding);
  }

  for (uint32_t i = 0; i < _numImages; i++) {
    VkDescriptorSetLayoutBinding binding = {};
    binding.binding = bindingIdx++;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings.push_back(binding);
  }

  VkDescriptorSetLayoutCreateInfo layoutInfo = {};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = (uint32_t)bindings.size();
  layoutInfo.pBindings = bindings.data();
  checkError(vkCreateDescriptorSetLayout(_dev.device, &layoutInfo, nullptr,
                                         &_descriptorSetLayout));

  // Create pipeline layout
  VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &_descriptorSetLayout;

  VkPushConstantRange pushRange = {};
  if (_pushConstantSize > 0) {
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = _pushConstantSize;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
  }

  checkError(vkCreatePipelineLayout(_dev.device, &pipelineLayoutInfo, nullptr,
                                    &_pipelineLayout));

  // Create compute pipeline
  VkComputePipelineCreateInfo pipelineInfo = {};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineInfo.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipelineInfo.stage.module = _module;
  pipelineInfo.stage.pName = _entryPoint.c_str();
  pipelineInfo.layout = _pipelineLayout;

  checkError(vkCreateComputePipelines(_dev.device, VK_NULL_HANDLE, 1,
                                      &pipelineInfo, nullptr, &_pipeline));

  _pipelineCreated = true;
}

void FunctionVulkan::execute(const ghost::Stream& s,
                             const LaunchArgs& launchArgs,
                             const std::vector<Attribute>& args) {
  auto& stream = *static_cast<StreamVulkan*>(s.impl().get());

  // Create pipeline on first execution
  if (!_pipelineCreated) {
    createPipeline(args);
  }

  stream.begin();

  // Allocate descriptor set
  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = _dev.descriptorPool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &_descriptorSetLayout;

  VkDescriptorSet descriptorSet;
  checkError(vkAllocateDescriptorSets(_dev.device, &allocInfo, &descriptorSet));

  // Write descriptors and push constants
  std::vector<VkWriteDescriptorSet> writes;
  std::vector<VkDescriptorBufferInfo> bufferInfos;
  std::vector<VkDescriptorImageInfo> imageInfos;
  bufferInfos.reserve(_numBuffers);
  imageInfos.reserve(_numImages);

  std::vector<uint8_t> pushData;
  pushData.reserve(_pushConstantSize);

  uint32_t bindingIdx = 0;

  for (auto& arg : args) {
    switch (arg.type()) {
      case Attribute::Type_Buffer: {
        auto* buf = arg.asBuffer();
        auto* vkBuf = static_cast<BufferVulkan*>(buf->impl().get());

        VkDescriptorBufferInfo bufInfo = {};
        bufInfo.buffer = vkBuf->buffer;
        bufInfo.offset = vkBuf->baseOffset();
        bufInfo.range = vkBuf->size();
        bufferInfos.push_back(bufInfo);

        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptorSet;
        write.dstBinding = bindingIdx++;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &bufferInfos.back();
        writes.push_back(write);
        break;
      }
      case Attribute::Type_Image: {
        auto* img = arg.asImage();
        auto* vkImg = static_cast<ImageVulkan*>(img->impl().get());

        VkDescriptorImageInfo imgInfo = {};
        imgInfo.imageView = vkImg->imageView;
        imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageInfos.push_back(imgInfo);

        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptorSet;
        write.dstBinding = bindingIdx++;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write.pImageInfo = &imageInfos.back();
        writes.push_back(write);
        break;
      }
      case Attribute::Type_Float: {
        size_t n = arg.count();
        const float* vals = arg.floatArray();
        size_t sz = sizeof(float) * n;
        size_t off = pushData.size();
        pushData.resize(off + sz);
        memcpy(pushData.data() + off, vals, sz);
        break;
      }
      case Attribute::Type_Int: {
        size_t n = arg.count();
        const int32_t* vals = arg.intArray();
        size_t sz = sizeof(int32_t) * n;
        size_t off = pushData.size();
        pushData.resize(off + sz);
        memcpy(pushData.data() + off, vals, sz);
        break;
      }
      case Attribute::Type_Bool: {
        size_t n = arg.count();
        const bool* vals = arg.boolArray();
        for (size_t i = 0; i < n; i++) {
          uint32_t v = vals[i] ? 1 : 0;
          size_t off = pushData.size();
          pushData.resize(off + sizeof(uint32_t));
          memcpy(pushData.data() + off, &v, sizeof(uint32_t));
        }
        break;
      }
      case Attribute::Type_ArgumentBuffer: {
        auto* ab = arg.asArgumentBuffer();
        if (ab->isStruct()) {
          size_t sz = ab->size();
          size_t off = pushData.size();
          pushData.resize(off + sz);
          memcpy(pushData.data() + off, ab->data(), sz);
        } else {
          auto bufImpl = ab->bufferImpl();
          auto* vkBuf = static_cast<BufferVulkan*>(bufImpl.get());

          VkDescriptorBufferInfo bufInfo = {};
          bufInfo.buffer = vkBuf->buffer;
          bufInfo.offset = 0;
          bufInfo.range = vkBuf->size();
          bufferInfos.push_back(bufInfo);

          VkWriteDescriptorSet write = {};
          write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
          write.dstSet = descriptorSet;
          write.dstBinding = bindingIdx++;
          write.descriptorCount = 1;
          write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          write.pBufferInfo = &bufferInfos.back();
          writes.push_back(write);
        }
        break;
      }
      case Attribute::Type_LocalMem:
        break;
      default:
        break;
    }
  }

  if (!writes.empty()) {
    vkUpdateDescriptorSets(_dev.device, (uint32_t)writes.size(), writes.data(),
                           0, nullptr);
  }

  // Record compute dispatch
  vkCmdBindPipeline(stream.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    _pipeline);
  vkCmdBindDescriptorSets(stream.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          _pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

  if (!pushData.empty()) {
    // Pad to 4-byte alignment
    while (pushData.size() % 4 != 0) pushData.push_back(0);
    vkCmdPushConstants(stream.commandBuffer, _pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       (uint32_t)pushData.size(), pushData.data());
  }

  vkCmdDispatch(stream.commandBuffer,
                launchArgs.dims() >= 1 ? (uint32_t)launchArgs.count(0) : 1,
                launchArgs.dims() >= 2 ? (uint32_t)launchArgs.count(1) : 1,
                launchArgs.dims() >= 3 ? (uint32_t)launchArgs.count(2) : 1);

  // Memory barrier for compute shader writes
  VkMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(
      stream.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
      1, &barrier, 0, nullptr, 0, nullptr);
}

Attribute FunctionVulkan::getAttribute(FunctionAttributeId what) const {
  switch (what) {
    case kFunctionMaxThreads:
      return Attribute(
          (int32_t)_dev.properties.limits.maxComputeWorkGroupInvocations);
    case kFunctionRequiredWorkSize:
      return Attribute((int32_t)0, (int32_t)0, (int32_t)0);
    case kFunctionPreferredWorkMultiple:
      // Typical subgroup size for most Vulkan GPUs
      return Attribute((int32_t)32);
    case kFunctionThreadWidth:
      return Attribute((int32_t)32);
    default:
      return Attribute();
  }
}

// ---------------------------------------------------------------------------
// LibraryVulkan
// ---------------------------------------------------------------------------

LibraryVulkan::LibraryVulkan(const DeviceVulkan& dev, bool retainBinary)
    : Library(retainBinary), _dev(dev), _module(VK_NULL_HANDLE) {}

LibraryVulkan::~LibraryVulkan() {
  if (_module != VK_NULL_HANDLE)
    vkDestroyShaderModule(_dev.device, _module, nullptr);
}

void LibraryVulkan::loadFromCache(const void* data, size_t length,
                                  const std::string& options) {
  auto& cache = Device::binaryCache();
  if (!cache.isEnabled()) return;

  std::vector<std::vector<unsigned char>> binaries;
  std::vector<size_t> sizes;
  if (cache.loadBinaries(binaries, sizes, _dev, data, length, options)) {
    if (!binaries.empty() && !binaries[0].empty()) {
      VkShaderModuleCreateInfo createInfo = {};
      createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
      createInfo.codeSize = binaries[0].size();
      createInfo.pCode = reinterpret_cast<const uint32_t*>(binaries[0].data());

      if (vkCreateShaderModule(_dev.device, &createInfo, nullptr, &_module) ==
          VK_SUCCESS) {
        return;
      }
      _module = VK_NULL_HANDLE;
    }
  }
}

void LibraryVulkan::saveToCache(const void* data, size_t length,
                                const std::string& options) const {
  auto& cache = Device::binaryCache();
  if (!cache.isEnabled()) return;

  std::vector<unsigned char*> binaries = {
      const_cast<unsigned char*>(static_cast<const unsigned char*>(data))};
  std::vector<size_t> sizes = {length};
  cache.saveBinaries(_dev, binaries, sizes, data, length, options);
}

void LibraryVulkan::loadFromData(const void* data, size_t len,
                                 const std::string& options) {
  // Try cache first
  loadFromCache(data, len, options);
  if (_module != VK_NULL_HANDLE) return;

  // SPIR-V data must be uint32-aligned
  VkShaderModuleCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = len;
  createInfo.pCode = reinterpret_cast<const uint32_t*>(data);

  checkError(vkCreateShaderModule(_dev.device, &createInfo, nullptr, &_module));

  if (retainBinary()) {
    auto bytes = reinterpret_cast<const uint8_t*>(data);
    _spirvData.assign(bytes, bytes + len);
  }

  saveToCache(data, len, options);
}

ghost::Function LibraryVulkan::lookupFunction(const std::string& name) const {
  // Each function gets its own reference to the same shader module.
  // The FunctionVulkan does NOT own the shader module (LibraryVulkan does).
  // We pass the module but the function won't destroy it.
  return ghost::Function(std::make_shared<FunctionVulkan>(_dev, _module, name));
}

std::vector<uint8_t> LibraryVulkan::getBinary() const { return _spirvData; }

}  // namespace implementation
}  // namespace ghost

#endif
