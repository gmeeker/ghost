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
#include <ghost/vulkan/reflect.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ghost {
namespace implementation {
using namespace vk;

// ---------------------------------------------------------------------------
// FunctionVulkan
// ---------------------------------------------------------------------------

FunctionVulkan::FunctionVulkan(const DeviceVulkan& dev, VkShaderModule module,
                               const std::string& entryPoint,
                               const ghost::vk::ReflectedShader& reflection)
    : _dev(dev),
      _module(module),
      _entryPoint(entryPoint),
      _reflection(reflection),
      _pipelineLayout(dev.device),
      _pipeline(dev.device),
      _pipelineCreated(false) {
  createPipeline();
}

FunctionVulkan::FunctionVulkan(const DeviceVulkan& dev, VkShaderModule module,
                               const std::string& entryPoint,
                               const ghost::vk::ReflectedShader& reflection,
                               const std::vector<Attribute>& specConstants)
    : _dev(dev),
      _module(module),
      _entryPoint(entryPoint),
      _reflection(reflection),
      _pipelineLayout(dev.device),
      _pipeline(dev.device),
      _pipelineCreated(false) {
  buildSpecializationData(specConstants);
  createPipeline();
}

void FunctionVulkan::buildSpecializationData(
    const std::vector<Attribute>& specConstants) {
  uint32_t offset = 0;
  for (uint32_t i = 0; i < (uint32_t)specConstants.size(); i++) {
    auto& attr = specConstants[i];
    VkSpecializationMapEntry entry = {};
    entry.constantID = i;
    entry.offset = offset;
    switch (attr.type()) {
      case Attribute::Type_Bool: {
        entry.size = sizeof(uint32_t);
        uint32_t v = attr.boolArray()[0] ? VK_TRUE : VK_FALSE;
        size_t pos = _specData.size();
        _specData.resize(pos + sizeof(uint32_t));
        memcpy(_specData.data() + pos, &v, sizeof(uint32_t));
        break;
      }
      case Attribute::Type_Int: {
        entry.size = sizeof(int32_t);
        size_t pos = _specData.size();
        _specData.resize(pos + sizeof(int32_t));
        memcpy(_specData.data() + pos, attr.intArray(), sizeof(int32_t));
        break;
      }
      case Attribute::Type_UInt: {
        entry.size = sizeof(uint32_t);
        size_t pos = _specData.size();
        _specData.resize(pos + sizeof(uint32_t));
        memcpy(_specData.data() + pos, attr.uintArray(), sizeof(uint32_t));
        break;
      }
      case Attribute::Type_Float: {
        entry.size = sizeof(float);
        size_t pos = _specData.size();
        _specData.resize(pos + sizeof(float));
        memcpy(_specData.data() + pos, attr.floatArray(), sizeof(float));
        break;
      }
      default:
        continue;
    }
    offset += (uint32_t)entry.size;
    _specEntries.push_back(entry);
  }
}

namespace {

VkDescriptorType vulkanDescriptorType(ghost::vk::ResourceKind kind) {
  switch (kind) {
    case ghost::vk::ResourceKind::StorageBuffer:
      return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case ghost::vk::ResourceKind::UniformBuffer:
      return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    case ghost::vk::ResourceKind::StorageImage:
      return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case ghost::vk::ResourceKind::SampledImage:
      return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case ghost::vk::ResourceKind::Sampler:
      return VK_DESCRIPTOR_TYPE_SAMPLER;
    case ghost::vk::ResourceKind::CombinedImageSampler:
      return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  }
  return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
}

bool isBufferKind(ghost::vk::ResourceKind kind) {
  return kind == ghost::vk::ResourceKind::StorageBuffer ||
         kind == ghost::vk::ResourceKind::UniformBuffer;
}

bool isImageKind(ghost::vk::ResourceKind kind) {
  return kind == ghost::vk::ResourceKind::StorageImage ||
         kind == ghost::vk::ResourceKind::SampledImage ||
         kind == ghost::vk::ResourceKind::CombinedImageSampler;
}

// Append the host bytes of a scalar/struct Attribute to a byte vector.
// Returns false if the Attribute type can't contribute to a constant block.
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

void FunctionVulkan::createPipeline() {
  // Group bindings by descriptor set. Reflection has them in (set, binding)
  // sorted order, so consecutive runs share a set.
  std::vector<std::vector<VkDescriptorSetLayoutBinding>> perSetBindings;
  std::vector<uint32_t>
      perSetIndex;  // setIndex for each entry of perSetBindings

  uint32_t currentSet = UINT32_MAX;
  for (auto& b : _reflection.bindings) {
    if (b.set != currentSet) {
      currentSet = b.set;
      perSetBindings.emplace_back();
      perSetIndex.push_back(b.set);
    }
    VkDescriptorSetLayoutBinding lb = {};
    lb.binding = b.binding;
    lb.descriptorType = vulkanDescriptorType(b.kind);
    lb.descriptorCount = 1;
    lb.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    perSetBindings.back().push_back(lb);
  }

  // Build a VkDescriptorSetLayout per used set, plus null layouts for any
  // gaps in set indices (Vulkan requires contiguous pSetLayouts).
  uint32_t maxSet = 0;
  for (auto s : perSetIndex)
    if (s > maxSet) maxSet = s;
  size_t numSetLayouts = perSetIndex.empty() ? 0 : (maxSet + 1);

  // Empty layout used for any unused set indices below maxSet.
  vk::ptr<VkDescriptorSetLayout> emptyLayout(_dev.device);
  if (numSetLayouts > perSetIndex.size()) {
    VkDescriptorSetLayoutCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = 0;
    info.pBindings = nullptr;
    checkError(
        vkCreateDescriptorSetLayout(_dev.device, &info, nullptr, &emptyLayout));
  }

  std::vector<VkDescriptorSetLayout> setLayoutHandles(numSetLayouts,
                                                      VK_NULL_HANDLE);
  for (size_t i = 0; i < perSetIndex.size(); i++) {
    DescriptorSetInfo dsi(_dev.device);
    dsi.setIndex = perSetIndex[i];
    VkDescriptorSetLayoutCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = (uint32_t)perSetBindings[i].size();
    info.pBindings = perSetBindings[i].data();
    checkError(
        vkCreateDescriptorSetLayout(_dev.device, &info, nullptr, &dsi.layout));
    setLayoutHandles[perSetIndex[i]] = dsi.layout;
    _descriptorSets.push_back(std::move(dsi));
  }
  for (size_t i = 0; i < numSetLayouts; i++) {
    if (setLayoutHandles[i] == VK_NULL_HANDLE) {
      setLayoutHandles[i] = emptyLayout;
    }
  }

  // Pipeline layout: includes the per-set descriptor layouts plus a push
  // constant range derived from reflection.
  VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = (uint32_t)numSetLayouts;
  pipelineLayoutInfo.pSetLayouts =
      numSetLayouts ? setLayoutHandles.data() : nullptr;

  VkPushConstantRange pcRange = {};
  if (_reflection.pushConstants.present && _reflection.pushConstants.size > 0) {
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = _reflection.pushConstants.size;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pcRange;
  }

  checkError(vkCreatePipelineLayout(_dev.device, &pipelineLayoutInfo, nullptr,
                                    &_pipelineLayout));

  // Compute pipeline.
  VkComputePipelineCreateInfo pipelineInfo = {};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineInfo.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipelineInfo.stage.module = _module;
  pipelineInfo.stage.pName = _entryPoint.c_str();
  pipelineInfo.layout = _pipelineLayout;

  VkSpecializationInfo specInfo = {};
  if (!_specEntries.empty()) {
    specInfo.mapEntryCount = (uint32_t)_specEntries.size();
    specInfo.pMapEntries = _specEntries.data();
    specInfo.dataSize = _specData.size();
    specInfo.pData = _specData.data();
    pipelineInfo.stage.pSpecializationInfo = &specInfo;
  }

  checkError(vkCreateComputePipelines(_dev.device, VK_NULL_HANDLE, 1,
                                      &pipelineInfo, nullptr, &_pipeline));

  _pipelineCreated = true;
}

void FunctionVulkan::execute(const ghost::Stream& s,
                             const LaunchArgs& launchArgs,
                             const std::vector<Attribute>& args) {
  auto& stream = *static_cast<StreamVulkan*>(s.impl().get());

  stream.begin();

  // Allocate one VkDescriptorSet per used set. Vulkan requires writes to
  // happen against a concrete descriptor set, not against a layout, so
  // we allocate them fresh per dispatch from the device pool.
  std::vector<VkDescriptorSet> sets;
  if (!_descriptorSets.empty()) {
    std::vector<VkDescriptorSetLayout> layouts;
    layouts.reserve(_descriptorSets.size());
    for (auto& dsi : _descriptorSets) layouts.push_back(dsi.layout);

    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = _dev.descriptorPool;
    allocInfo.descriptorSetCount = (uint32_t)layouts.size();
    allocInfo.pSetLayouts = layouts.data();

    sets.resize(layouts.size());
    checkError(vkAllocateDescriptorSets(_dev.device, &allocInfo, sets.data()));
  }

  // Walk the reflected slot list, consuming user args one-for-one. Bindings
  // come first (in (set, binding) order), then the push-constant block as
  // a single trailing slot.
  std::vector<VkWriteDescriptorSet> writes;
  // Hold backing storage for buffer/image descriptor infos until
  // vkUpdateDescriptorSets is called.
  std::vector<VkDescriptorBufferInfo> bufferInfos;
  std::vector<VkDescriptorImageInfo> imageInfos;
  bufferInfos.reserve(_reflection.bindings.size());
  imageInfos.reserve(_reflection.bindings.size());

  std::vector<uint8_t> pushBytes;

  size_t argIdx = 0;
  auto nextArg = [&]() -> const Attribute* {
    while (argIdx < args.size()) {
      const Attribute& a = args[argIdx++];
      // Skip Type_LocalMem and Type_Unknown — they don't bind to a slot.
      if (a.type() == Attribute::Type_LocalMem) continue;
      if (a.type() == Attribute::Type_Unknown) continue;
      return &a;
    }
    return nullptr;
  };

  // Match bindings.
  for (size_t bIdx = 0; bIdx < _reflection.bindings.size(); bIdx++) {
    const auto& b = _reflection.bindings[bIdx];

    // Find the descriptor set we allocated for b.set.
    size_t setSlot = SIZE_MAX;
    for (size_t i = 0; i < _descriptorSets.size(); i++) {
      if (_descriptorSets[i].setIndex == b.set) {
        setSlot = i;
        break;
      }
    }
    if (setSlot == SIZE_MAX)
      continue;  // shouldn't happen if reflection is correct

    const Attribute* arg = nextArg();
    if (!arg) {
      throw std::invalid_argument(
          "FunctionVulkan: not enough arguments for shader bindings");
    }

    if (isBufferKind(b.kind)) {
      // Two routes: a Buffer arg binds the buffer directly; a scalar/struct
      // arg gets staged into a transient buffer of the right kind.
      if (arg->type() == Attribute::Type_Buffer ||
          (arg->type() == Attribute::Type_ArgumentBuffer &&
           arg->argumentBuffer() && !arg->argumentBuffer()->isStruct())) {
        std::shared_ptr<implementation::Buffer> bufImpl;
        if (arg->type() == Attribute::Type_Buffer) {
          bufImpl = arg->bufferImpl();
        } else {
          bufImpl = arg->argumentBuffer()->bufferImpl();
        }
        auto* vkBuf = static_cast<BufferVulkan*>(bufImpl.get());
        VkDescriptorBufferInfo info = {};
        info.buffer = vkBuf->buffer;
        info.offset = vkBuf->baseOffset();
        info.range = vkBuf->size();
        bufferInfos.push_back(info);

        VkWriteDescriptorSet w = {};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = sets[setSlot];
        w.dstBinding = b.binding;
        w.descriptorCount = 1;
        w.descriptorType = vulkanDescriptorType(b.kind);
        w.pBufferInfo = &bufferInfos.back();
        writes.push_back(w);
      } else {
        // Scalar/struct → transient buffer.
        std::vector<uint8_t> bytes;
        if (!appendScalarBytes(*arg, bytes)) {
          throw std::invalid_argument(
              "FunctionVulkan: unsupported argument type for buffer binding");
        }
        if (bytes.size() < 4) bytes.resize(4, 0);

        BufferOptions opts;
        opts.hint = AllocHint::Staging;  // host-visible upload buffer
        vk::ptr<VkBuffer> tmpBuf(_dev.device);
        vk::ptr<VkDeviceMemory> tmpMem(_dev.device);
        VkBufferUsageFlags usage =
            (b.kind == ghost::vk::ResourceKind::UniformBuffer)
                ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        _dev.createBuffer(bytes.size(), usage,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          tmpBuf, tmpMem);

        void* mapped = nullptr;
        checkError(
            vkMapMemory(_dev.device, tmpMem, 0, bytes.size(), 0, &mapped));
        memcpy(mapped, bytes.data(), bytes.size());
        vkUnmapMemory(_dev.device, tmpMem);

        VkDescriptorBufferInfo info = {};
        info.buffer = tmpBuf;
        info.offset = 0;
        info.range = bytes.size();
        bufferInfos.push_back(info);

        VkWriteDescriptorSet w = {};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = sets[setSlot];
        w.dstBinding = b.binding;
        w.descriptorCount = 1;
        w.descriptorType = vulkanDescriptorType(b.kind);
        w.pBufferInfo = &bufferInfos.back();
        writes.push_back(w);

        // Hand the buffer/memory off to the stream so they get freed
        // after the dispatch completes.
        stream.addStagingResource(std::move(tmpBuf), std::move(tmpMem));
      }
    } else if (isImageKind(b.kind)) {
      if (arg->type() != Attribute::Type_Image) {
        throw std::invalid_argument(
            "FunctionVulkan: argument type does not match image binding");
      }
      auto* vkImg = static_cast<ImageVulkan*>(arg->imageImpl().get());

      VkDescriptorImageInfo info = {};
      info.imageView = vkImg->imageView;
      info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      imageInfos.push_back(info);

      VkWriteDescriptorSet w = {};
      w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w.dstSet = sets[setSlot];
      w.dstBinding = b.binding;
      w.descriptorCount = 1;
      w.descriptorType = vulkanDescriptorType(b.kind);
      w.pImageInfo = &imageInfos.back();
      writes.push_back(w);
    } else {
      throw ghost::unsupported_error();
    }
  }

  // Push constant block (always last in the slot order). Greedily
  // consume any remaining scalar/struct args.
  if (_reflection.pushConstants.present && _reflection.pushConstants.size > 0) {
    while (true) {
      const Attribute* arg = nextArg();
      if (!arg) break;
      if (!appendScalarBytes(*arg, pushBytes)) {
        throw std::invalid_argument(
            "FunctionVulkan: argument can't be packed into push constants");
      }
    }
    if (pushBytes.size() < _reflection.pushConstants.size) {
      pushBytes.resize(_reflection.pushConstants.size, 0);
    } else if (pushBytes.size() > _reflection.pushConstants.size) {
      // Trim to declared size — extra trailing bytes are ignored. This
      // matches what the SPIR-V actually reads.
      pushBytes.resize(_reflection.pushConstants.size);
    }
  }

  if (!writes.empty()) {
    vkUpdateDescriptorSets(_dev.device, (uint32_t)writes.size(), writes.data(),
                           0, nullptr);
  }

  // Record compute dispatch.
  vkCmdBindPipeline(stream.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    _pipeline);
  if (!sets.empty()) {
    vkCmdBindDescriptorSets(stream.commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE, _pipelineLayout, 0,
                            (uint32_t)sets.size(), sets.data(), 0, nullptr);
  }

  if (!pushBytes.empty()) {
    vkCmdPushConstants(stream.commandBuffer, _pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       (uint32_t)pushBytes.size(), pushBytes.data());
  }

  if (launchArgs.requiredSubgroupSize() != 0) {
    // VK_EXT_subgroup_size_control is not currently enabled at device
    // creation; honoring this would require recreating the pipeline with
    // VkPipelineShaderStageRequiredSubgroupSizeCreateInfo.
    throw ghost::unsupported_error();
  }

  uint32_t gx = launchArgs.dims() >= 1
                    ? narrowDim(launchArgs.count(0), "global_size[0] / count")
                    : 1;
  uint32_t gy = launchArgs.dims() >= 2
                    ? narrowDim(launchArgs.count(1), "global_size[1] / count")
                    : 1;
  uint32_t gz = launchArgs.dims() >= 3
                    ? narrowDim(launchArgs.count(2), "global_size[2] / count")
                    : 1;
  vkCmdDispatch(stream.commandBuffer, gx, gy, gz);

  // Memory barrier for compute shader writes.
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
      return Attribute((int32_t)preferredSubgroupSize());
    case kFunctionThreadWidth:
      return Attribute((int32_t)preferredSubgroupSize());
    default:
      return Attribute();
  }
}

uint32_t FunctionVulkan::preferredSubgroupSize() const {
  // VkPhysicalDeviceSubgroupProperties is core in Vulkan 1.1; the device is
  // created with VK_API_VERSION_1_2 so vkGetPhysicalDeviceProperties2 is
  // available.
  VkPhysicalDeviceSubgroupProperties subgroupProps = {};
  subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  VkPhysicalDeviceProperties2 props2 = {};
  props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  props2.pNext = &subgroupProps;
  vkGetPhysicalDeviceProperties2(_dev.physicalDevice, &props2);
  return subgroupProps.subgroupSize ? subgroupProps.subgroupSize : 32u;
}

// ---------------------------------------------------------------------------
// LibraryVulkan
// ---------------------------------------------------------------------------

LibraryVulkan::LibraryVulkan(const DeviceVulkan& dev, bool retainBinary)
    : Library(retainBinary), _dev(dev), _module(dev.device) {}

// ~LibraryVulkan: implicit. _module's vk::ptr destructor unloads the
// shader module.

void LibraryVulkan::loadFromCache(const void* data, size_t length,
                                  const CompilerOptions& options) {
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
        ghost::vk::reflectSpirv(binaries[0].data(), binaries[0].size(),
                                _reflection);
        return;
      }
      _module.reset();
    }
  }
}

void LibraryVulkan::saveToCache(const void* data, size_t length,
                                const CompilerOptions& options) const {
  auto& cache = Device::binaryCache();
  if (!cache.isEnabled()) return;

  std::vector<unsigned char*> binaries = {
      const_cast<unsigned char*>(static_cast<const unsigned char*>(data))};
  std::vector<size_t> sizes = {length};
  cache.saveBinaries(_dev, binaries, sizes, data, length, options);
}

void LibraryVulkan::loadFromData(const void* data, size_t len,
                                 const CompilerOptions& options) {
  // Try cache first
  loadFromCache(data, len, options);
  if ((VkShaderModule)_module != VK_NULL_HANDLE) return;

  // SPIR-V data must be uint32-aligned
  VkShaderModuleCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = len;
  createInfo.pCode = reinterpret_cast<const uint32_t*>(data);

  checkError(vkCreateShaderModule(_dev.device, &createInfo, nullptr, &_module));

  // Reflect the SPIR-V binary now so lookupFunction can build a matching
  // pipeline layout without re-parsing.
  ghost::vk::reflectSpirv(data, len, _reflection);

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
  return ghost::Function(
      std::make_shared<FunctionVulkan>(_dev, _module, name, _reflection));
}

ghost::Function LibraryVulkan::specializeFunction(
    const std::string& name, const std::vector<Attribute>& args) const {
  return ghost::Function(
      std::make_shared<FunctionVulkan>(_dev, _module, name, _reflection, args));
}

std::vector<uint8_t> LibraryVulkan::getBinary() const { return _spirvData; }

}  // namespace implementation
}  // namespace ghost

#endif
