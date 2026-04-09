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

#include <ghost/exception.h>
#include <ghost/vulkan/device.h>
#include <ghost/vulkan/exception.h>
#include <ghost/vulkan/impl_device.h>
#include <ghost/vulkan/impl_function.h>

#include <algorithm>
#include <cstring>

namespace ghost {
namespace implementation {
using namespace vk;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

VkImageType getImageType(const ImageDescription& descr) {
  if (descr.size.z > 1) return VK_IMAGE_TYPE_3D;
  if (descr.size.y > 1) return VK_IMAGE_TYPE_2D;
  return VK_IMAGE_TYPE_1D;
}

VkImageViewType getImageViewType(const ImageDescription& descr) {
  if (descr.size.z > 1) return VK_IMAGE_VIEW_TYPE_3D;
  if (descr.size.y > 1) return VK_IMAGE_VIEW_TYPE_2D;
  return VK_IMAGE_VIEW_TYPE_1D;
}

VkAccessFlags getAccessFlags(Access access) {
  switch (access) {
    case Access::ReadOnly:
      return VK_ACCESS_SHADER_READ_BIT;
    case Access::WriteOnly:
      return VK_ACCESS_SHADER_WRITE_BIT;
    default:
      return VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// EventVulkan
// ---------------------------------------------------------------------------

EventVulkan::EventVulkan(VkDevice dev, VkFence f, bool owns)
    : device(dev), fence(f), ownsHandle(owns) {}

EventVulkan::~EventVulkan() {
  if (ownsHandle && fence != VK_NULL_HANDLE) {
    vkDestroyFence(device, fence, nullptr);
  }
}

void EventVulkan::wait() {
  vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
}

bool EventVulkan::isComplete() const {
  return vkGetFenceStatus(device, fence) == VK_SUCCESS;
}

double EventVulkan::elapsed(const Event& other) const {
  // Vulkan fences don't carry timestamp data by default.
  // Full profiling requires timestamp queries.
  return 0.0;
}

// ---------------------------------------------------------------------------
// StreamVulkan
// ---------------------------------------------------------------------------

StreamVulkan::StreamVulkan(const DeviceVulkan& dev_)
    : dev(dev_),
      commandPool(VK_NULL_HANDLE),
      commandBuffer(VK_NULL_HANDLE),
      fence(VK_NULL_HANDLE),
      recording(false),
      submitted(false) {
  // Create command pool
  VkCommandPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = dev.computeQueueFamily;
  checkError(vkCreateCommandPool(dev.device, &poolInfo, nullptr, &commandPool));

  // Allocate command buffer
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;
  checkError(vkAllocateCommandBuffers(dev.device, &allocInfo, &commandBuffer));

  // Create fence
  VkFenceCreateInfo fenceInfo = {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  checkError(vkCreateFence(dev.device, &fenceInfo, nullptr, &fence));
}

StreamVulkan::~StreamVulkan() {
  if (recording || submitted) {
    try {
      sync();
    } catch (...) {
    }
  }
  cleanupStaging();
  if (fence != VK_NULL_HANDLE) vkDestroyFence(dev.device, fence, nullptr);
  if (commandPool != VK_NULL_HANDLE)
    vkDestroyCommandPool(dev.device, commandPool, nullptr);
}

void StreamVulkan::begin() {
  if (recording) return;

  // Wait for any previous submission to complete
  vkWaitForFences(dev.device, 1, &fence, VK_TRUE, UINT64_MAX);
  vkResetFences(dev.device, 1, &fence);

  // Process deferred reads from previous batch
  for (auto& dr : deferredReads) {
    void* mapped = nullptr;
    vkMapMemory(dev.device, dr.stagingMemory, dr.offset, dr.size, 0, &mapped);
    if (mapped) {
      memcpy(dr.dstPtr, mapped, dr.size);
      vkUnmapMemory(dev.device, dr.stagingMemory);
    }
  }
  deferredReads.clear();

  cleanupStaging();

  vkResetCommandBuffer(commandBuffer, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  checkError(vkBeginCommandBuffer(commandBuffer, &beginInfo));

  recording = true;
  submitted = false;
}

void StreamVulkan::submit() {
  if (!recording) return;

  checkError(vkEndCommandBuffer(commandBuffer));

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  checkError(vkQueueSubmit(dev.computeQueue, 1, &submitInfo, fence));

  recording = false;
  submitted = true;
}

void StreamVulkan::sync() {
  if (recording) submit();
  if (submitted) {
    vkWaitForFences(dev.device, 1, &fence, VK_TRUE, UINT64_MAX);
    submitted = false;

    // Process deferred reads
    for (auto& dr : deferredReads) {
      void* mapped = nullptr;
      vkMapMemory(dev.device, dr.stagingMemory, dr.offset, dr.size, 0, &mapped);
      if (mapped) {
        memcpy(dr.dstPtr, mapped, dr.size);
        vkUnmapMemory(dev.device, dr.stagingMemory);
      }
    }
    deferredReads.clear();
    cleanupStaging();
  }
}

std::shared_ptr<Event> StreamVulkan::record() {
  if (recording) submit();

  // The current fence represents the completion point
  VkFenceCreateInfo fenceInfo = {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence eventFence;
  checkError(vkCreateFence(dev.device, &fenceInfo, nullptr, &eventFence));

  // Submit a no-op to signal the event fence
  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  checkError(vkQueueSubmit(dev.computeQueue, 1, &submitInfo, eventFence));

  return std::make_shared<EventVulkan>(dev.device, eventFence, true);
}

void StreamVulkan::waitForEvent(const std::shared_ptr<Event>& e) {
  auto* evt = static_cast<EventVulkan*>(e.get());
  // Insert a pipeline barrier - the fence will have completed by the time
  // GPU processes subsequent commands after QueueWaitIdle.
  // For proper cross-queue sync, semaphores would be needed.
  evt->wait();
}

void StreamVulkan::cleanupStaging() {
  for (auto& res : pendingStaging) {
    vkDestroyBuffer(dev.device, res.buffer, nullptr);
    vkFreeMemory(dev.device, res.memory, nullptr);
  }
  pendingStaging.clear();

  // Clean up deferred read staging too
  for (auto& dr : deferredReads) {
    vkDestroyBuffer(dev.device, dr.stagingBuffer, nullptr);
    vkFreeMemory(dev.device, dr.stagingMemory, nullptr);
  }
}

void StreamVulkan::addStagingResource(VkBuffer buf, VkDeviceMemory mem) {
  pendingStaging.push_back({buf, mem});
}

// ---------------------------------------------------------------------------
// BufferVulkan
// ---------------------------------------------------------------------------

BufferVulkan::BufferVulkan(const DeviceVulkan& dev_, size_t bytes,
                           const BufferOptions& opts)
    : dev(dev_),
      deviceAlive(dev_.alive),
      buffer(VK_NULL_HANDLE),
      memory(VK_NULL_HANDLE),
      _size(bytes),
      ownsHandles(true) {
  VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  // Staging hint routes to host-visible memory. Readback (kernel writes,
  // host reads) benefits from HOST_CACHED for fast host-side reads.
  VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  if (opts.hint == AllocHint::Staging) {
    memProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (opts.access == Access::WriteOnly) {
      memProps |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    }
  }
  dev.createBuffer(bytes, usage, memProps, buffer, memory);
}

BufferVulkan::BufferVulkan(const DeviceVulkan& dev_, VkBuffer buf,
                           VkDeviceMemory mem, size_t bytes, bool owns)
    : dev(dev_),
      deviceAlive(dev_.alive),
      buffer(buf),
      memory(mem),
      _size(bytes),
      ownsHandles(owns) {}

BufferVulkan::~BufferVulkan() {
  if (ownsHandles && *deviceAlive) {
    if (buffer != VK_NULL_HANDLE) vkDestroyBuffer(dev.device, buffer, nullptr);
    if (memory != VK_NULL_HANDLE) vkFreeMemory(dev.device, memory, nullptr);
  }
}

size_t BufferVulkan::size() const { return _size; }

void BufferVulkan::copy(const ghost::Stream& s, const ghost::Buffer& src,
                        size_t bytes) {
  copy(s, src, 0, 0, bytes);
}

void BufferVulkan::copy(const ghost::Stream& s, const void* src, size_t bytes) {
  copy(s, src, 0, bytes);
}

void BufferVulkan::copyTo(const ghost::Stream& s, void* dst,
                          size_t bytes) const {
  copyTo(s, dst, 0, bytes);
}

void BufferVulkan::copy(const ghost::Stream& s, const ghost::Buffer& src,
                        size_t srcOffset, size_t dstOffset, size_t bytes) {
  auto& stream = *static_cast<StreamVulkan*>(s.impl().get());
  auto* srcBuf = static_cast<BufferVulkan*>(src.impl().get());

  stream.begin();

  VkBufferCopy region = {};
  region.srcOffset = srcOffset + srcBuf->baseOffset();
  region.dstOffset = dstOffset + baseOffset();
  region.size = bytes;
  vkCmdCopyBuffer(stream.commandBuffer, srcBuf->buffer, buffer, 1, &region);

  // Barrier to ensure copy completes before next operation
  VkMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(
      stream.commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
      1, &barrier, 0, nullptr, 0, nullptr);
}

void BufferVulkan::copy(const ghost::Stream& s, const void* src,
                        size_t dstOffset, size_t bytes) {
  auto& stream = *static_cast<StreamVulkan*>(s.impl().get());

  // Create staging buffer
  VkBuffer staging;
  VkDeviceMemory stagingMem;
  dev.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   staging, stagingMem);

  // Copy data to staging buffer
  void* mapped;
  checkError(vkMapMemory(dev.device, stagingMem, 0, bytes, 0, &mapped));
  memcpy(mapped, src, bytes);
  vkUnmapMemory(dev.device, stagingMem);

  stream.begin();

  // Copy staging to device buffer
  VkBufferCopy region = {};
  region.srcOffset = 0;
  region.dstOffset = dstOffset + baseOffset();
  region.size = bytes;
  vkCmdCopyBuffer(stream.commandBuffer, staging, buffer, 1, &region);

  VkMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(
      stream.commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
      1, &barrier, 0, nullptr, 0, nullptr);

  stream.addStagingResource(staging, stagingMem);
}

void BufferVulkan::copyTo(const ghost::Stream& s, void* dst, size_t srcOffset,
                          size_t bytes) const {
  auto& stream = *static_cast<StreamVulkan*>(s.impl().get());

  // Create staging buffer for readback
  VkBuffer staging;
  VkDeviceMemory stagingMem;
  dev.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   staging, stagingMem);

  stream.begin();

  VkBufferCopy region = {};
  region.srcOffset = srcOffset + baseOffset();
  region.dstOffset = 0;
  region.size = bytes;
  vkCmdCopyBuffer(stream.commandBuffer, buffer, staging, 1, &region);

  VkMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(stream.commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr,
                       0, nullptr);

  // Defer the actual host read until sync
  StreamVulkan::DeferredRead dr;
  dr.stagingBuffer = staging;
  dr.stagingMemory = stagingMem;
  dr.dstPtr = dst;
  dr.offset = 0;
  dr.size = bytes;
  stream.deferredReads.push_back(dr);
}

void BufferVulkan::fill(const ghost::Stream& s, size_t offset, size_t sz,
                        uint8_t value) {
  auto& stream = *static_cast<StreamVulkan*>(s.impl().get());
  stream.begin();

  // vkCmdFillBuffer takes a uint32_t data value
  uint32_t data32 = (uint32_t)value | ((uint32_t)value << 8) |
                    ((uint32_t)value << 16) | ((uint32_t)value << 24);

  // Align offset and size to 4 bytes for vkCmdFillBuffer
  VkDeviceSize alignedOffset = (offset + baseOffset()) & ~3ULL;
  VkDeviceSize alignedSize = (sz + 3) & ~3ULL;

  vkCmdFillBuffer(stream.commandBuffer, buffer, alignedOffset, alignedSize,
                  data32);

  VkMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(
      stream.commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
      1, &barrier, 0, nullptr, 0, nullptr);
}

void BufferVulkan::fill(const ghost::Stream& s, size_t offset, size_t sz,
                        const void* pattern, size_t patternSize) {
  if (patternSize == 1) {
    fill(s, offset, sz, *static_cast<const uint8_t*>(pattern));
    return;
  }

  // For arbitrary patterns, create a staging buffer with the pattern repeated
  std::vector<uint8_t> fillData(sz);
  for (size_t i = 0; i < sz; i += patternSize) {
    size_t n = std::min(patternSize, sz - i);
    memcpy(fillData.data() + i, pattern, n);
  }
  copy(s, fillData.data(), offset, sz);
}

std::shared_ptr<Buffer> BufferVulkan::createSubBuffer(
    const std::shared_ptr<Buffer>& self, size_t offset, size_t sz) {
  return std::make_shared<SubBufferVulkan>(self, dev, buffer,
                                           baseOffset() + offset, sz);
}

// ---------------------------------------------------------------------------
// SubBufferVulkan
// ---------------------------------------------------------------------------

SubBufferVulkan::SubBufferVulkan(std::shared_ptr<Buffer> parent,
                                 const DeviceVulkan& dev_, VkBuffer buf,
                                 size_t offset, size_t bytes)
    : BufferVulkan(dev_, buf, VK_NULL_HANDLE, bytes, false),
      _parent(parent),
      _offset(offset) {}

size_t SubBufferVulkan::baseOffset() const {
  return _offset + static_cast<BufferVulkan*>(_parent.get())->baseOffset();
}

void SubBufferVulkan::copy(const ghost::Stream& s, const ghost::Buffer& src,
                           size_t bytes) {
  BufferVulkan::copy(s, src, 0, 0, bytes);
}

void SubBufferVulkan::copy(const ghost::Stream& s, const void* src,
                           size_t bytes) {
  BufferVulkan::copy(s, src, 0, bytes);
}

void SubBufferVulkan::copyTo(const ghost::Stream& s, void* dst,
                             size_t bytes) const {
  BufferVulkan::copyTo(s, dst, 0, bytes);
}

// ---------------------------------------------------------------------------
// MappedBufferVulkan
// ---------------------------------------------------------------------------

MappedBufferVulkan::MappedBufferVulkan(const DeviceVulkan& dev_, size_t bytes,
                                       const BufferOptions& opts)
    : BufferVulkan(dev_, VK_NULL_HANDLE, VK_NULL_HANDLE, bytes, true),
      mappedPtr(nullptr) {
  VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  // For readback-dominant staging, prefer HOST_CACHED for faster host reads.
  if (opts.access == Access::WriteOnly) {
    memProps |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  }
  dev.createBuffer(bytes, usage, memProps, buffer, memory);

  // Persistently map the buffer
  checkError(vkMapMemory(dev.device, memory, 0, bytes, 0, &mappedPtr));
}

void* MappedBufferVulkan::map(const ghost::Stream& s, Access access,
                              bool doSync) {
  if (doSync) {
    auto& stream = *static_cast<StreamVulkan*>(s.impl().get());
    stream.sync();
  }
  return mappedPtr;
}

void MappedBufferVulkan::unmap(const ghost::Stream& s) {
  // Persistently mapped - no-op
}

// ---------------------------------------------------------------------------
// ImageVulkan
// ---------------------------------------------------------------------------

ImageVulkan::ImageVulkan(const DeviceVulkan& dev_, const ImageDescription& d)
    : dev(dev_),
      deviceAlive(dev_.alive),
      image(VK_NULL_HANDLE),
      memory(VK_NULL_HANDLE),
      imageView(VK_NULL_HANDLE),
      descr(d),
      ownsHandles(true) {
  VkFormat format = dev.getImageFormat(d);

  VkImageCreateInfo imageInfo = {};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = getImageType(d);
  imageInfo.format = format;
  imageInfo.extent.width = (uint32_t)d.size.x;
  imageInfo.extent.height = std::max((uint32_t)d.size.y, 1u);
  imageInfo.extent.depth = std::max((uint32_t)d.size.z, 1u);
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  checkError(vkCreateImage(dev.device, &imageInfo, nullptr, &image));

  // Allocate memory
  VkMemoryRequirements memReqs;
  vkGetImageMemoryRequirements(dev.device, image, &memReqs);

  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memReqs.size;
  allocInfo.memoryTypeIndex = dev.findMemoryType(
      memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  checkError(vkAllocateMemory(dev.device, &allocInfo, nullptr, &memory));
  checkError(vkBindImageMemory(dev.device, image, memory, 0));

  // Create image view
  VkImageViewCreateInfo viewInfo = {};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = image;
  viewInfo.viewType = getImageViewType(d);
  viewInfo.format = format;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;

  checkError(vkCreateImageView(dev.device, &viewInfo, nullptr, &imageView));
}

ImageVulkan::ImageVulkan(const DeviceVulkan& dev_, const ImageDescription& d,
                         BufferVulkan& buf)
    : dev(dev_),
      deviceAlive(dev_.alive),
      image(VK_NULL_HANDLE),
      memory(VK_NULL_HANDLE),
      imageView(VK_NULL_HANDLE),
      descr(d),
      ownsHandles(true) {
  VkFormat format = dev.getImageFormat(d);

  // Create image backed by buffer memory using VkBufferImageCopy pattern.
  // Vulkan doesn't support aliasing buffer memory to an image directly
  // like Metal/OpenCL do, so we create a separate image and note that
  // copies between them are the expected use pattern.
  VkImageCreateInfo imageInfo = {};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = getImageType(d);
  imageInfo.format = format;
  imageInfo.extent.width = (uint32_t)d.size.x;
  imageInfo.extent.height = std::max((uint32_t)d.size.y, 1u);
  imageInfo.extent.depth = std::max((uint32_t)d.size.z, 1u);
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  checkError(vkCreateImage(dev.device, &imageInfo, nullptr, &image));

  VkMemoryRequirements memReqs;
  vkGetImageMemoryRequirements(dev.device, image, &memReqs);

  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memReqs.size;
  allocInfo.memoryTypeIndex = dev.findMemoryType(
      memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  checkError(vkAllocateMemory(dev.device, &allocInfo, nullptr, &memory));
  checkError(vkBindImageMemory(dev.device, image, memory, 0));

  VkImageViewCreateInfo viewInfo = {};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = image;
  viewInfo.viewType = getImageViewType(d);
  viewInfo.format = format;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;

  checkError(vkCreateImageView(dev.device, &viewInfo, nullptr, &imageView));
}

ImageVulkan::ImageVulkan(const DeviceVulkan& dev_, const ImageDescription& d,
                         ImageVulkan& other)
    : dev(dev_),
      deviceAlive(dev_.alive),
      image(other.image),
      memory(VK_NULL_HANDLE),
      imageView(VK_NULL_HANDLE),
      descr(d),
      ownsHandles(false) {
  // Create a new view into the same image
  VkFormat format = dev.getImageFormat(d);

  VkImageViewCreateInfo viewInfo = {};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = image;
  viewInfo.viewType = getImageViewType(d);
  viewInfo.format = format;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;

  checkError(vkCreateImageView(dev.device, &viewInfo, nullptr, &imageView));
}

ImageVulkan::~ImageVulkan() {
  if (!*deviceAlive) return;
  if (imageView != VK_NULL_HANDLE)
    vkDestroyImageView(dev.device, imageView, nullptr);
  if (ownsHandles) {
    if (image != VK_NULL_HANDLE) vkDestroyImage(dev.device, image, nullptr);
    if (memory != VK_NULL_HANDLE) vkFreeMemory(dev.device, memory, nullptr);
  }
}

void ImageVulkan::copy(const ghost::Stream& s, const ghost::Image& src) {
  auto& stream = *static_cast<StreamVulkan*>(s.impl().get());
  auto* srcImg = static_cast<ImageVulkan*>(src.impl().get());

  stream.begin();

  // Transition layouts
  VkImageMemoryBarrier srcBarrier = {};
  srcBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  srcBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  srcBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  srcBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  srcBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  srcBarrier.image = srcImg->image;
  srcBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  srcBarrier.subresourceRange.levelCount = 1;
  srcBarrier.subresourceRange.layerCount = 1;

  VkImageMemoryBarrier dstBarrier = {};
  dstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  dstBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  dstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  dstBarrier.srcAccessMask = 0;
  dstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  dstBarrier.image = image;
  dstBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  dstBarrier.subresourceRange.levelCount = 1;
  dstBarrier.subresourceRange.layerCount = 1;

  VkImageMemoryBarrier barriers[] = {srcBarrier, dstBarrier};
  vkCmdPipelineBarrier(
      stream.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);

  VkImageCopy region = {};
  region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.srcSubresource.layerCount = 1;
  region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.dstSubresource.layerCount = 1;
  region.extent.width = (uint32_t)descr.size.x;
  region.extent.height = std::max((uint32_t)descr.size.y, 1u);
  region.extent.depth = std::max((uint32_t)descr.size.z, 1u);

  vkCmdCopyImage(stream.commandBuffer, srcImg->image,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  // Transition both images back to general layout
  VkImageMemoryBarrier postBarriers[2] = {};
  postBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  postBarriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  postBarriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
  postBarriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  postBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  postBarriers[0].image = image;
  postBarriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  postBarriers[0].subresourceRange.levelCount = 1;
  postBarriers[0].subresourceRange.layerCount = 1;

  postBarriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  postBarriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  postBarriers[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
  postBarriers[1].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  postBarriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  postBarriers[1].image = srcImg->image;
  postBarriers[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  postBarriers[1].subresourceRange.levelCount = 1;
  postBarriers[1].subresourceRange.layerCount = 1;

  vkCmdPipelineBarrier(stream.commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 2, postBarriers);
}

void ImageVulkan::copy(const ghost::Stream& s, const ghost::Buffer& src,
                       const BufferLayout& layout) {
  auto& stream = *static_cast<StreamVulkan*>(s.impl().get());
  auto* srcBuf = static_cast<BufferVulkan*>(src.impl().get());

  stream.begin();

  VkImageMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.srcAccessMask = 0;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.layerCount = 1;

  vkCmdPipelineBarrier(stream.commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);

  VkBufferImageCopy region = {};
  region.bufferOffset = srcBuf->baseOffset();
  region.bufferRowLength =
      layout.stride.x > 0 ? (uint32_t)(layout.stride.x / descr.pixelSize()) : 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.layerCount = 1;
  region.imageExtent.width = (uint32_t)layout.size.x;
  region.imageExtent.height = std::max((uint32_t)layout.size.y, 1u);
  region.imageExtent.depth = std::max((uint32_t)layout.size.z, 1u);

  vkCmdCopyBufferToImage(stream.commandBuffer, srcBuf->buffer, image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  vkCmdPipelineBarrier(stream.commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);
}

void ImageVulkan::copy(const ghost::Stream& s, const void* src,
                       const BufferLayout& layout) {
  auto& stream = *static_cast<StreamVulkan*>(s.impl().get());

  size_t dataSize = descr.dataSize();

  // Create staging buffer
  VkBuffer staging;
  VkDeviceMemory stagingMem;
  dev.createBuffer(dataSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   staging, stagingMem);

  void* mapped;
  checkError(vkMapMemory(dev.device, stagingMem, 0, dataSize, 0, &mapped));
  memcpy(mapped, src, dataSize);
  vkUnmapMemory(dev.device, stagingMem);

  stream.begin();

  VkImageMemoryBarrier imgBarrier = {};
  imgBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  imgBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imgBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  imgBarrier.srcAccessMask = 0;
  imgBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  imgBarrier.image = image;
  imgBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  imgBarrier.subresourceRange.levelCount = 1;
  imgBarrier.subresourceRange.layerCount = 1;

  vkCmdPipelineBarrier(stream.commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &imgBarrier);

  VkBufferImageCopy region = {};
  region.bufferOffset = 0;
  region.bufferRowLength =
      layout.stride.x > 0 ? (uint32_t)(layout.stride.x / descr.pixelSize()) : 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.layerCount = 1;
  region.imageExtent.width = (uint32_t)layout.size.x;
  region.imageExtent.height = std::max((uint32_t)layout.size.y, 1u);
  region.imageExtent.depth = std::max((uint32_t)layout.size.z, 1u);

  vkCmdCopyBufferToImage(stream.commandBuffer, staging, image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  imgBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  imgBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  imgBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  imgBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  vkCmdPipelineBarrier(stream.commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &imgBarrier);

  stream.addStagingResource(staging, stagingMem);
}

void ImageVulkan::copyTo(const ghost::Stream& s, ghost::Buffer& dst,
                         const BufferLayout& layout) const {
  auto& stream = *static_cast<StreamVulkan*>(s.impl().get());
  auto* dstBuf = static_cast<BufferVulkan*>(dst.impl().get());

  stream.begin();

  VkImageMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.layerCount = 1;

  vkCmdPipelineBarrier(
      stream.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

  VkBufferImageCopy region = {};
  region.bufferOffset = dstBuf->baseOffset();
  region.bufferRowLength =
      layout.stride.x > 0 ? (uint32_t)(layout.stride.x / descr.pixelSize()) : 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.layerCount = 1;
  region.imageExtent.width = (uint32_t)layout.size.x;
  region.imageExtent.height = std::max((uint32_t)layout.size.y, 1u);
  region.imageExtent.depth = std::max((uint32_t)layout.size.z, 1u);

  vkCmdCopyImageToBuffer(stream.commandBuffer, image,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstBuf->buffer,
                         1, &region);

  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  vkCmdPipelineBarrier(stream.commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);
}

void ImageVulkan::copyTo(const ghost::Stream& s, void* dst,
                         const BufferLayout& layout) const {
  auto& stream = *static_cast<StreamVulkan*>(s.impl().get());

  size_t dataSize = descr.dataSize();

  // Create staging buffer for readback
  VkBuffer staging;
  VkDeviceMemory stagingMem;
  dev.createBuffer(dataSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   staging, stagingMem);

  stream.begin();

  VkImageMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.layerCount = 1;

  vkCmdPipelineBarrier(
      stream.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

  VkBufferImageCopy region = {};
  region.bufferOffset = 0;
  region.bufferRowLength =
      layout.stride.x > 0 ? (uint32_t)(layout.stride.x / descr.pixelSize()) : 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.layerCount = 1;
  region.imageExtent.width = (uint32_t)layout.size.x;
  region.imageExtent.height = std::max((uint32_t)layout.size.y, 1u);
  region.imageExtent.depth = std::max((uint32_t)layout.size.z, 1u);

  vkCmdCopyImageToBuffer(stream.commandBuffer, image,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1,
                         &region);

  // Transition image back to general layout
  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  vkCmdPipelineBarrier(stream.commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);

  // Ensure staging buffer write is visible to host
  VkMemoryBarrier memBarrier = {};
  memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(stream.commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &memBarrier, 0,
                       nullptr, 0, nullptr);

  StreamVulkan::DeferredRead dr;
  dr.stagingBuffer = staging;
  dr.stagingMemory = stagingMem;
  dr.dstPtr = dst;
  dr.offset = 0;
  dr.size = dataSize;
  stream.deferredReads.push_back(dr);
}

void ImageVulkan::copy(const ghost::Stream& s, const ghost::Buffer& src,
                       const BufferLayout& layout, const Origin3& imageOrigin) {
  auto& stream = *static_cast<StreamVulkan*>(s.impl().get());
  auto* srcBuf = static_cast<BufferVulkan*>(src.impl().get());

  stream.begin();

  VkImageMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.srcAccessMask = 0;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.layerCount = 1;

  vkCmdPipelineBarrier(stream.commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);

  VkBufferImageCopy region = {};
  region.bufferOffset = srcBuf->baseOffset();
  region.bufferRowLength =
      layout.stride.x > 0 ? (uint32_t)(layout.stride.x / descr.pixelSize()) : 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.layerCount = 1;
  region.imageOffset.x = (int32_t)imageOrigin.x;
  region.imageOffset.y = (int32_t)imageOrigin.y;
  region.imageOffset.z = (int32_t)imageOrigin.z;
  region.imageExtent.width = (uint32_t)layout.size.x;
  region.imageExtent.height = std::max((uint32_t)layout.size.y, 1u);
  region.imageExtent.depth = std::max((uint32_t)layout.size.z, 1u);

  vkCmdCopyBufferToImage(stream.commandBuffer, srcBuf->buffer, image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  vkCmdPipelineBarrier(stream.commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);
}

void ImageVulkan::copyTo(const ghost::Stream& s, ghost::Buffer& dst,
                         const BufferLayout& layout,
                         const Origin3& imageOrigin) const {
  auto& stream = *static_cast<StreamVulkan*>(s.impl().get());
  auto* dstBuf = static_cast<BufferVulkan*>(dst.impl().get());

  stream.begin();

  VkImageMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.layerCount = 1;

  vkCmdPipelineBarrier(
      stream.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

  VkBufferImageCopy region = {};
  region.bufferOffset = dstBuf->baseOffset();
  region.bufferRowLength =
      layout.stride.x > 0 ? (uint32_t)(layout.stride.x / descr.pixelSize()) : 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.layerCount = 1;
  region.imageOffset.x = (int32_t)imageOrigin.x;
  region.imageOffset.y = (int32_t)imageOrigin.y;
  region.imageOffset.z = (int32_t)imageOrigin.z;
  region.imageExtent.width = (uint32_t)layout.size.x;
  region.imageExtent.height = std::max((uint32_t)layout.size.y, 1u);
  region.imageExtent.depth = std::max((uint32_t)layout.size.z, 1u);

  vkCmdCopyImageToBuffer(stream.commandBuffer, image,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstBuf->buffer,
                         1, &region);

  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  vkCmdPipelineBarrier(stream.commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);
}

void ImageVulkan::copy(const ghost::Stream& s, const ghost::Image& src,
                       const Size3& region, const Origin3& srcOrigin,
                       const Origin3& dstOrigin) {
  auto& stream = *static_cast<StreamVulkan*>(s.impl().get());
  auto* srcImg = static_cast<ImageVulkan*>(src.impl().get());

  stream.begin();

  // Transition layouts
  VkImageMemoryBarrier srcBarrier = {};
  srcBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  srcBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  srcBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  srcBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  srcBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  srcBarrier.image = srcImg->image;
  srcBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  srcBarrier.subresourceRange.levelCount = 1;
  srcBarrier.subresourceRange.layerCount = 1;

  VkImageMemoryBarrier dstBarrier = {};
  dstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  dstBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  dstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  dstBarrier.srcAccessMask = 0;
  dstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  dstBarrier.image = image;
  dstBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  dstBarrier.subresourceRange.levelCount = 1;
  dstBarrier.subresourceRange.layerCount = 1;

  VkImageMemoryBarrier barriers[] = {srcBarrier, dstBarrier};
  vkCmdPipelineBarrier(
      stream.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);

  VkImageCopy copyRegion = {};
  copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copyRegion.srcSubresource.layerCount = 1;
  copyRegion.srcOffset = {(int32_t)srcOrigin.x, (int32_t)srcOrigin.y,
                          (int32_t)srcOrigin.z};
  copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copyRegion.dstSubresource.layerCount = 1;
  copyRegion.dstOffset = {(int32_t)dstOrigin.x, (int32_t)dstOrigin.y,
                          (int32_t)dstOrigin.z};
  copyRegion.extent.width = (uint32_t)region.x;
  copyRegion.extent.height = std::max((uint32_t)region.y, 1u);
  copyRegion.extent.depth = std::max((uint32_t)region.z, 1u);

  vkCmdCopyImage(stream.commandBuffer, srcImg->image,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

  // Transition both images back to general layout
  VkImageMemoryBarrier postBarriers[2] = {};
  postBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  postBarriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  postBarriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
  postBarriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  postBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  postBarriers[0].image = image;
  postBarriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  postBarriers[0].subresourceRange.levelCount = 1;
  postBarriers[0].subresourceRange.layerCount = 1;

  postBarriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  postBarriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  postBarriers[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
  postBarriers[1].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  postBarriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  postBarriers[1].image = srcImg->image;
  postBarriers[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  postBarriers[1].subresourceRange.levelCount = 1;
  postBarriers[1].subresourceRange.layerCount = 1;

  vkCmdPipelineBarrier(stream.commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 2, postBarriers);
}

// ---------------------------------------------------------------------------
// DeviceVulkan (implementation)
// ---------------------------------------------------------------------------

DeviceVulkan::DeviceVulkan(const SharedContext& share)
    : instance(VK_NULL_HANDLE),
      physicalDevice(VK_NULL_HANDLE),
      device(VK_NULL_HANDLE),
      computeQueue(VK_NULL_HANDLE),
      computeQueueFamily(0),
      descriptorPool(VK_NULL_HANDLE),
      ownsInstance(false),
      alive(std::make_shared<bool>(true)) {
  if (share.context) {
    // Reuse existing Vulkan objects
    instance = static_cast<VkInstance>(share.context);
    device = static_cast<VkDevice>(share.device);
    computeQueue = static_cast<VkQueue>(share.queue);
    physicalDevice = static_cast<VkPhysicalDevice>(share.platform);
  } else {
    // Create new instance
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Ghost";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "Ghost";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    checkError(vkCreateInstance(&createInfo, nullptr, &instance));
    ownsInstance = true;

    // Pick physical device (first discrete GPU, or first available)
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0)
      throw std::runtime_error("No Vulkan-capable GPU found");

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    physicalDevice = devices[0];
    for (auto& pd : devices) {
      VkPhysicalDeviceProperties props;
      vkGetPhysicalDeviceProperties(pd, &props);
      if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        physicalDevice = pd;
        break;
      }
    }

    // Find compute queue family
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                             nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                             queueFamilies.data());

    bool found = false;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
      if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        computeQueueFamily = i;
        found = true;
        // Prefer a compute-only queue family (not graphics)
        if (!(queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) break;
      }
    }
    if (!found) throw std::runtime_error("No compute queue family found");

    // Create logical device
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

    checkError(
        vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));

    vkGetDeviceQueue(device, computeQueueFamily, 0, &computeQueue);
  }

  // Cache device properties
  vkGetPhysicalDeviceProperties(physicalDevice, &properties);
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

  // Create descriptor pool
  VkDescriptorPoolSize poolSizes[] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024},
  };

  VkDescriptorPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  poolInfo.maxSets = 4096;
  poolInfo.poolSizeCount = 2;
  poolInfo.pPoolSizes = poolSizes;

  checkError(
      vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));
}

DeviceVulkan::DeviceVulkan(const GpuInfo& info)
    : instance(VK_NULL_HANDLE),
      physicalDevice(VK_NULL_HANDLE),
      device(VK_NULL_HANDLE),
      computeQueue(VK_NULL_HANDLE),
      computeQueueFamily(0),
      descriptorPool(VK_NULL_HANDLE),
      ownsInstance(true),
      alive(std::make_shared<bool>(true)) {
  // Create new instance
  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Ghost";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "Ghost";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  checkError(vkCreateInstance(&createInfo, nullptr, &instance));

  // Pick physical device by index
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

  if (info.index < 0 || info.index >= (int)deviceCount) {
    vkDestroyInstance(instance, nullptr);
    instance = VK_NULL_HANDLE;
    ownsInstance = false;
    throw std::runtime_error("Invalid Vulkan device index");
  }
  physicalDevice = devices[info.index];

  // Find compute queue family
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                           nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                           queueFamilies.data());

  bool found = false;
  for (uint32_t i = 0; i < queueFamilyCount; i++) {
    if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      computeQueueFamily = i;
      found = true;
      if (!(queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) break;
    }
  }
  if (!found) throw std::runtime_error("No compute queue family found");

  // Create logical device
  float queuePriority = 1.0f;
  VkDeviceQueueCreateInfo queueCreateInfo = {};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = computeQueueFamily;
  queueCreateInfo.queueCount = 1;
  queueCreateInfo.pQueuePriorities = &queuePriority;

  VkDeviceCreateInfo deviceCreateInfo = {};
  deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCreateInfo.queueCreateInfoCount = 1;
  deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

  checkError(
      vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));

  vkGetDeviceQueue(device, computeQueueFamily, 0, &computeQueue);

  // Cache device properties
  vkGetPhysicalDeviceProperties(physicalDevice, &properties);
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

  // Create descriptor pool
  VkDescriptorPoolSize poolSizes[] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024},
  };

  VkDescriptorPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  poolInfo.maxSets = 4096;
  poolInfo.poolSizeCount = 2;
  poolInfo.pPoolSizes = poolSizes;

  checkError(
      vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));
}

DeviceVulkan::~DeviceVulkan() {
  *alive = false;
  if (device != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device);
    if (descriptorPool != VK_NULL_HANDLE)
      vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    if (ownsInstance) vkDestroyDevice(device, nullptr);
  }
  if (ownsInstance && instance != VK_NULL_HANDLE)
    vkDestroyInstance(instance, nullptr);
}

uint32_t DeviceVulkan::findMemoryType(uint32_t typeFilter,
                                      VkMemoryPropertyFlags props) const {
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) &&
        (memProperties.memoryTypes[i].propertyFlags & props) == props) {
      return i;
    }
  }
  // Fallback: try without DEVICE_LOCAL for host-visible requests
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) &&
        (memProperties.memoryTypes[i].propertyFlags &
         (props & ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) ==
            (props & ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      return i;
    }
  }
  throw std::runtime_error("Failed to find suitable Vulkan memory type");
}

void DeviceVulkan::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                VkMemoryPropertyFlags props, VkBuffer& buf,
                                VkDeviceMemory& mem) const {
  VkBufferCreateInfo bufInfo = {};
  bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufInfo.size = size;
  bufInfo.usage = usage;
  bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  checkError(vkCreateBuffer(device, &bufInfo, nullptr, &buf));

  VkMemoryRequirements memReqs;
  vkGetBufferMemoryRequirements(device, buf, &memReqs);

  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memReqs.size;
  allocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, props);

  checkError(vkAllocateMemory(device, &allocInfo, nullptr, &mem));
  checkError(vkBindBufferMemory(device, buf, mem, 0));
}

VkFormat DeviceVulkan::getImageFormat(const ImageDescription& descr) const {
  switch (descr.channels) {
    case 1:
      switch (descr.type) {
        case DataType_Float16:
          return VK_FORMAT_R16_SFLOAT;
        case DataType_Float:
          return VK_FORMAT_R32_SFLOAT;
        case DataType_UInt16:
          return VK_FORMAT_R16_UNORM;
        case DataType_Int16:
          return VK_FORMAT_R16_SNORM;
        case DataType_UInt8:
          return VK_FORMAT_R8_UNORM;
        case DataType_Int8:
          return VK_FORMAT_R8_SNORM;
        default:
          return VK_FORMAT_R8_UNORM;
      }
    case 2:
      switch (descr.type) {
        case DataType_Float16:
          return VK_FORMAT_R16G16_SFLOAT;
        case DataType_Float:
          return VK_FORMAT_R32G32_SFLOAT;
        case DataType_UInt16:
          return VK_FORMAT_R16G16_UNORM;
        case DataType_Int16:
          return VK_FORMAT_R16G16_SNORM;
        case DataType_UInt8:
          return VK_FORMAT_R8G8_UNORM;
        case DataType_Int8:
          return VK_FORMAT_R8G8_SNORM;
        default:
          return VK_FORMAT_R8G8_UNORM;
      }
    case 4:
    default:
      if (descr.order == PixelOrder_BGRA && descr.type == DataType_UInt8)
        return VK_FORMAT_B8G8R8A8_UNORM;
      switch (descr.type) {
        case DataType_Float16:
          return VK_FORMAT_R16G16B16A16_SFLOAT;
        case DataType_Float:
          return VK_FORMAT_R32G32B32A32_SFLOAT;
        case DataType_UInt16:
          return VK_FORMAT_R16G16B16A16_UNORM;
        case DataType_Int16:
          return VK_FORMAT_R16G16B16A16_SNORM;
        case DataType_UInt8:
          return VK_FORMAT_R8G8B8A8_UNORM;
        case DataType_Int8:
          return VK_FORMAT_R8G8B8A8_SNORM;
        default:
          return VK_FORMAT_R8G8B8A8_UNORM;
      }
  }
}

ghost::Library DeviceVulkan::loadLibraryFromText(const std::string& text,
                                                 const CompilerOptions& options,
                                                 bool retainBinary) const {
  // Vulkan requires pre-compiled SPIR-V; runtime GLSL compilation requires
  // shaderc or glslang which are not linked by default.
  throw ghost::unsupported_error();
}

ghost::Library DeviceVulkan::loadLibraryFromData(const void* data, size_t len,
                                                 const CompilerOptions& options,
                                                 bool retainBinary) const {
  auto lib = std::make_shared<LibraryVulkan>(*this, retainBinary);
  lib->loadFromData(data, len, options);
  return ghost::Library(lib);
}

SharedContext DeviceVulkan::shareContext() const {
  return SharedContext(instance, computeQueue, device, physicalDevice);
}

ghost::Stream DeviceVulkan::createStream() const {
  return ghost::Stream(std::make_shared<StreamVulkan>(*this));
}

ghost::Buffer DeviceVulkan::allocateBuffer(size_t bytes,
                                           const BufferOptions& opts) const {
  return ghost::Buffer(std::make_shared<BufferVulkan>(*this, bytes, opts));
}

ghost::MappedBuffer DeviceVulkan::allocateMappedBuffer(
    size_t bytes, const BufferOptions& opts) const {
  return ghost::MappedBuffer(
      std::make_shared<MappedBufferVulkan>(*this, bytes, opts));
}

ghost::Image DeviceVulkan::allocateImage(const ImageDescription& descr) const {
  return ghost::Image(std::make_shared<ImageVulkan>(*this, descr));
}

ghost::Image DeviceVulkan::sharedImage(const ImageDescription& descr,
                                       ghost::Buffer& buffer) const {
  // Vulkan doesn't support aliasing buffer memory to an image directly
  // like Metal/OpenCL do. Writes to the buffer would not be visible
  // through the image, so report this as unsupported.
  throw ghost::unsupported_error();
}

ghost::Image DeviceVulkan::sharedImage(const ImageDescription& descr,
                                       ghost::Image& image) const {
  auto* vkImg = static_cast<ImageVulkan*>(image.impl().get());
  return ghost::Image(std::make_shared<ImageVulkan>(*this, descr, *vkImg));
}

Attribute DeviceVulkan::getAttribute(DeviceAttributeId what) const {
  switch (what) {
    case kDeviceImplementation:
      return Attribute("Vulkan");
    case kDeviceName:
      return Attribute(properties.deviceName);
    case kDeviceVendor: {
      switch (properties.vendorID) {
        case 0x1002:
          return Attribute("AMD");
        case 0x10DE:
          return Attribute("NVIDIA");
        case 0x8086:
          return Attribute("Intel");
        case 0x13B5:
          return Attribute("ARM");
        case 0x5143:
          return Attribute("Qualcomm");
        default:
          return Attribute("Unknown");
      }
    }
    case kDeviceDriverVersion:
      return Attribute(
          std::to_string(VK_VERSION_MAJOR(properties.driverVersion)) + "." +
          std::to_string(VK_VERSION_MINOR(properties.driverVersion)) + "." +
          std::to_string(VK_VERSION_PATCH(properties.driverVersion)));
    case kDeviceFamily:
      return Attribute(std::to_string(VK_VERSION_MAJOR(properties.apiVersion)) +
                       "." +
                       std::to_string(VK_VERSION_MINOR(properties.apiVersion)));
    case kDeviceProcessorCount:
      return Attribute((int32_t)properties.limits.maxComputeWorkGroupCount[0]);
    case kDeviceUnifiedMemory:
      return Attribute(properties.deviceType ==
                       VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
    case kDeviceMemory: {
      VkDeviceSize totalMem = 0;
      for (uint32_t i = 0; i < memProperties.memoryHeapCount; i++) {
        if (memProperties.memoryHeaps[i].flags &
            VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
          totalMem += memProperties.memoryHeaps[i].size;
        }
      }
      return Attribute((uint64_t)totalMem);
    }
    case kDeviceLocalMemory:
      return Attribute((int32_t)properties.limits.maxComputeSharedMemorySize);
    case kDeviceMaxThreads:
      return Attribute(
          (int32_t)properties.limits.maxComputeWorkGroupInvocations);
    case kDeviceMaxWorkSize:
      return Attribute((int32_t)properties.limits.maxComputeWorkGroupSize[0],
                       (int32_t)properties.limits.maxComputeWorkGroupSize[1],
                       (int32_t)properties.limits.maxComputeWorkGroupSize[2]);
    case kDeviceMaxRegisters:
      return Attribute((int32_t)0);
    case kDeviceMaxImageSize1:
      return Attribute((int32_t)properties.limits.maxImageDimension1D);
    case kDeviceMaxImageSize2:
      return Attribute((int32_t)properties.limits.maxImageDimension2D,
                       (int32_t)properties.limits.maxImageDimension2D);
    case kDeviceMaxImageSize3:
      return Attribute((int32_t)properties.limits.maxImageDimension3D,
                       (int32_t)properties.limits.maxImageDimension3D,
                       (int32_t)properties.limits.maxImageDimension3D);
    case kDeviceImageAlignment:
      return Attribute(
          (int32_t)properties.limits.optimalBufferCopyRowPitchAlignment);
    case kDeviceSupportsImageIntegerFiltering:
      return Attribute(true);
    case kDeviceSupportsImageFloatFiltering:
      return Attribute(true);
    case kDeviceSupportsMappedBuffer:
      return Attribute(true);
    case kDeviceSupportsProgramConstants:
      return Attribute(true);
    case kDeviceSupportsSubgroup:
      return Attribute(true);
    case kDeviceSupportsSubgroupShuffle:
      return Attribute(true);
    case kDeviceSubgroupWidth:
      // Requires VK_KHR_subgroup_properties (Vulkan 1.1+)
      // Default to 32 (NVIDIA) or 64 (AMD)
      return Attribute((int32_t)32);
    case kDeviceMaxComputeUnits:
      // Not directly queryable in Vulkan; return 1 as the minimum
      return Attribute((int32_t)1);
    case kDeviceMemoryAlignment:
      return Attribute(
          (int32_t)properties.limits.minStorageBufferOffsetAlignment);
    case kDeviceBufferAlignment:
      return Attribute(
          (int32_t)properties.limits.minStorageBufferOffsetAlignment);
    case kDeviceMaxBufferSize:
      return Attribute((uint64_t)properties.limits.maxStorageBufferRange);
    case kDeviceMaxConstantBufferSize:
      return Attribute((int32_t)properties.limits.maxPushConstantsSize);
    case kDeviceTimestampPeriod:
      return Attribute(properties.limits.timestampPeriod);
    case kDeviceSupportsProfilingTimer:
      return Attribute(properties.limits.timestampComputeAndGraphics !=
                       VK_FALSE);
    default:
      return Attribute();
  }
}

}  // namespace implementation

// ---------------------------------------------------------------------------
// Public DeviceVulkan
// ---------------------------------------------------------------------------

DeviceVulkan::DeviceVulkan(const SharedContext& share)
    : Device(std::make_shared<implementation::DeviceVulkan>(share)) {
  setDefaultStream(std::make_shared<implementation::StreamVulkan>(
      *static_cast<implementation::DeviceVulkan*>(impl().get())));
}

DeviceVulkan::DeviceVulkan(const GpuInfo& info)
    : Device(std::make_shared<implementation::DeviceVulkan>(info)) {
  setDefaultStream(std::make_shared<implementation::StreamVulkan>(
      *static_cast<implementation::DeviceVulkan*>(impl().get())));
}

std::vector<GpuInfo> DeviceVulkan::enumerateDevices() {
  std::vector<GpuInfo> result;

  VkInstance inst;
  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Ghost";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "Ghost";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  if (vkCreateInstance(&createInfo, nullptr, &inst) != VK_SUCCESS)
    return result;

  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(inst, &deviceCount, nullptr);
  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(inst, &deviceCount, devices.data());

  for (uint32_t i = 0; i < deviceCount; i++) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(devices[i], &props);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(devices[i], &memProps);

    GpuInfo info;
    info.name = props.deviceName;
    info.implementation = "Vulkan";
    info.index = (int)i;
    info.unifiedMemory =
        (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);

    switch (props.vendorID) {
      case 0x1002:
        info.vendor = "AMD";
        break;
      case 0x10DE:
        info.vendor = "NVIDIA";
        break;
      case 0x8086:
        info.vendor = "Intel";
        break;
      case 0x13B5:
        info.vendor = "ARM";
        break;
      case 0x5143:
        info.vendor = "Qualcomm";
        break;
      default:
        info.vendor = "Unknown";
        break;
    }

    uint64_t totalMem = 0;
    for (uint32_t j = 0; j < memProps.memoryHeapCount; j++) {
      if (memProps.memoryHeaps[j].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
        totalMem += memProps.memoryHeaps[j].size;
      }
    }
    info.memory = totalMem;

    result.push_back(info);
  }

  vkDestroyInstance(inst, nullptr);
  return result;
}

}  // namespace ghost

#endif
