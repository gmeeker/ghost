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

#ifndef GHOST_VULKAN_IMPL_DEVICE_H
#define GHOST_VULKAN_IMPL_DEVICE_H

#include <ghost/device.h>
#include <ghost/vulkan/ptr.h>

#include <memory>
#include <vector>

namespace ghost {
namespace implementation {
class DeviceVulkan;

class EventVulkan : public Event {
 public:
  vk::ptr<VkFence> fence;

  EventVulkan(VkDevice dev, VkFence f, bool owns = true);

  virtual void wait() override;
  virtual bool isComplete() const override;
  virtual double elapsed(const Event& other) const override;
};

class StreamVulkan : public Stream {
 public:
  struct StagingResource {
    vk::ptr<VkBuffer> buffer;
    vk::ptr<VkDeviceMemory> memory;
  };

  struct DeferredRead {
    vk::ptr<VkBuffer> stagingBuffer;
    vk::ptr<VkDeviceMemory> stagingMemory;
    void* dstPtr;
    size_t offset;
    size_t size;
  };

  const DeviceVulkan& dev;
  // commandBuffer is allocated from _commandPool and freed implicitly when
  // the pool is destroyed; no separate destroy call needed. Public so
  // BufferVulkan/ImageVulkan can record copy commands directly.
  VkCommandBuffer commandBuffer;
  // Public so copyTo paths can attach deferred host reads after staging.
  std::vector<DeferredRead> deferredReads;

  StreamVulkan(const DeviceVulkan& dev_);
  ~StreamVulkan();

  virtual void sync() override;
  virtual std::shared_ptr<Event> record() override;
  virtual void waitForEvent(const std::shared_ptr<Event>& e) override;

  void begin();
  void submit();
  void cleanupStaging();
  void addStagingResource(vk::ptr<VkBuffer> buf, vk::ptr<VkDeviceMemory> mem);

 private:
  vk::ptr<VkCommandPool> _commandPool;
  vk::ptr<VkFence> _fence;
  bool _recording;
  bool _submitted;
  std::vector<StagingResource> _pendingStaging;
};

class BufferVulkan : public Buffer {
 public:
  vk::ptr<VkBuffer> buffer;
  vk::ptr<VkDeviceMemory> memory;
  size_t _size;

  BufferVulkan(const DeviceVulkan& dev, size_t bytes,
               const BufferOptions& opts = {});
  // Borrowed-handle constructor: takes a (device, buffer, memory) tuple.
  // With owns=false, used by SubBufferVulkan to alias a parent buffer's
  // handle. With null handles + owns=true, used by MappedBufferVulkan as
  // an empty base before populating in the derived ctor body.
  BufferVulkan(VkDevice device, VkBuffer buf, VkDeviceMemory mem, size_t bytes,
               bool owns = true);

  virtual size_t size() const override;

  virtual void copy(const ghost::Encoder& s, const ghost::Buffer& src,
                    size_t bytes) override;
  virtual void copy(const ghost::Encoder& s, const void* src,
                    size_t bytes) override;
  virtual void copyTo(const ghost::Encoder& s, void* dst,
                      size_t bytes) const override;

  virtual void copy(const ghost::Encoder& s, const ghost::Buffer& src,
                    size_t srcOffset, size_t dstOffset, size_t bytes) override;
  virtual void copy(const ghost::Encoder& s, const void* src, size_t dstOffset,
                    size_t bytes) override;
  virtual void copyTo(const ghost::Encoder& s, void* dst, size_t srcOffset,
                      size_t bytes) const override;

  virtual void fill(const ghost::Encoder& s, size_t offset, size_t size,
                    uint8_t value) override;
  virtual void fill(const ghost::Encoder& s, size_t offset, size_t size,
                    const void* pattern, size_t patternSize) override;

  virtual std::shared_ptr<Buffer> createSubBuffer(
      const std::shared_ptr<Buffer>& self, size_t offset, size_t size) override;
};

class SubBufferVulkan : public BufferVulkan {
 public:
  std::shared_ptr<Buffer> _parent;
  size_t _offset;

  SubBufferVulkan(std::shared_ptr<Buffer> parent, VkDevice device, VkBuffer buf,
                  size_t offset, size_t bytes);

  virtual size_t baseOffset() const override;

  virtual void copy(const ghost::Encoder& s, const ghost::Buffer& src,
                    size_t bytes) override;
  virtual void copy(const ghost::Encoder& s, const void* src,
                    size_t bytes) override;
  virtual void copyTo(const ghost::Encoder& s, void* dst,
                      size_t bytes) const override;
};

class MappedBufferVulkan : public BufferVulkan {
 public:
  void* mappedPtr;

  MappedBufferVulkan(const DeviceVulkan& dev_, size_t bytes,
                     const BufferOptions& opts = {});

  virtual void* map(const ghost::Encoder& s, Access access,
                    bool sync = true) override;
  virtual void unmap(const ghost::Encoder& s) override;
};

class ImageVulkan : public Image {
 public:
  vk::ptr<VkImage> image;
  vk::ptr<VkDeviceMemory> memory;
  // imageView is always owned by this object, even for image-from-image
  // aliases (which create a fresh view into another image's memory).
  vk::ptr<VkImageView> imageView;
  ImageDescription descr;

  ImageVulkan(const DeviceVulkan& dev, const ImageDescription& descr);
  ImageVulkan(const DeviceVulkan& dev, const ImageDescription& descr,
              BufferVulkan& buffer);
  ImageVulkan(const DeviceVulkan& dev, const ImageDescription& descr,
              ImageVulkan& image);

  virtual const ImageDescription& description() const override { return descr; }

  virtual void copy(const ghost::Encoder& s, const ghost::Image& src) override;
  virtual void copy(const ghost::Encoder& s, const ghost::Buffer& src,
                    const BufferLayout& layout) override;
  virtual void copy(const ghost::Encoder& s, const void* src,
                    const BufferLayout& layout) override;
  virtual void copyTo(const ghost::Encoder& s, ghost::Buffer& dst,
                      const BufferLayout& layout) const override;
  virtual void copyTo(const ghost::Encoder& s, void* dst,
                      const BufferLayout& layout) const override;
  virtual void copy(const ghost::Encoder& s, const ghost::Buffer& src,
                    const BufferLayout& layout,
                    const Origin3& imageOrigin) override;
  virtual void copyTo(const ghost::Encoder& s, ghost::Buffer& dst,
                      const BufferLayout& layout,
                      const Origin3& imageOrigin) const override;
  virtual void copy(const ghost::Encoder& s, const ghost::Image& src,
                    const Size3& region, const Origin3& srcOrigin,
                    const Origin3& dstOrigin) override;
};

class DeviceVulkan : public Device {
 public:
  // VkInstance and VkDevice are kept as raw handles because their destroy
  // functions take no parent device — they are special-cased in the
  // destructor and gated by _ownsInstance.
  VkInstance instance;
  VkPhysicalDevice physicalDevice;
  VkDevice device;
  VkQueue computeQueue;
  uint32_t computeQueueFamily;
  vk::ptr<VkDescriptorPool> descriptorPool;
  VkPhysicalDeviceProperties properties;

  DeviceVulkan(const SharedContext& share);
  DeviceVulkan(const GpuInfo& info);
  ~DeviceVulkan();

  virtual ghost::Library loadLibraryFromText(
      const std::string& text,
      const CompilerOptions& options = CompilerOptions(),
      bool retainBinary = false) const override;
  virtual ghost::Library loadLibraryFromData(
      const void* data, size_t len,
      const CompilerOptions& options = CompilerOptions(),
      bool retainBinary = false) const override;

  virtual SharedContext shareContext() const override;

  virtual ghost::Stream createStream(
      const StreamOptions& options = {}) const override;

  virtual ghost::Buffer allocateBuffer(
      size_t bytes, const BufferOptions& opts = {}) const override;
  virtual ghost::MappedBuffer allocateMappedBuffer(
      size_t bytes, const BufferOptions& opts = {}) const override;
  virtual ghost::Image allocateImage(
      const ImageDescription& descr) const override;
  virtual ghost::Image sharedImage(const ImageDescription& descr,
                                   ghost::Buffer& buffer) const override;
  virtual ghost::Image sharedImage(const ImageDescription& descr,
                                   ghost::Image& image) const override;

  virtual Attribute getAttribute(DeviceAttributeId what) const override;

  uint32_t findMemoryType(uint32_t typeFilter,
                          VkMemoryPropertyFlags props) const;
  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags props, vk::ptr<VkBuffer>& buf,
                    vk::ptr<VkDeviceMemory>& mem) const;
  VkFormat getImageFormat(const ImageDescription& descr) const;

 private:
  bool _ownsInstance;
  VkPhysicalDeviceMemoryProperties _memProperties;
};
}  // namespace implementation
}  // namespace ghost

#endif
