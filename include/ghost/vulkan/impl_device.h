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
#include <ghost/implementation/executable.h>
#include <ghost/implementation/recorded_command_buffer.h>
#include <ghost/vulkan/ptr.h>

#include <map>
#include <memory>
#include <vector>

namespace ghost {

/// @brief Layout an @c Allocator must use when returning a Vulkan buffer.
///
/// Vulkan resources require two handles ( @c VkBuffer and the backing
/// @c VkDeviceMemory ). Ghost cannot reconstruct that pair from a single
/// opaque pointer, so on Vulkan the @c void* returned by
/// @c Allocator::allocateBuffer / @c allocateMappedBuffer must point to a
/// host-owned struct that begins with these two fields. The host retains
/// ownership of the struct's storage and frees it from
/// @c freeBuffer / @c freeMappedBuffer along with the underlying Vulkan
/// resources.
struct VulkanBufferHandle {
  VkBuffer buffer;
  VkDeviceMemory memory;
};

/// @brief Layout an @c Allocator must use when returning a Vulkan image.
///
/// Same conventions as @c VulkanBufferHandle. The optional @c imageView field
/// is reserved for future use; Ghost currently creates its own view.
struct VulkanImageHandle {
  VkImage image;
  VkDeviceMemory memory;
};

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

/// @brief State and lifecycle shared by every Vulkan encoder (Stream and
/// CommandBuffer).
///
/// BufferVulkan / ImageVulkan / FunctionVulkan downcast a @c ghost::Encoder
/// to this type to find the @c VkCommandBuffer to record into. Both
/// @c StreamVulkan and @c CommandBufferVulkan inherit from this mixin in
/// addition to their @c implementation::Encoder-derived base. Use
/// @c vulkanEncoder(s) to perform the cross-cast.
class VulkanEncoder {
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
  // commandBuffer is allocated from the encoder's command pool and freed
  // implicitly when the pool is destroyed. Public so BufferVulkan /
  // ImageVulkan / FunctionVulkan can record commands directly.
  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
  // Public so copyTo paths can attach deferred host reads after staging.
  std::vector<DeferredRead> deferredReads;
  // When true, @c FunctionVulkan::execute skips its post-dispatch
  // @c vkCmdPipelineBarrier so consecutive dispatches in this cb may run
  // concurrently — caller takes responsibility for inter-dispatch hazards.
  // Set once at construction by StreamVulkan / CommandBufferVulkan.
  bool concurrent = false;

  explicit VulkanEncoder(const DeviceVulkan& dev_) : dev(dev_) {}

  virtual ~VulkanEncoder() = default;

  /// @brief Ensure @c commandBuffer is in the recording state. Idempotent.
  virtual void begin() = 0;

  /// @brief Attach a staging buffer to this encoder so its lifetime extends
  /// to the encoder's next drain (sync / fence wait).
  void addStagingResource(vk::ptr<VkBuffer> buf, vk::ptr<VkDeviceMemory> mem) {
    _pendingStaging.push_back({std::move(buf), std::move(mem)});
  }

 protected:
  std::vector<StagingResource> _pendingStaging;
};

class StreamVulkan : public Stream, public VulkanEncoder {
 public:
  StreamVulkan(const DeviceVulkan& dev_, const StreamOptions& options = {});
  ~StreamVulkan();

  virtual void sync() override;
  virtual std::shared_ptr<Event> record() override;
  virtual void waitForEvent(const std::shared_ptr<Event>& e) override;

  void begin() override;
  void submit();
  void cleanupStaging();

  /// @brief Take ownership of an attachment fence representing an
  /// externally-submitted command buffer's completion.
  ///
  /// CommandBufferVulkan calls this after queueing its own work plus a
  /// trailing empty signal-only submission on the stream's queue. Queue
  /// FIFO order guarantees the attachment fence fires after the cb's
  /// work completes, so waiting on every attached fence in @c sync()
  /// gives the stream correct "drain everything I've been used to
  /// submit" semantics even though Vulkan binary fences can't be
  /// signaled twice from a single submission.
  void attachFence(vk::ptr<VkFence> fence);

 private:
  vk::ptr<VkCommandPool> _commandPool;
  vk::ptr<VkFence> _fence;
  bool _recording;
  bool _submitted;
  // Pending CommandBuffer-completion fences, owned here and drained on
  // sync(). See attachFence().
  std::vector<vk::ptr<VkFence>> _attachedFences;
};

/// @brief Cross-cast a @c ghost::Encoder to its underlying @c VulkanEncoder.
///
/// Throws @c ghost::unsupported_error if the encoder's impl is not a
/// Vulkan encoder.
VulkanEncoder& vulkanEncoder(const ghost::Encoder& s);

/// @brief Native Vulkan @c CommandBuffer wrapping its own @c VkCommandBuffer,
/// @c VkCommandPool and @c VkFence.
///
/// Inherits the variant-recording machinery from @ref RecordedCommandBuffer
/// (dispatch / copy / fill / barrier records accumulate into @c commands)
/// and the encoder interface from @ref VulkanEncoder (its @c commandBuffer
/// is the recording target). On @c submit() the variants are replayed
/// directly into the owned @c VkCommandBuffer, then submitted via
/// @c vkQueueSubmit on the target Stream's queue with this CommandBuffer's
/// fence. Resources captured by the variants stay live until @c reset()
/// (which waits on the fence first) or destruction.
class CommandBufferVulkan : public RecordedCommandBuffer, public VulkanEncoder {
 public:
  CommandBufferVulkan(const DeviceVulkan& dev_,
                      const CommandBufferOptions& options = {});
  ~CommandBufferVulkan();

  void begin() override;
  void submit(const ghost::Stream& stream) override;
  void reset() override;

  /// @brief Defer a native Vulkan encoding step to submit-time replay.
  ///
  /// At replay, Ghost ensures @c commandBuffer is in the recording state
  /// and invokes @p body with this encoder. Record @c vkCmdXxx directly
  /// onto @c encoder.commandBuffer.
  ///
  /// Body contract (initial, may tighten as concrete uses appear):
  ///   1. Do not call @c vkEndCommandBuffer on @c commandBuffer.
  ///   2. If the body writes resources that subsequently recorded Ghost
  ///      commands read, issue a @c vkCmdPipelineBarrier (or sync2
  ///      equivalent) at the end of the body whose @c dstStageMask covers
  ///      @c VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT and
  ///      @c VK_PIPELINE_STAGE_TRANSFER_BIT with matching access masks
  ///      for the resources written.
  ///   3. Resource state (image layouts in particular) must be left in
  ///      the state the body found it in unless the body owns the
  ///      resource.
  void encodeNative(std::function<void(VulkanEncoder&)> body);

  std::shared_ptr<Executable> compile(const CompileOptions& options) override;

  std::shared_ptr<RecordedCommandBuffer> cloneEmpty() const override;

  /// @brief Record the current @c commands into @c commandBuffer once, in a
  /// resubmittable state (no @c ONE_TIME_SUBMIT), leaving it ended and ready
  /// for repeated @ref submitRecorded. Used by @ref ExecutableVulkan.
  void recordResubmittable();

  /// @brief Submit the already-recorded @c commandBuffer to @p stream's queue
  /// without re-recording. Serializes against the prior submission via
  /// @c _fence. Used by @ref ExecutableVulkan.
  void submitRecorded(const ghost::Stream& stream);

 private:
  vk::ptr<VkCommandPool> _commandPool;
  vk::ptr<VkFence> _fence;
  bool _recording = false;
  bool _submitted = false;

  /// @brief Wait on @c _fence if a submission is in flight. Idempotent.
  void waitForCompletion();

  /// @brief Replay the recorded @c commands into @c commandBuffer (assumes it
  /// is already in the recording state; does not begin/end it).
  void replayCommandsIntoCb();

  /// @brief Submit @c commandBuffer on @p streamVk's queue with @c _fence and
  /// queue the trailing attachment-fence so @c Stream::sync drains it.
  void queueSubmitAndAttach(StreamVulkan* streamVk);
};

/// @brief Vulkan-native @ref Executable: a @ref CommandBufferVulkan whose
/// @c VkCommandBuffer is recorded once (resubmittable) at compile time and
/// re-submitted on each @ref submit without re-recording.
class ExecutableVulkan : public Executable {
 public:
  explicit ExecutableVulkan(std::shared_ptr<CommandBufferVulkan> cb)
      : _cb(std::move(cb)) {}

  void submit(const ghost::Stream& stream) override {
    _cb->submitRecorded(stream);
  }

  void update(const std::vector<Command>& commands) override {
    _cb->reset();  // waits for any in-flight submission, clears prior state
    _cb->commands = commands;
    _cb->recordResubmittable();
  }

  bool accelerated() const override { return true; }

 private:
  std::shared_ptr<CommandBufferVulkan> _cb;
};

class BufferVulkan : public Buffer {
 public:
  vk::ptr<VkBuffer> buffer;
  vk::ptr<VkDeviceMemory> memory;
  size_t _size;
  // Vulkan-only: opaque handle returned by an external @c Allocator. The
  // {VkBuffer, VkDeviceMemory} pair cannot reconstruct the host's struct, so
  // we hold the pointer the host returned and pass it back on free.
  void* _externalHandle = nullptr;

  BufferVulkan(const DeviceVulkan& dev, size_t bytes,
               const BufferOptions& opts = {});
  // Borrowed-handle constructor: takes a (device, buffer, memory) tuple.
  // With owns=false, used by SubBufferVulkan to alias a parent buffer's
  // handle. With null handles + owns=true, used by MappedBufferVulkan as
  // an empty base before populating in the derived ctor body.
  BufferVulkan(VkDevice device, VkBuffer buf, VkDeviceMemory mem, size_t bytes,
               bool owns = true);
  ~BufferVulkan();

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
  ~MappedBufferVulkan();

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
  // Vulkan-only: opaque handle from external @c Allocator (see BufferVulkan).
  void* _externalHandle = nullptr;

  ImageVulkan(const DeviceVulkan& dev, const ImageDescription& descr);
  ImageVulkan(const DeviceVulkan& dev, const ImageDescription& descr,
              BufferVulkan& buffer);
  ImageVulkan(const DeviceVulkan& dev, const ImageDescription& descr,
              ImageVulkan& image);
  /// @brief External-handle constructor used by the allocator path. Takes
  /// host-owned @c VkImage / @c VkDeviceMemory (non-owning wrappers) and
  /// creates a fresh @c VkImageView over the external image.
  ImageVulkan(const DeviceVulkan& dev, const ImageDescription& descr,
              VkImage extImage, VkDeviceMemory extMemory);
  ~ImageVulkan();

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

  virtual std::shared_ptr<CommandBuffer> createCommandBuffer(
      const CommandBufferOptions& options = {}) const override;

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

  virtual ghost::Buffer wrapBuffer(const SharedBuffer& shared) const override;
  virtual ghost::Image wrapImage(const SharedImage& shared) const override;

  virtual Attribute getAttribute(DeviceAttributeId what) const override;

  uint32_t findMemoryType(uint32_t typeFilter,
                          VkMemoryPropertyFlags props) const;
  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags props, vk::ptr<VkBuffer>& buf,
                    vk::ptr<VkDeviceMemory>& mem) const;
  VkFormat getImageFormat(const ImageDescription& descr) const;

  /// @brief Get or create a cached @c VkSampler matching @p desc.
  ///
  /// Samplers are cheap, device-wide, and fully described by @c filter,
  /// @c address and @c normalizedCoords. Caching avoids recreating one per
  /// dispatch when the same description is reused. Lives for the lifetime
  /// of the device.
  VkSampler getOrCreateSampler(const SamplerDescription& desc) const;

 private:
  bool _ownsInstance;
  VkPhysicalDeviceMemoryProperties _memProperties;
  mutable std::map<SamplerDescription, vk::ptr<VkSampler>> _samplerCache;
};
}  // namespace implementation
}  // namespace ghost

#endif
