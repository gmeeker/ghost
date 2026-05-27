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

#ifndef GHOST_DIRECTX_IMPL_DEVICE_H
#define GHOST_DIRECTX_IMPL_DEVICE_H

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d12.h>
#include <dxgi1_6.h>
#include <ghost/device.h>
#include <ghost/implementation/recorded_command_buffer.h>
#include <wrl/client.h>

#include <vector>

using Microsoft::WRL::ComPtr;

namespace ghost {
namespace implementation {
class DeviceDirectX;

class EventDirectX : public Event {
 public:
  EventDirectX(ComPtr<ID3D12Fence> fence_, HANDLE event_, UINT64 value_);
  ~EventDirectX();

  virtual void wait() override;
  virtual bool isComplete() const override;
  virtual double elapsed(const Event& other) const override;

 private:
  ComPtr<ID3D12Fence> _fence;
  HANDLE _fenceEvent;
  UINT64 _fenceValue;
};

/// @brief State and lifecycle shared by every DirectX encoder (Stream and
/// CommandBuffer).
///
/// BufferDirectX / ImageDirectX / FunctionDirectX downcast a
/// @c ghost::Encoder to this type to find the @c ID3D12GraphicsCommandList
/// to record into, plus the per-cb shader-visible descriptor heaps used by
/// compute dispatches. Both @c StreamDirectX and @c CommandBufferDirectX
/// inherit from this mixin. Use @c directxEncoder(s) to perform the
/// cross-cast.
class DirectXEncoder {
 public:
  struct StagingResource {
    ComPtr<ID3D12Resource> resource;
  };

  struct DeferredRead {
    ComPtr<ID3D12Resource> staging;
    void* dstPtr;
    size_t offset;
    size_t size;
    // For image readbacks with pitch alignment
    size_t srcRowPitch =
        0;  // aligned row pitch in staging buffer (0 = flat copy)
    size_t dstRowPitch = 0;  // actual row pitch in destination
    size_t rowCount = 0;     // number of rows to copy
    size_t rowBytes = 0;     // bytes per row to copy
  };

  const DeviceDirectX& dev;
  // Public so BufferDirectX/ImageDirectX can record copy commands directly.
  ComPtr<ID3D12GraphicsCommandList> commandList;
  // Public so copyTo / copy paths can attach staging resources and deferred
  // host reads directly.
  std::vector<StagingResource> pendingStaging;
  std::vector<DeferredRead> deferredReads;

  // Shader-visible descriptor heaps for compute dispatches that need
  // descriptor tables (textures, samplers). Created lazily on first use.
  // Each dispatch bump-allocates slots; offsets are reset when the fence
  // for the previous submission has signaled (see @c begin).
  ComPtr<ID3D12DescriptorHeap> srvHeap;
  ComPtr<ID3D12DescriptorHeap> samplerHeap;
  UINT srvHandleSize = 0;
  UINT samplerHandleSize = 0;
  UINT srvNextSlot = 0;
  UINT samplerNextSlot = 0;
  UINT srvHeapCapacity = 0;
  UINT samplerHeapCapacity = 0;

  // When true, @c FunctionDirectX::execute skips its post-dispatch global
  // UAV barrier so consecutive dispatches in this cb may run concurrently —
  // caller takes responsibility for inter-dispatch hazards. Set once at
  // construction by StreamDirectX / CommandBufferDirectX.
  bool concurrent = false;

  explicit DirectXEncoder(const DeviceDirectX& dev_) : dev(dev_) {}

  virtual ~DirectXEncoder() = default;

  /// @brief Ensure @c commandList is in the recording state. Idempotent.
  virtual void begin() = 0;

  /// @brief Allocate @p count consecutive CBV/SRV/UAV heap slots and return
  /// the CPU and GPU handles of the first slot. Creates the heap on first
  /// use. Throws if the bump allocator would overflow the heap.
  void allocSrvSlots(UINT count, D3D12_CPU_DESCRIPTOR_HANDLE& cpuOut,
                     D3D12_GPU_DESCRIPTOR_HANDLE& gpuOut);
  void allocSamplerSlots(UINT count, D3D12_CPU_DESCRIPTOR_HANDLE& cpuOut,
                         D3D12_GPU_DESCRIPTOR_HANDLE& gpuOut);
  /// @brief Bind the descriptor heaps on the command list. Called once per
  /// command-list recording; safe to call multiple times (no-op if both are
  /// already the active heaps).
  void bindDescriptorHeaps();
};

class StreamDirectX : public Stream, public DirectXEncoder {
 public:
  StreamDirectX(const DeviceDirectX& dev_, const StreamOptions& options = {});
  ~StreamDirectX();

  virtual void sync() override;
  virtual std::shared_ptr<Event> record() override;
  virtual void waitForEvent(const std::shared_ptr<Event>& e) override;

  void begin() override;
  void submit();
  void cleanupStaging();

  /// @brief Execute a pre-recorded command list (e.g. from a
  /// @ref CommandBufferDirectX) on this stream's queue, signaling @p fence
  /// at @p fenceValue. Flushes any in-flight stream recording first so
  /// the externally-recorded list runs after the stream's prior work.
  void executeOnQueue(ID3D12CommandList* const* lists, UINT count,
                      ID3D12Fence* fence, UINT64 fenceValue);

 private:
  ComPtr<ID3D12CommandQueue> _commandQueue;
  ComPtr<ID3D12CommandAllocator> _commandAllocator;
  ComPtr<ID3D12Fence> _fence;
  HANDLE _fenceEvent;
  UINT64 _fenceValue;
  bool _recording;
  bool _submitted;
};

/// @brief Cross-cast a @c ghost::Encoder to its underlying @c DirectXEncoder.
///
/// Throws @c ghost::unsupported_error if the encoder's impl is not a
/// DirectX encoder.
DirectXEncoder& directxEncoder(const ghost::Encoder& s);

/// @brief Native DirectX @c CommandBuffer wrapping its own
/// @c ID3D12GraphicsCommandList, @c ID3D12CommandAllocator, fence value,
/// and per-cb descriptor heaps.
///
/// Inherits the variant-recording machinery from @ref RecordedCommandBuffer
/// and the encoder interface from @ref DirectXEncoder. On @c submit() the
/// variants are replayed into the owned @c ID3D12GraphicsCommandList, then
/// submitted via @c ExecuteCommandLists on the target Stream's queue with
/// this CommandBuffer's fence value. Resources captured by the variants
/// stay live until @c reset() (which waits on the fence first) or
/// destruction.
class CommandBufferDirectX : public RecordedCommandBuffer,
                             public DirectXEncoder {
 public:
  CommandBufferDirectX(const DeviceDirectX& dev_,
                       const CommandBufferOptions& options = {});
  ~CommandBufferDirectX();

  void begin() override;
  void submit(const ghost::Stream& stream) override;
  void reset() override;

 private:
  ComPtr<ID3D12CommandAllocator> _commandAllocator;
  ComPtr<ID3D12Fence> _fence;
  HANDLE _fenceEvent = nullptr;
  UINT64 _fenceValue = 0;
  bool _recording = false;
  bool _submitted = false;

  /// @brief Wait on @c _fence if a submission is in flight. Idempotent.
  void waitForCompletion();
};

class BufferDirectX : public Buffer {
 public:
  ComPtr<ID3D12Resource> resource;
  size_t _size;
  D3D12_RESOURCE_STATES currentState;
  // UPLOAD and READBACK heap resources are locked in their initial state by
  // D3D12; emitting a ResourceBarrier against them returns E_INVALIDARG. We
  // record the heap so transitionTo() can skip those.
  D3D12_HEAP_TYPE _heapType = D3D12_HEAP_TYPE_DEFAULT;

  BufferDirectX(const DeviceDirectX& dev, size_t bytes,
                const BufferOptions& opts = {});
  BufferDirectX(ComPtr<ID3D12Resource> res, size_t bytes,
                D3D12_RESOURCE_STATES state);
  ~BufferDirectX();

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

  void transitionTo(ID3D12GraphicsCommandList* cmdList,
                    D3D12_RESOURCE_STATES newState);
};

class SubBufferDirectX : public BufferDirectX {
 public:
  std::shared_ptr<Buffer> _parent;
  size_t _offset;

  SubBufferDirectX(std::shared_ptr<Buffer> parent, ComPtr<ID3D12Resource> res,
                   size_t offset, size_t bytes);

  virtual size_t baseOffset() const override;
};

class MappedBufferDirectX : public BufferDirectX {
 public:
  void* mappedPtr;

  MappedBufferDirectX(const DeviceDirectX& dev_, size_t bytes,
                      const BufferOptions& opts = {});
  ~MappedBufferDirectX();

  virtual void* map(const ghost::Encoder& s, Access access,
                    bool sync = true) override;
  virtual void unmap(const ghost::Encoder& s) override;
};

class ImageDirectX : public Image {
 public:
  ComPtr<ID3D12Resource> resource;
  ImageDescription descr;
  D3D12_RESOURCE_STATES currentState;

  ImageDirectX(const DeviceDirectX& dev, const ImageDescription& descr);
  ImageDirectX(const DeviceDirectX& dev, const ImageDescription& descr,
               BufferDirectX& buffer);
  ImageDirectX(const DeviceDirectX& dev, const ImageDescription& descr,
               ImageDirectX& image);
  ~ImageDirectX();

  virtual const ImageDescription& description() const override { return descr; }

  virtual void copy(const ghost::Encoder& s, const ghost::Image& src) override;
  virtual void copy(const ghost::Encoder& s, const ghost::Image& src,
                    const Size3& region, const Origin3& srcOrigin,
                    const Origin3& dstOrigin) override;
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

  void transitionTo(ID3D12GraphicsCommandList* cmdList,
                    D3D12_RESOURCE_STATES newState);
};

class DeviceDirectX : public Device {
 public:
  ComPtr<ID3D12Device> device;
  ComPtr<ID3D12CommandQueue> commandQueue;

  DeviceDirectX(const SharedContext& share);
  DeviceDirectX(const GpuInfo& info);
  ~DeviceDirectX();

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

  DXGI_FORMAT getImageFormat(const ImageDescription& descr) const;
  ComPtr<ID3D12Resource> createCommittedBuffer(
      size_t bytes, D3D12_HEAP_TYPE heapType,
      D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
      D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_COMMON) const;

 private:
  ComPtr<IDXGIFactory4> _factory;
  ComPtr<IDXGIAdapter1> _adapter;
  DXGI_ADAPTER_DESC1 _adapterDesc;
};
}  // namespace implementation
}  // namespace ghost

#endif
