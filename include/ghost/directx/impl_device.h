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
#include <wrl/client.h>

#include <vector>

using Microsoft::WRL::ComPtr;

namespace ghost {
namespace implementation {
class DeviceDirectX;

class EventDirectX : public Event {
 public:
  ComPtr<ID3D12Fence> fence;
  HANDLE fenceEvent;
  UINT64 fenceValue;

  EventDirectX(ComPtr<ID3D12Fence> fence_, HANDLE event_, UINT64 value_);
  ~EventDirectX();

  virtual void wait() override;
  virtual bool isComplete() const override;
  virtual double elapsed(const Event& other) const override;
};

class StreamDirectX : public Stream {
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
  ComPtr<ID3D12CommandQueue> commandQueue;
  ComPtr<ID3D12CommandAllocator> commandAllocator;
  ComPtr<ID3D12GraphicsCommandList> commandList;
  ComPtr<ID3D12Fence> fence;
  HANDLE fenceEvent;
  UINT64 fenceValue;
  bool recording;
  bool submitted;
  std::vector<StagingResource> pendingStaging;
  std::vector<DeferredRead> deferredReads;

  StreamDirectX(const DeviceDirectX& dev_);
  ~StreamDirectX();

  virtual void sync() override;
  virtual std::shared_ptr<Event> record() override;
  virtual void waitForEvent(const std::shared_ptr<Event>& e) override;

  void begin();
  void submit();
  void cleanupStaging();
};

class BufferDirectX : public Buffer {
 public:
  const DeviceDirectX& dev;
  ComPtr<ID3D12Resource> resource;
  size_t _size;
  D3D12_RESOURCE_STATES currentState;

  BufferDirectX(const DeviceDirectX& dev_, size_t bytes,
                Access access = Access_ReadWrite);
  BufferDirectX(const DeviceDirectX& dev_, ComPtr<ID3D12Resource> res,
                size_t bytes, D3D12_RESOURCE_STATES state);
  ~BufferDirectX();

  virtual size_t size() const override;

  virtual void copy(const ghost::Stream& s, const ghost::Buffer& src,
                    size_t bytes) override;
  virtual void copy(const ghost::Stream& s, const void* src,
                    size_t bytes) override;
  virtual void copyTo(const ghost::Stream& s, void* dst,
                      size_t bytes) const override;

  virtual void copy(const ghost::Stream& s, const ghost::Buffer& src,
                    size_t srcOffset, size_t dstOffset, size_t bytes) override;
  virtual void copy(const ghost::Stream& s, const void* src, size_t dstOffset,
                    size_t bytes) override;
  virtual void copyTo(const ghost::Stream& s, void* dst, size_t srcOffset,
                      size_t bytes) const override;

  virtual void fill(const ghost::Stream& s, size_t offset, size_t size,
                    uint8_t value) override;
  virtual void fill(const ghost::Stream& s, size_t offset, size_t size,
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

  SubBufferDirectX(std::shared_ptr<Buffer> parent, const DeviceDirectX& dev_,
                   ComPtr<ID3D12Resource> res, size_t offset, size_t bytes);

  virtual size_t baseOffset() const override;

  virtual void copy(const ghost::Stream& s, const ghost::Buffer& src,
                    size_t bytes) override;
  virtual void copy(const ghost::Stream& s, const void* src,
                    size_t bytes) override;
  virtual void copyTo(const ghost::Stream& s, void* dst,
                      size_t bytes) const override;
};

class MappedBufferDirectX : public BufferDirectX {
 public:
  void* mappedPtr;

  MappedBufferDirectX(const DeviceDirectX& dev_, size_t bytes,
                      Access access = Access_ReadWrite);
  ~MappedBufferDirectX();

  virtual void* map(const ghost::Stream& s, Access access,
                    bool sync = true) override;
  virtual void unmap(const ghost::Stream& s) override;
};

class ImageDirectX : public Image {
 public:
  const DeviceDirectX& dev;
  ComPtr<ID3D12Resource> resource;
  ImageDescription descr;
  D3D12_RESOURCE_STATES currentState;

  ImageDirectX(const DeviceDirectX& dev_, const ImageDescription& descr);
  ImageDirectX(const DeviceDirectX& dev_, const ImageDescription& descr,
               BufferDirectX& buffer);
  ImageDirectX(const DeviceDirectX& dev_, const ImageDescription& descr,
               ImageDirectX& image);
  ~ImageDirectX();

  virtual void copy(const ghost::Stream& s, const ghost::Image& src) override;
  virtual void copy(const ghost::Stream& s, const ghost::Buffer& src,
                    const ImageDescription& descr) override;
  virtual void copy(const ghost::Stream& s, const void* src,
                    const ImageDescription& descr) override;
  virtual void copyTo(const ghost::Stream& s, ghost::Buffer& dst,
                      const ImageDescription& descr) const override;
  virtual void copyTo(const ghost::Stream& s, void* dst,
                      const ImageDescription& descr) const override;
  virtual void copy(const ghost::Stream& s, const ghost::Buffer& src,
                    const ImageDescription& descr,
                    const Size3& imageOrigin) override;
  virtual void copyTo(const ghost::Stream& s, ghost::Buffer& dst,
                      const ImageDescription& descr,
                      const Size3& imageOrigin) const override;

  void transitionTo(ID3D12GraphicsCommandList* cmdList,
                    D3D12_RESOURCE_STATES newState);
};

class DeviceDirectX : public Device {
 public:
  ComPtr<IDXGIFactory4> factory;
  ComPtr<IDXGIAdapter1> adapter;
  ComPtr<ID3D12Device> device;
  ComPtr<ID3D12CommandQueue> commandQueue;
  DXGI_ADAPTER_DESC1 adapterDesc;

  DeviceDirectX(const SharedContext& share);
  DeviceDirectX(const GpuInfo& info);
  ~DeviceDirectX();

  virtual ghost::Library loadLibraryFromText(
      const std::string& text, const std::string& options = "",
      bool retainBinary = false) const override;
  virtual ghost::Library loadLibraryFromData(
      const void* data, size_t len, const std::string& options = "",
      bool retainBinary = false) const override;

  virtual SharedContext shareContext() const override;

  virtual ghost::Stream createStream() const override;

  virtual ghost::Buffer allocateBuffer(
      size_t bytes, Access access = Access_ReadWrite) const override;
  virtual ghost::MappedBuffer allocateMappedBuffer(
      size_t bytes, Access access = Access_ReadWrite) const override;
  virtual ghost::Image allocateImage(
      const ImageDescription& descr) const override;
  virtual ghost::Image sharedImage(const ImageDescription& descr,
                                   ghost::Buffer& buffer) const override;
  virtual ghost::Image sharedImage(const ImageDescription& descr,
                                   ghost::Image& image) const override;

  virtual Attribute getAttribute(DeviceAttributeId what) const override;

  DXGI_FORMAT getImageFormat(const ImageDescription& descr) const;
  ComPtr<ID3D12Resource> createCommittedBuffer(
      size_t bytes, D3D12_HEAP_TYPE heapType,
      D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
      D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_COMMON) const;
};
}  // namespace implementation
}  // namespace ghost

#endif
