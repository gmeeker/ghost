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

#include <ghost/directx/device.h>
#include <ghost/directx/exception.h>
#include <ghost/directx/impl_device.h>
#include <ghost/directx/impl_function.h>
#include <ghost/exception.h>

#include <algorithm>
#include <cstring>

namespace ghost {
namespace implementation {
using namespace dx;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

DXGI_FORMAT getFormat(const ImageDescription& descr) {
  switch (descr.channels) {
    case 1:
      switch (descr.type) {
        case DataType_Float16:
          return DXGI_FORMAT_R16_FLOAT;
        case DataType_Float:
          return DXGI_FORMAT_R32_FLOAT;
        case DataType_UInt16:
          return DXGI_FORMAT_R16_UNORM;
        case DataType_Int16:
          return DXGI_FORMAT_R16_SNORM;
        case DataType_UInt8:
          return DXGI_FORMAT_R8_UNORM;
        case DataType_Int8:
          return DXGI_FORMAT_R8_SNORM;
        default:
          return DXGI_FORMAT_R8_UNORM;
      }
    case 2:
      switch (descr.type) {
        case DataType_Float16:
          return DXGI_FORMAT_R16G16_FLOAT;
        case DataType_Float:
          return DXGI_FORMAT_R32G32_FLOAT;
        case DataType_UInt16:
          return DXGI_FORMAT_R16G16_UNORM;
        case DataType_Int16:
          return DXGI_FORMAT_R16G16_SNORM;
        case DataType_UInt8:
          return DXGI_FORMAT_R8G8_UNORM;
        case DataType_Int8:
          return DXGI_FORMAT_R8G8_SNORM;
        default:
          return DXGI_FORMAT_R8G8_UNORM;
      }
    case 4:
    default:
      if (descr.order == PixelOrder_BGRA && descr.type == DataType_UInt8)
        return DXGI_FORMAT_B8G8R8A8_UNORM;
      switch (descr.type) {
        case DataType_Float16:
          return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case DataType_Float:
          return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case DataType_UInt16:
          return DXGI_FORMAT_R16G16B16A16_UNORM;
        case DataType_Int16:
          return DXGI_FORMAT_R16G16B16A16_SNORM;
        case DataType_UInt8:
          return DXGI_FORMAT_R8G8B8A8_UNORM;
        case DataType_Int8:
          return DXGI_FORMAT_R8G8B8A8_SNORM;
        default:
          return DXGI_FORMAT_R8G8B8A8_UNORM;
      }
  }
}

D3D12_RESOURCE_DIMENSION getResourceDimension(const ImageDescription& descr) {
  if (descr.size.z > 1) return D3D12_RESOURCE_DIMENSION_TEXTURE3D;
  if (descr.size.y > 1) return D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  return D3D12_RESOURCE_DIMENSION_TEXTURE1D;
}

}  // namespace

// ---------------------------------------------------------------------------
// EventDirectX
// ---------------------------------------------------------------------------

EventDirectX::EventDirectX(ComPtr<ID3D12Fence> fence_, HANDLE event_,
                           UINT64 value_)
    : fence(fence_), fenceEvent(event_), fenceValue(value_) {}

EventDirectX::~EventDirectX() {
  if (fenceEvent) CloseHandle(fenceEvent);
}

void EventDirectX::wait() {
  if (fence->GetCompletedValue() < fenceValue) {
    fence->SetEventOnCompletion(fenceValue, fenceEvent);
    WaitForSingleObject(fenceEvent, INFINITE);
  }
}

bool EventDirectX::isComplete() const {
  return fence->GetCompletedValue() >= fenceValue;
}

double EventDirectX::elapsed(const Event& other) const { return 0.0; }

// ---------------------------------------------------------------------------
// StreamDirectX
// ---------------------------------------------------------------------------

StreamDirectX::StreamDirectX(const DeviceDirectX& dev_)
    : dev(dev_),
      fenceEvent(nullptr),
      fenceValue(0),
      recording(false),
      submitted(false) {
  // Create compute command queue
  D3D12_COMMAND_QUEUE_DESC queueDesc = {};
  queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
  queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
  queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
  checkHR(
      dev.device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue)));

  checkHR(dev.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                             IID_PPV_ARGS(&commandAllocator)));

  checkHR(dev.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                        commandAllocator.Get(), nullptr,
                                        IID_PPV_ARGS(&commandList)));

  // Command list starts in recording state; close it initially
  commandList->Close();

  checkHR(
      dev.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
  fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
}

StreamDirectX::~StreamDirectX() {
  if (recording || submitted) {
    try {
      sync();
    } catch (...) {
    }
  }
  cleanupStaging();
  if (fenceEvent) CloseHandle(fenceEvent);
}

void StreamDirectX::begin() {
  if (recording) return;

  // Wait for previous work
  if (submitted) {
    if (fence->GetCompletedValue() < fenceValue) {
      fence->SetEventOnCompletion(fenceValue, fenceEvent);
      WaitForSingleObject(fenceEvent, INFINITE);
    }

    // Process deferred reads
    for (auto& dr : deferredReads) {
      void* mapped = nullptr;
      D3D12_RANGE readRange = {dr.offset, dr.offset + dr.size};
      dr.staging->Map(0, &readRange, &mapped);
      if (mapped) {
        auto* src = static_cast<uint8_t*>(mapped) + dr.offset;
        if (dr.srcRowPitch != 0 && dr.srcRowPitch != dr.dstRowPitch) {
          auto* dstBytes = static_cast<uint8_t*>(dr.dstPtr);
          for (size_t row = 0; row < dr.rowCount; row++) {
            memcpy(dstBytes + row * dr.dstRowPitch, src + row * dr.srcRowPitch,
                   dr.rowBytes);
          }
        } else {
          memcpy(dr.dstPtr, src, dr.size);
        }
        D3D12_RANGE writeRange = {0, 0};
        dr.staging->Unmap(0, &writeRange);
      }
    }
    deferredReads.clear();
    cleanupStaging();
    submitted = false;
  }

  checkHR(commandAllocator->Reset());
  checkHR(commandList->Reset(commandAllocator.Get(), nullptr));
  recording = true;
}

void StreamDirectX::submit() {
  if (!recording) return;

  checkHR(commandList->Close());

  ID3D12CommandList* cmdLists[] = {commandList.Get()};
  commandQueue->ExecuteCommandLists(1, cmdLists);

  fenceValue++;
  checkHR(commandQueue->Signal(fence.Get(), fenceValue));

  recording = false;
  submitted = true;
}

void StreamDirectX::sync() {
  if (recording) submit();
  if (submitted) {
    if (fence->GetCompletedValue() < fenceValue) {
      fence->SetEventOnCompletion(fenceValue, fenceEvent);
      WaitForSingleObject(fenceEvent, INFINITE);
    }

    // Process deferred reads
    for (auto& dr : deferredReads) {
      void* mapped = nullptr;
      D3D12_RANGE readRange = {dr.offset, dr.offset + dr.size};
      dr.staging->Map(0, &readRange, &mapped);
      if (mapped) {
        auto* src = static_cast<uint8_t*>(mapped) + dr.offset;
        if (dr.srcRowPitch != 0 && dr.srcRowPitch != dr.dstRowPitch) {
          auto* dstBytes = static_cast<uint8_t*>(dr.dstPtr);
          for (size_t row = 0; row < dr.rowCount; row++) {
            memcpy(dstBytes + row * dr.dstRowPitch, src + row * dr.srcRowPitch,
                   dr.rowBytes);
          }
        } else {
          memcpy(dr.dstPtr, src, dr.size);
        }
        D3D12_RANGE writeRange = {0, 0};
        dr.staging->Unmap(0, &writeRange);
      }
    }
    deferredReads.clear();
    cleanupStaging();
    submitted = false;
  }
}

std::shared_ptr<Event> StreamDirectX::record() {
  if (recording) submit();

  ComPtr<ID3D12Fence> eventFence;
  checkHR(dev.device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                  IID_PPV_ARGS(&eventFence)));
  HANDLE eventHandle = CreateEvent(nullptr, FALSE, FALSE, nullptr);

  UINT64 value = 1;
  checkHR(commandQueue->Signal(eventFence.Get(), value));

  return std::make_shared<EventDirectX>(eventFence, eventHandle, value);
}

void StreamDirectX::waitForEvent(const std::shared_ptr<Event>& e) {
  auto* evt = static_cast<EventDirectX*>(e.get());
  evt->wait();
}

void StreamDirectX::cleanupStaging() {
  pendingStaging.clear();
  // DeferredRead cleanup handled separately after reads complete
}

// ---------------------------------------------------------------------------
// BufferDirectX
// ---------------------------------------------------------------------------

BufferDirectX::BufferDirectX(const DeviceDirectX& dev_, size_t bytes,
                             const BufferOptions& opts)
    : dev(dev_), _size(bytes), currentState(D3D12_RESOURCE_STATE_COMMON) {
  // Staging routes to UPLOAD (kernel reads / host writes) or READBACK
  // (kernel writes / host reads). Everything else uses DEFAULT (device-local).
  D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_COMMON;
  if (opts.hint == AllocHint::Staging) {
    if (opts.access == Access::WriteOnly) {
      heapType = D3D12_HEAP_TYPE_READBACK;
      initialState = D3D12_RESOURCE_STATE_COPY_DEST;
    } else {
      heapType = D3D12_HEAP_TYPE_UPLOAD;
      initialState = D3D12_RESOURCE_STATE_GENERIC_READ;
    }
    // UPLOAD/READBACK heaps don't permit UAV access.
    flags = D3D12_RESOURCE_FLAG_NONE;
  }
  resource = dev.createCommittedBuffer(bytes, heapType, flags, initialState);
  currentState = initialState;
}

BufferDirectX::BufferDirectX(const DeviceDirectX& dev_,
                             ComPtr<ID3D12Resource> res, size_t bytes,
                             D3D12_RESOURCE_STATES state)
    : dev(dev_), resource(res), _size(bytes), currentState(state) {}

BufferDirectX::~BufferDirectX() {}

size_t BufferDirectX::size() const { return _size; }

void BufferDirectX::transitionTo(ID3D12GraphicsCommandList* cmdList,
                                 D3D12_RESOURCE_STATES newState) {
  if (currentState == newState) return;

  D3D12_RESOURCE_BARRIER barrier = {};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Transition.pResource = resource.Get();
  barrier.Transition.StateBefore = currentState;
  barrier.Transition.StateAfter = newState;
  barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

  cmdList->ResourceBarrier(1, &barrier);
  currentState = newState;
}

void BufferDirectX::copy(const ghost::Stream& s, const ghost::Buffer& src,
                         size_t bytes) {
  copy(s, src, 0, 0, bytes);
}

void BufferDirectX::copy(const ghost::Stream& s, const void* src,
                         size_t bytes) {
  copy(s, src, 0, bytes);
}

void BufferDirectX::copyTo(const ghost::Stream& s, void* dst,
                           size_t bytes) const {
  copyTo(s, dst, 0, bytes);
}

void BufferDirectX::copy(const ghost::Stream& s, const ghost::Buffer& src,
                         size_t srcOffset, size_t dstOffset, size_t bytes) {
  auto& stream = *static_cast<StreamDirectX*>(s.impl().get());
  auto* srcBuf = static_cast<BufferDirectX*>(src.impl().get());

  stream.begin();
  auto* cmdList = stream.commandList.Get();

  srcBuf->transitionTo(cmdList, D3D12_RESOURCE_STATE_COPY_SOURCE);
  transitionTo(cmdList, D3D12_RESOURCE_STATE_COPY_DEST);

  cmdList->CopyBufferRegion(resource.Get(), dstOffset, srcBuf->resource.Get(),
                            srcOffset, bytes);

  transitionTo(cmdList, D3D12_RESOURCE_STATE_COMMON);
  srcBuf->transitionTo(cmdList, D3D12_RESOURCE_STATE_COMMON);
}

void BufferDirectX::copy(const ghost::Stream& s, const void* src,
                         size_t dstOffset, size_t bytes) {
  auto& stream = *static_cast<StreamDirectX*>(s.impl().get());

  // Create upload buffer
  auto uploadBuf = dev.createCommittedBuffer(bytes, D3D12_HEAP_TYPE_UPLOAD,
                                             D3D12_RESOURCE_FLAG_NONE,
                                             D3D12_RESOURCE_STATE_GENERIC_READ);

  // Map and copy data
  void* mapped = nullptr;
  D3D12_RANGE readRange = {0, 0};
  checkHR(uploadBuf->Map(0, &readRange, &mapped));
  memcpy(mapped, src, bytes);
  uploadBuf->Unmap(0, nullptr);

  stream.begin();
  auto* cmdList = stream.commandList.Get();

  transitionTo(cmdList, D3D12_RESOURCE_STATE_COPY_DEST);

  cmdList->CopyBufferRegion(resource.Get(), dstOffset, uploadBuf.Get(), 0,
                            bytes);

  transitionTo(cmdList, D3D12_RESOURCE_STATE_COMMON);

  stream.pendingStaging.push_back({uploadBuf});
}

void BufferDirectX::copyTo(const ghost::Stream& s, void* dst, size_t srcOffset,
                           size_t bytes) const {
  auto& stream = *static_cast<StreamDirectX*>(s.impl().get());

  // Create readback buffer
  auto readbackBuf = dev.createCommittedBuffer(bytes, D3D12_HEAP_TYPE_READBACK,
                                               D3D12_RESOURCE_FLAG_NONE,
                                               D3D12_RESOURCE_STATE_COPY_DEST);

  stream.begin();
  auto* cmdList = stream.commandList.Get();

  const_cast<BufferDirectX*>(this)->transitionTo(
      cmdList, D3D12_RESOURCE_STATE_COPY_SOURCE);

  cmdList->CopyBufferRegion(readbackBuf.Get(), 0, resource.Get(), srcOffset,
                            bytes);

  const_cast<BufferDirectX*>(this)->transitionTo(cmdList,
                                                 D3D12_RESOURCE_STATE_COMMON);

  // Defer the read
  StreamDirectX::DeferredRead dr;
  dr.staging = readbackBuf;
  dr.dstPtr = dst;
  dr.offset = 0;
  dr.size = bytes;
  stream.deferredReads.push_back(dr);
}

void BufferDirectX::fill(const ghost::Stream& s, size_t offset, size_t sz,
                         uint8_t value) {
  // D3D12 doesn't have a direct fill command. Use a CPU-side buffer.
  std::vector<uint8_t> data(sz, value);
  copy(s, data.data(), offset, sz);
}

void BufferDirectX::fill(const ghost::Stream& s, size_t offset, size_t sz,
                         const void* pattern, size_t patternSize) {
  std::vector<uint8_t> data(sz);
  for (size_t i = 0; i < sz; i += patternSize) {
    size_t n = std::min(patternSize, sz - i);
    memcpy(data.data() + i, pattern, n);
  }
  copy(s, data.data(), offset, sz);
}

std::shared_ptr<Buffer> BufferDirectX::createSubBuffer(
    const std::shared_ptr<Buffer>& self, size_t offset, size_t sz) {
  return std::make_shared<SubBufferDirectX>(self, dev, resource,
                                            baseOffset() + offset, sz);
}

// ---------------------------------------------------------------------------
// SubBufferDirectX
// ---------------------------------------------------------------------------

SubBufferDirectX::SubBufferDirectX(std::shared_ptr<Buffer> parent,
                                   const DeviceDirectX& dev_,
                                   ComPtr<ID3D12Resource> res, size_t offset,
                                   size_t bytes)
    : BufferDirectX(dev_, res, bytes, D3D12_RESOURCE_STATE_COMMON),
      _parent(parent),
      _offset(offset) {}

size_t SubBufferDirectX::baseOffset() const {
  return _offset + static_cast<BufferDirectX*>(_parent.get())->baseOffset();
}

void SubBufferDirectX::copy(const ghost::Stream& s, const ghost::Buffer& src,
                            size_t bytes) {
  BufferDirectX::copy(s, src, 0, _offset, bytes);
}

void SubBufferDirectX::copy(const ghost::Stream& s, const void* src,
                            size_t bytes) {
  BufferDirectX::copy(s, src, _offset, bytes);
}

void SubBufferDirectX::copyTo(const ghost::Stream& s, void* dst,
                              size_t bytes) const {
  BufferDirectX::copyTo(s, dst, _offset, bytes);
}

// ---------------------------------------------------------------------------
// MappedBufferDirectX
// ---------------------------------------------------------------------------

MappedBufferDirectX::MappedBufferDirectX(const DeviceDirectX& dev_,
                                         size_t bytes,
                                         const BufferOptions& opts)
    : BufferDirectX(dev_, nullptr, bytes, D3D12_RESOURCE_STATE_COMMON),
      mappedPtr(nullptr) {
  // For WriteOnly access (kernel writes / host reads), use a READBACK heap.
  // Otherwise (ReadOnly or ReadWrite) use an UPLOAD heap.
  D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_UPLOAD;
  D3D12_RESOURCE_STATES initialState = D3D12_RESOURCE_STATE_GENERIC_READ;
  if (opts.access == Access::WriteOnly) {
    heapType = D3D12_HEAP_TYPE_READBACK;
    initialState = D3D12_RESOURCE_STATE_COPY_DEST;
  }
  resource = dev.createCommittedBuffer(bytes, heapType,
                                       D3D12_RESOURCE_FLAG_NONE, initialState);
  currentState = initialState;

  // Persistently map
  D3D12_RANGE readRange = {0, 0};
  checkHR(resource->Map(0, &readRange, &mappedPtr));
}

MappedBufferDirectX::~MappedBufferDirectX() {
  if (mappedPtr && resource) {
    resource->Unmap(0, nullptr);
  }
}

void* MappedBufferDirectX::map(const ghost::Stream& s, Access access,
                               bool doSync) {
  if (doSync) {
    auto& stream = *static_cast<StreamDirectX*>(s.impl().get());
    stream.sync();
  }
  return mappedPtr;
}

void MappedBufferDirectX::unmap(const ghost::Stream& s) {
  // Persistently mapped - no-op
}

// ---------------------------------------------------------------------------
// ImageDirectX
// ---------------------------------------------------------------------------

ImageDirectX::ImageDirectX(const DeviceDirectX& dev_, const ImageDescription& d)
    : dev(dev_), descr(d), currentState(D3D12_RESOURCE_STATE_COMMON) {
  DXGI_FORMAT format = dev.getImageFormat(d);

  D3D12_RESOURCE_DESC resDesc = {};
  resDesc.Dimension = getResourceDimension(d);
  resDesc.Width = (UINT64)d.size.x;
  resDesc.Height = (UINT)std::max(d.size.y, (size_t)1);
  resDesc.DepthOrArraySize = (UINT16)std::max(d.size.z, (size_t)1);
  resDesc.MipLevels = 1;
  resDesc.Format = format;
  resDesc.SampleDesc.Count = 1;
  resDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  resDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

  D3D12_HEAP_PROPERTIES heapProps = {};
  heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

  checkHR(dev.device->CreateCommittedResource(
      &heapProps, D3D12_HEAP_FLAG_NONE, &resDesc, D3D12_RESOURCE_STATE_COMMON,
      nullptr, IID_PPV_ARGS(&resource)));
}

ImageDirectX::ImageDirectX(const DeviceDirectX& dev_, const ImageDescription& d,
                           BufferDirectX& buf)
    : dev(dev_), descr(d), currentState(D3D12_RESOURCE_STATE_COMMON) {
  // In D3D12, we create a placed resource in the same heap if possible.
  // For simplicity, create a new committed resource and note that copies
  // between buffer and image are the expected use pattern.
  DXGI_FORMAT format = dev.getImageFormat(d);

  D3D12_RESOURCE_DESC resDesc = {};
  resDesc.Dimension = getResourceDimension(d);
  resDesc.Width = (UINT64)d.size.x;
  resDesc.Height = (UINT)std::max(d.size.y, (size_t)1);
  resDesc.DepthOrArraySize = (UINT16)std::max(d.size.z, (size_t)1);
  resDesc.MipLevels = 1;
  resDesc.Format = format;
  resDesc.SampleDesc.Count = 1;
  resDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  resDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

  D3D12_HEAP_PROPERTIES heapProps = {};
  heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

  checkHR(dev.device->CreateCommittedResource(
      &heapProps, D3D12_HEAP_FLAG_NONE, &resDesc, D3D12_RESOURCE_STATE_COMMON,
      nullptr, IID_PPV_ARGS(&resource)));
}

ImageDirectX::ImageDirectX(const DeviceDirectX& dev_, const ImageDescription& d,
                           ImageDirectX& other)
    : dev(dev_),
      resource(other.resource),
      descr(d),
      currentState(other.currentState) {}

ImageDirectX::~ImageDirectX() {}

void ImageDirectX::transitionTo(ID3D12GraphicsCommandList* cmdList,
                                D3D12_RESOURCE_STATES newState) {
  if (currentState == newState) return;

  D3D12_RESOURCE_BARRIER barrier = {};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Transition.pResource = resource.Get();
  barrier.Transition.StateBefore = currentState;
  barrier.Transition.StateAfter = newState;
  barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

  cmdList->ResourceBarrier(1, &barrier);
  currentState = newState;
}

void ImageDirectX::copy(const ghost::Stream& s, const ghost::Image& src) {
  auto& stream = *static_cast<StreamDirectX*>(s.impl().get());
  auto* srcImg = static_cast<ImageDirectX*>(src.impl().get());

  stream.begin();
  auto* cmdList = stream.commandList.Get();

  srcImg->transitionTo(cmdList, D3D12_RESOURCE_STATE_COPY_SOURCE);
  transitionTo(cmdList, D3D12_RESOURCE_STATE_COPY_DEST);

  cmdList->CopyResource(resource.Get(), srcImg->resource.Get());

  transitionTo(cmdList, D3D12_RESOURCE_STATE_COMMON);
  srcImg->transitionTo(cmdList, D3D12_RESOURCE_STATE_COMMON);
}

void ImageDirectX::copy(const ghost::Stream& s, const ghost::Image& src,
                        const Size3& region, const Origin3& srcOrigin,
                        const Origin3& dstOrigin) {
  auto& stream = *static_cast<StreamDirectX*>(s.impl().get());
  auto* srcImg = static_cast<ImageDirectX*>(src.impl().get());

  stream.begin();
  auto* cmdList = stream.commandList.Get();

  srcImg->transitionTo(cmdList, D3D12_RESOURCE_STATE_COPY_SOURCE);
  transitionTo(cmdList, D3D12_RESOURCE_STATE_COPY_DEST);

  D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
  srcLoc.pResource = srcImg->resource.Get();
  srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  srcLoc.SubresourceIndex = 0;

  D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
  dstLoc.pResource = resource.Get();
  dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  dstLoc.SubresourceIndex = 0;

  D3D12_BOX srcBox = {};
  srcBox.left = (UINT)srcOrigin.x;
  srcBox.top = (UINT)srcOrigin.y;
  srcBox.front = (UINT)srcOrigin.z;
  srcBox.right = (UINT)(srcOrigin.x + region.x);
  srcBox.bottom = (UINT)(srcOrigin.y + std::max(region.y, (size_t)1));
  srcBox.back = (UINT)(srcOrigin.z + std::max(region.z, (size_t)1));

  cmdList->CopyTextureRegion(&dstLoc, (UINT)dstOrigin.x, (UINT)dstOrigin.y,
                             (UINT)dstOrigin.z, &srcLoc, &srcBox);

  transitionTo(cmdList, D3D12_RESOURCE_STATE_COMMON);
  srcImg->transitionTo(cmdList, D3D12_RESOURCE_STATE_COMMON);
}

void ImageDirectX::copy(const ghost::Stream& s, const ghost::Buffer& src,
                        const BufferLayout& layout) {
  auto& stream = *static_cast<StreamDirectX*>(s.impl().get());
  auto* srcBuf = static_cast<BufferDirectX*>(src.impl().get());

  stream.begin();
  auto* cmdList = stream.commandList.Get();

  srcBuf->transitionTo(cmdList, D3D12_RESOURCE_STATE_COPY_SOURCE);
  transitionTo(cmdList, D3D12_RESOURCE_STATE_COPY_DEST);

  D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
  dstLoc.pResource = resource.Get();
  dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  dstLoc.SubresourceIndex = 0;

  D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
  srcLoc.pResource = srcBuf->resource.Get();
  srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  srcLoc.PlacedFootprint.Offset = srcBuf->baseOffset();
  srcLoc.PlacedFootprint.Footprint.Format = dev.getImageFormat(descr);
  srcLoc.PlacedFootprint.Footprint.Width = (UINT)layout.size.x;
  srcLoc.PlacedFootprint.Footprint.Height =
      (UINT)std::max(layout.size.y, (size_t)1);
  srcLoc.PlacedFootprint.Footprint.Depth =
      (UINT)std::max(layout.size.z, (size_t)1);
  srcLoc.PlacedFootprint.Footprint.RowPitch =
      layout.stride.x > 0 ? (UINT)layout.stride.x
                          : (UINT)(layout.size.x * descr.pixelSize());
  // Align row pitch to 256 bytes
  srcLoc.PlacedFootprint.Footprint.RowPitch =
      (srcLoc.PlacedFootprint.Footprint.RowPitch + 255) & ~255u;

  cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);

  transitionTo(cmdList, D3D12_RESOURCE_STATE_COMMON);
  srcBuf->transitionTo(cmdList, D3D12_RESOURCE_STATE_COMMON);
}

void ImageDirectX::copy(const ghost::Stream& s, const void* src,
                        const BufferLayout& layout) {
  auto& stream = *static_cast<StreamDirectX*>(s.impl().get());

  size_t rowPitch =
      layout.stride.x > 0 ? layout.stride.x : layout.size.x * descr.pixelSize();
  rowPitch = (rowPitch + 255) & ~(size_t)255;  // D3D12 alignment requirement
  size_t height = std::max(layout.size.y, (size_t)1);
  size_t uploadSize = rowPitch * height * std::max(layout.size.z, (size_t)1);

  auto uploadBuf = dev.createCommittedBuffer(uploadSize, D3D12_HEAP_TYPE_UPLOAD,
                                             D3D12_RESOURCE_FLAG_NONE,
                                             D3D12_RESOURCE_STATE_GENERIC_READ);

  // Map and copy row by row (to handle pitch alignment)
  void* mapped = nullptr;
  D3D12_RANGE readRange = {0, 0};
  checkHR(uploadBuf->Map(0, &readRange, &mapped));

  size_t srcRowPitch =
      layout.stride.x > 0 ? layout.stride.x : layout.size.x * descr.pixelSize();
  auto* dstBytes = static_cast<uint8_t*>(mapped);
  auto* srcBytes = static_cast<const uint8_t*>(src);

  for (size_t z = 0; z < std::max(layout.size.z, (size_t)1); z++) {
    for (size_t y = 0; y < height; y++) {
      memcpy(dstBytes + z * rowPitch * height + y * rowPitch,
             srcBytes + z * srcRowPitch * height + y * srcRowPitch,
             layout.size.x * descr.pixelSize());
    }
  }
  uploadBuf->Unmap(0, nullptr);

  stream.begin();
  auto* cmdList = stream.commandList.Get();

  transitionTo(cmdList, D3D12_RESOURCE_STATE_COPY_DEST);

  D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
  dstLoc.pResource = resource.Get();
  dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  dstLoc.SubresourceIndex = 0;

  D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
  srcLoc.pResource = uploadBuf.Get();
  srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  srcLoc.PlacedFootprint.Offset = 0;
  srcLoc.PlacedFootprint.Footprint.Format = dev.getImageFormat(descr);
  srcLoc.PlacedFootprint.Footprint.Width = (UINT)layout.size.x;
  srcLoc.PlacedFootprint.Footprint.Height = (UINT)height;
  srcLoc.PlacedFootprint.Footprint.Depth =
      (UINT)std::max(layout.size.z, (size_t)1);
  srcLoc.PlacedFootprint.Footprint.RowPitch = (UINT)rowPitch;

  cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);

  transitionTo(cmdList, D3D12_RESOURCE_STATE_COMMON);

  stream.pendingStaging.push_back({uploadBuf});
}

void ImageDirectX::copyTo(const ghost::Stream& s, ghost::Buffer& dst,
                          const BufferLayout& layout) const {
  auto& stream = *static_cast<StreamDirectX*>(s.impl().get());
  auto* dstBuf = static_cast<BufferDirectX*>(dst.impl().get());

  stream.begin();
  auto* cmdList = stream.commandList.Get();

  const_cast<ImageDirectX*>(this)->transitionTo(
      cmdList, D3D12_RESOURCE_STATE_COPY_SOURCE);
  dstBuf->transitionTo(cmdList, D3D12_RESOURCE_STATE_COPY_DEST);

  D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
  srcLoc.pResource = resource.Get();
  srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  srcLoc.SubresourceIndex = 0;

  D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
  dstLoc.pResource = dstBuf->resource.Get();
  dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  dstLoc.PlacedFootprint.Offset = dstBuf->baseOffset();
  dstLoc.PlacedFootprint.Footprint.Format = dev.getImageFormat(descr);
  dstLoc.PlacedFootprint.Footprint.Width = (UINT)layout.size.x;
  dstLoc.PlacedFootprint.Footprint.Height =
      (UINT)std::max(layout.size.y, (size_t)1);
  dstLoc.PlacedFootprint.Footprint.Depth =
      (UINT)std::max(layout.size.z, (size_t)1);
  size_t rowPitch =
      layout.stride.x > 0 ? layout.stride.x : layout.size.x * descr.pixelSize();
  dstLoc.PlacedFootprint.Footprint.RowPitch =
      (UINT)((rowPitch + 255) & ~(size_t)255);

  cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);

  const_cast<ImageDirectX*>(this)->transitionTo(cmdList,
                                                D3D12_RESOURCE_STATE_COMMON);
  dstBuf->transitionTo(cmdList, D3D12_RESOURCE_STATE_COMMON);
}

void ImageDirectX::copyTo(const ghost::Stream& s, void* dst,
                          const BufferLayout& layout) const {
  auto& stream = *static_cast<StreamDirectX*>(s.impl().get());

  size_t rowPitch =
      layout.stride.x > 0 ? layout.stride.x : layout.size.x * descr.pixelSize();
  size_t alignedPitch = (rowPitch + 255) & ~(size_t)255;
  size_t height = std::max(layout.size.y, (size_t)1);
  size_t readbackSize =
      alignedPitch * height * std::max(layout.size.z, (size_t)1);

  auto readbackBuf = dev.createCommittedBuffer(
      readbackSize, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
      D3D12_RESOURCE_STATE_COPY_DEST);

  stream.begin();
  auto* cmdList = stream.commandList.Get();

  const_cast<ImageDirectX*>(this)->transitionTo(
      cmdList, D3D12_RESOURCE_STATE_COPY_SOURCE);

  D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
  srcLoc.pResource = resource.Get();
  srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  srcLoc.SubresourceIndex = 0;

  D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
  dstLoc.pResource = readbackBuf.Get();
  dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  dstLoc.PlacedFootprint.Offset = 0;
  dstLoc.PlacedFootprint.Footprint.Format = dev.getImageFormat(descr);
  dstLoc.PlacedFootprint.Footprint.Width = (UINT)layout.size.x;
  dstLoc.PlacedFootprint.Footprint.Height = (UINT)height;
  dstLoc.PlacedFootprint.Footprint.Depth =
      (UINT)std::max(layout.size.z, (size_t)1);
  dstLoc.PlacedFootprint.Footprint.RowPitch = (UINT)alignedPitch;

  cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);

  const_cast<ImageDirectX*>(this)->transitionTo(cmdList,
                                                D3D12_RESOURCE_STATE_COMMON);

  // Deferred read - will copy row-by-row after fence
  StreamDirectX::DeferredRead dr;
  dr.staging = readbackBuf;
  dr.dstPtr = dst;
  dr.offset = 0;
  dr.size = readbackSize;
  dr.srcRowPitch = alignedPitch;
  dr.dstRowPitch = rowPitch;
  dr.rowCount = height * std::max(layout.size.z, (size_t)1);
  dr.rowBytes = layout.size.x * descr.pixelSize();
  stream.deferredReads.push_back(dr);
}

void ImageDirectX::copy(const ghost::Stream& s, const ghost::Buffer& src,
                        const BufferLayout& layout,
                        const Origin3& imageOrigin) {
  auto& stream = *static_cast<StreamDirectX*>(s.impl().get());
  auto* srcBuf = static_cast<BufferDirectX*>(src.impl().get());

  stream.begin();
  auto* cmdList = stream.commandList.Get();

  srcBuf->transitionTo(cmdList, D3D12_RESOURCE_STATE_COPY_SOURCE);
  transitionTo(cmdList, D3D12_RESOURCE_STATE_COPY_DEST);

  D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
  dstLoc.pResource = resource.Get();
  dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  dstLoc.SubresourceIndex = 0;

  D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
  srcLoc.pResource = srcBuf->resource.Get();
  srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  srcLoc.PlacedFootprint.Offset = srcBuf->baseOffset();
  srcLoc.PlacedFootprint.Footprint.Format = dev.getImageFormat(descr);
  srcLoc.PlacedFootprint.Footprint.Width = (UINT)layout.size.x;
  srcLoc.PlacedFootprint.Footprint.Height =
      (UINT)std::max(layout.size.y, (size_t)1);
  srcLoc.PlacedFootprint.Footprint.Depth =
      (UINT)std::max(layout.size.z, (size_t)1);
  srcLoc.PlacedFootprint.Footprint.RowPitch =
      layout.stride.x > 0 ? (UINT)layout.stride.x
                          : (UINT)(layout.size.x * descr.pixelSize());
  srcLoc.PlacedFootprint.Footprint.RowPitch =
      (srcLoc.PlacedFootprint.Footprint.RowPitch + 255) & ~255u;

  cmdList->CopyTextureRegion(&dstLoc, (UINT)imageOrigin.x, (UINT)imageOrigin.y,
                             (UINT)imageOrigin.z, &srcLoc, nullptr);

  transitionTo(cmdList, D3D12_RESOURCE_STATE_COMMON);
  srcBuf->transitionTo(cmdList, D3D12_RESOURCE_STATE_COMMON);
}

void ImageDirectX::copyTo(const ghost::Stream& s, ghost::Buffer& dst,
                          const BufferLayout& layout,
                          const Origin3& imageOrigin) const {
  auto& stream = *static_cast<StreamDirectX*>(s.impl().get());
  auto* dstBuf = static_cast<BufferDirectX*>(dst.impl().get());

  stream.begin();
  auto* cmdList = stream.commandList.Get();

  const_cast<ImageDirectX*>(this)->transitionTo(
      cmdList, D3D12_RESOURCE_STATE_COPY_SOURCE);
  dstBuf->transitionTo(cmdList, D3D12_RESOURCE_STATE_COPY_DEST);

  D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
  srcLoc.pResource = resource.Get();
  srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  srcLoc.SubresourceIndex = 0;

  D3D12_BOX srcBox = {};
  srcBox.left = (UINT)imageOrigin.x;
  srcBox.top = (UINT)imageOrigin.y;
  srcBox.front = (UINT)imageOrigin.z;
  srcBox.right = (UINT)(imageOrigin.x + layout.size.x);
  srcBox.bottom = (UINT)(imageOrigin.y + std::max(layout.size.y, (size_t)1));
  srcBox.back = (UINT)(imageOrigin.z + std::max(layout.size.z, (size_t)1));

  D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
  dstLoc.pResource = dstBuf->resource.Get();
  dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  dstLoc.PlacedFootprint.Offset = dstBuf->baseOffset();
  dstLoc.PlacedFootprint.Footprint.Format = dev.getImageFormat(descr);
  dstLoc.PlacedFootprint.Footprint.Width = (UINT)layout.size.x;
  dstLoc.PlacedFootprint.Footprint.Height =
      (UINT)std::max(layout.size.y, (size_t)1);
  dstLoc.PlacedFootprint.Footprint.Depth =
      (UINT)std::max(layout.size.z, (size_t)1);
  size_t rowPitch =
      layout.stride.x > 0 ? layout.stride.x : layout.size.x * descr.pixelSize();
  dstLoc.PlacedFootprint.Footprint.RowPitch =
      (UINT)((rowPitch + 255) & ~(size_t)255);

  cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, &srcBox);

  const_cast<ImageDirectX*>(this)->transitionTo(cmdList,
                                                D3D12_RESOURCE_STATE_COMMON);
  dstBuf->transitionTo(cmdList, D3D12_RESOURCE_STATE_COMMON);
}

// ---------------------------------------------------------------------------
// DeviceDirectX (implementation)
// ---------------------------------------------------------------------------

DeviceDirectX::DeviceDirectX(const SharedContext& share) : adapterDesc{} {
  if (share.context) {
    // Reuse existing objects
    device = static_cast<ID3D12Device*>(share.context);
    device->AddRef();
    if (share.queue) {
      commandQueue = static_cast<ID3D12CommandQueue*>(share.queue);
      commandQueue->AddRef();
    }
  } else {
    // Create DXGI factory
    checkHR(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));

    // Enumerate adapters and pick the first hardware adapter
    for (UINT i = 0;
         factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++) {
      DXGI_ADAPTER_DESC1 desc;
      adapter->GetDesc1(&desc);
      if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
        adapter.Reset();
        continue;
      }
      // Try to create a D3D12 device
      if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                      IID_PPV_ARGS(&device)))) {
        adapterDesc = desc;
        break;
      }
      adapter.Reset();
    }

    if (!device)
      throw std::runtime_error("No DirectX 12 capable adapter found");

    // Create compute command queue
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    checkHR(
        device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue)));
  }
}

DeviceDirectX::DeviceDirectX(const GpuInfo& info) : adapterDesc{} {
  checkHR(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));

  int adapterIdx = 0;
  for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND;
       i++) {
    DXGI_ADAPTER_DESC1 desc;
    adapter->GetDesc1(&desc);
    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
      adapter.Reset();
      continue;
    }
    if (adapterIdx == info.index) {
      if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                      IID_PPV_ARGS(&device)))) {
        adapterDesc = desc;
        break;
      }
    }
    adapterIdx++;
    adapter.Reset();
  }

  if (!device) throw std::runtime_error("Invalid DirectX device index");

  D3D12_COMMAND_QUEUE_DESC queueDesc = {};
  queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
  queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
  checkHR(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue)));
}

DeviceDirectX::~DeviceDirectX() {}

ComPtr<ID3D12Resource> DeviceDirectX::createCommittedBuffer(
    size_t bytes, D3D12_HEAP_TYPE heapType, D3D12_RESOURCE_FLAGS flags,
    D3D12_RESOURCE_STATES initialState) const {
  D3D12_HEAP_PROPERTIES heapProps = {};
  heapProps.Type = heapType;

  D3D12_RESOURCE_DESC resDesc = {};
  resDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  resDesc.Width = bytes;
  resDesc.Height = 1;
  resDesc.DepthOrArraySize = 1;
  resDesc.MipLevels = 1;
  resDesc.Format = DXGI_FORMAT_UNKNOWN;
  resDesc.SampleDesc.Count = 1;
  resDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  resDesc.Flags = flags;

  ComPtr<ID3D12Resource> resource;
  checkHR(device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE,
                                          &resDesc, initialState, nullptr,
                                          IID_PPV_ARGS(&resource)));
  return resource;
}

DXGI_FORMAT DeviceDirectX::getImageFormat(const ImageDescription& descr) const {
  return ::ghost::implementation::getFormat(descr);
}

ghost::Library DeviceDirectX::loadLibraryFromText(
    const std::string& text, const CompilerOptions& options,
    bool retainBinary) const {
  // DirectX requires pre-compiled DXIL/CSO bytecode; runtime HLSL compilation
  // requires dxcompiler.dll which is not linked by default.
  throw ghost::unsupported_error();
}

ghost::Library DeviceDirectX::loadLibraryFromData(
    const void* data, size_t len, const CompilerOptions& options,
    bool retainBinary) const {
  auto lib = std::make_shared<LibraryDirectX>(*this);
  lib->loadFromData(data, len, options);
  return ghost::Library(lib);
}

SharedContext DeviceDirectX::shareContext() const {
  return SharedContext(device.Get(), commandQueue.Get(), adapter.Get(),
                       factory.Get());
}

ghost::Stream DeviceDirectX::createStream() const {
  return ghost::Stream(std::make_shared<StreamDirectX>(*this));
}

ghost::Buffer DeviceDirectX::allocateBuffer(size_t bytes,
                                            const BufferOptions& opts) const {
  return ghost::Buffer(std::make_shared<BufferDirectX>(*this, bytes, opts));
}

ghost::MappedBuffer DeviceDirectX::allocateMappedBuffer(
    size_t bytes, const BufferOptions& opts) const {
  return ghost::MappedBuffer(
      std::make_shared<MappedBufferDirectX>(*this, bytes, opts));
}

ghost::Image DeviceDirectX::allocateImage(const ImageDescription& descr) const {
  return ghost::Image(std::make_shared<ImageDirectX>(*this, descr));
}

ghost::Image DeviceDirectX::sharedImage(const ImageDescription& descr,
                                        ghost::Buffer& buffer) const {
  throw ghost::unsupported_error();
}

ghost::Image DeviceDirectX::sharedImage(const ImageDescription& descr,
                                        ghost::Image& image) const {
  auto* dxImg = static_cast<ImageDirectX*>(image.impl().get());
  return ghost::Image(std::make_shared<ImageDirectX>(*this, descr, *dxImg));
}

Attribute DeviceDirectX::getAttribute(DeviceAttributeId what) const {
  switch (what) {
    case kDeviceImplementation:
      return Attribute("DirectX");
    case kDeviceName: {
      // Convert wide string to narrow string
      char name[256];
      size_t len;
      wcstombs_s(&len, name, sizeof(name), adapterDesc.Description,
                 sizeof(name) - 1);
      return Attribute(std::string(name));
    }
    case kDeviceVendor: {
      switch (adapterDesc.VendorId) {
        case 0x1002:
          return Attribute("AMD");
        case 0x10DE:
          return Attribute("NVIDIA");
        case 0x8086:
          return Attribute("Intel");
        case 0x1414:
          return Attribute("Microsoft");
        default:
          return Attribute("Unknown");
      }
    }
    case kDeviceDriverVersion:
      return Attribute("DirectX 12");
    case kDeviceFamily:
      return Attribute("12.0");
    case kDeviceProcessorCount:
      return Attribute((int32_t)0);
    case kDeviceUnifiedMemory:
      return Attribute(adapterDesc.DedicatedVideoMemory == 0);
    case kDeviceMemory:
      return Attribute((uint64_t)adapterDesc.DedicatedVideoMemory);
    case kDeviceLocalMemory:
      return Attribute((int32_t)(32 * 1024));  // 32 KB typical
    case kDeviceMaxThreads:
      return Attribute((int32_t)1024);
    case kDeviceMaxWorkSize:
      return Attribute((int32_t)1024, (int32_t)1024, (int32_t)64);
    case kDeviceMaxRegisters:
      return Attribute((int32_t)0);
    case kDeviceMaxImageSize1:
      return Attribute((int32_t)16384);
    case kDeviceMaxImageSize2:
      return Attribute((int32_t)16384, (int32_t)16384);
    case kDeviceMaxImageSize3:
      return Attribute((int32_t)2048, (int32_t)2048, (int32_t)2048);
    case kDeviceImageAlignment:
      return Attribute((int32_t)256);
    case kDeviceSupportsImageIntegerFiltering:
      return Attribute(true);
    case kDeviceSupportsImageFloatFiltering:
      return Attribute(true);
    case kDeviceSupportsMappedBuffer:
      return Attribute(true);
    case kDeviceSupportsProgramConstants:
      return Attribute(false);
    case kDeviceSupportsSubgroup:
      return Attribute(true);
    case kDeviceSupportsSubgroupShuffle:
      return Attribute(true);
    case kDeviceSubgroupWidth:
      return Attribute((int32_t)32);
    case kDeviceMaxComputeUnits:
      // Not directly queryable in D3D12; return 1 as the minimum
      return Attribute((int32_t)1);
    case kDeviceMemoryAlignment:
      return Attribute((int32_t)256);
    case kDeviceBufferAlignment:
      return Attribute((int32_t)256);
    case kDeviceMaxBufferSize:
      return Attribute((uint64_t)adapterDesc.DedicatedVideoMemory);
    case kDeviceMaxConstantBufferSize:
      return Attribute((int32_t)(128));  // 128 bytes root constants
    case kDeviceTimestampPeriod:
      return Attribute(1.0f);
    case kDeviceSupportsProfilingTimer:
      return Attribute(true);
    default:
      return Attribute();
  }
}

}  // namespace implementation

// ---------------------------------------------------------------------------
// Public DeviceDirectX
// ---------------------------------------------------------------------------

DeviceDirectX::DeviceDirectX(const SharedContext& share)
    : Device(std::make_shared<implementation::DeviceDirectX>(share)) {
  setDefaultStream(std::make_shared<implementation::StreamDirectX>(
      *static_cast<implementation::DeviceDirectX*>(impl().get())));
}

DeviceDirectX::DeviceDirectX(const GpuInfo& info)
    : Device(std::make_shared<implementation::DeviceDirectX>(info)) {
  setDefaultStream(std::make_shared<implementation::StreamDirectX>(
      *static_cast<implementation::DeviceDirectX*>(impl().get())));
}

std::vector<GpuInfo> DeviceDirectX::enumerateDevices() {
  std::vector<GpuInfo> result;

  ComPtr<IDXGIFactory4> dxgiFactory;
  if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)))) return result;

  ComPtr<IDXGIAdapter1> dxgiAdapter;
  int index = 0;
  for (UINT i = 0;
       dxgiFactory->EnumAdapters1(i, &dxgiAdapter) != DXGI_ERROR_NOT_FOUND;
       i++) {
    DXGI_ADAPTER_DESC1 desc;
    dxgiAdapter->GetDesc1(&desc);

    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
      dxgiAdapter.Reset();
      continue;
    }

    // Check D3D12 support
    ComPtr<ID3D12Device> testDevice;
    if (FAILED(D3D12CreateDevice(dxgiAdapter.Get(), D3D_FEATURE_LEVEL_12_0,
                                 IID_PPV_ARGS(&testDevice)))) {
      dxgiAdapter.Reset();
      continue;
    }

    GpuInfo info;
    char name[256];
    size_t len;
    wcstombs_s(&len, name, sizeof(name), desc.Description, sizeof(name) - 1);
    info.name = name;
    info.implementation = "DirectX";
    info.index = index++;
    info.memory = desc.DedicatedVideoMemory;
    info.unifiedMemory = (desc.DedicatedVideoMemory == 0);

    switch (desc.VendorId) {
      case 0x1002:
        info.vendor = "AMD";
        break;
      case 0x10DE:
        info.vendor = "NVIDIA";
        break;
      case 0x8086:
        info.vendor = "Intel";
        break;
      case 0x1414:
        info.vendor = "Microsoft";
        break;
      default:
        info.vendor = "Unknown";
        break;
    }

    result.push_back(info);
    dxgiAdapter.Reset();
  }

  return result;
}

}  // namespace ghost

#endif
