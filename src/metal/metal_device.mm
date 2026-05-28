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

#if WITH_METAL

#include <ghost/allocator.h>
#include <ghost/argument_buffer.h>
#include <ghost/metal/device.h>
#include <ghost/metal/exception.h>
#include <ghost/metal/impl_device.h>
#include <ghost/metal/impl_function.h>

#include <sys/sysctl.h>
#include <sys/types.h>

#include <TargetConditionals.h>

#include <unordered_set>

namespace ghost {
namespace implementation {
using namespace metal;

#if !defined(MAC_OS_VERSION_26_0)
// SDK is too old for MTLGPUFamilyApple10
#define MTLGPUFamilyApple10 (MTLGPUFamily)1010
#endif

namespace {
MTLPixelFormat getFormat(const ImageDescription &descr) {
  switch (descr.channels) {
  case 1:
    switch (descr.type) {
    case DataType_Float16:
      return MTLPixelFormatR16Float;
    case DataType_Float:
      return MTLPixelFormatR32Float;
    case DataType_UInt16:
      return MTLPixelFormatR16Unorm;
    case DataType_Int16:
      return MTLPixelFormatR16Snorm;
    case DataType_Int8:
      return MTLPixelFormatR8Snorm;
    case DataType_UInt8:
    default:
      return MTLPixelFormatR8Unorm;
    }
  case 2:
    switch (descr.type) {
    case DataType_Float16:
      return MTLPixelFormatRG16Float;
    case DataType_Float:
      return MTLPixelFormatRG32Float;
    case DataType_UInt16:
      return MTLPixelFormatRG16Unorm;
    case DataType_Int16:
      return MTLPixelFormatRG16Snorm;
    case DataType_Int8:
      return MTLPixelFormatRG8Snorm;
    case DataType_UInt8:
    default:
      return MTLPixelFormatRG8Unorm;
    }
  default:
    if (descr.order == PixelOrder_BGRA && descr.type == DataType_UInt8) {
      return MTLPixelFormatBGRA8Unorm;
    } else if (descr.order != PixelOrder_RGBA) {
      throw ghost::unsupported_error();
    }
    switch (descr.type) {
    case DataType_Float16:
      return MTLPixelFormatRGBA16Float;
    case DataType_Float:
      return MTLPixelFormatRGBA32Float;
    case DataType_UInt16:
      return MTLPixelFormatRGBA16Unorm;
    case DataType_Int16:
      return MTLPixelFormatRGBA16Snorm;
    case DataType_Int8:
      return MTLPixelFormatRGBA8Snorm;
    case DataType_UInt8:
    default:
      return MTLPixelFormatRGBA8Unorm;
    }
  }
}

objc::ptr<MTLTextureDescriptor *>
getTextureDescriptor(const ImageDescription &descr, bool isPrivate = true) {
  objc::ptr<MTLTextureDescriptor *> d;
  d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:getFormat(descr)
                                                         width:descr.size.x
                                                        height:descr.size.y
                                                     mipmapped:NO];
  if (descr.size.z > 1) {
    d.get().textureType = MTLTextureType3D;
    d.get().depth = descr.size.z;
  }
  switch (descr.access) {
  case Access::ReadOnly:
    d.get().usage = MTLTextureUsageShaderRead;
    break;
  case Access::WriteOnly:
    d.get().usage = MTLTextureUsageShaderWrite;
    break;
  case Access::ReadWrite:
    d.get().usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    break;
  default:
    d.get().usage = MTLTextureUsageUnknown;
    break;
  }
  if (isPrivate)
    d.get().resourceOptions = MTLResourceStorageModePrivate;
  // d.get().resourceOptions |= MTLResourceHazardTrackingModeUntracked;
  return d;
}

objc::ptr<MTLTextureDescriptor *>
getTextureDescriptor(id<MTLResource> resource, const ImageDescription &descr) {
  objc::ptr<MTLTextureDescriptor *> d;
  d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:getFormat(descr)
                                                         width:descr.size.x
                                                        height:descr.size.y
                                                     mipmapped:NO];
  if (descr.size.z > 1) {
    d.get().textureType = MTLTextureType3D;
    d.get().depth = descr.size.z;
  }
  switch (descr.access) {
  case Access::ReadOnly:
    d.get().usage = MTLTextureUsageShaderRead;
    break;
  case Access::WriteOnly:
    d.get().usage = MTLTextureUsageShaderWrite;
    break;
  case Access::ReadWrite:
    d.get().usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    break;
  default:
    d.get().usage = MTLTextureUsageUnknown;
    break;
  }
  if (@available(macOS 11.0, iOS 13.0, *)) {
    d.get().resourceOptions = resource.resourceOptions;
  } else {
    d.get().cpuCacheMode = resource.cpuCacheMode;
    d.get().storageMode = resource.storageMode;
    // hazardTrackingMode is not yet available in this OS.
  }
  return d;
}

static bool IsPrivate(id<MTLResource> res) {
  switch (res.storageMode) {
  case MTLStorageModeShared:
  case MTLStorageModeManaged:
    return false;
  default:
    return true;
  }
}
} // namespace

EventMetal::EventMetal(objc::ptr<id<MTLSharedEvent>> event_, uint64_t value)
    : sharedEvent(event_), targetValue(value) {}

void EventMetal::wait() {
  if (sharedEvent.get().signaledValue >= targetValue)
    return;
  dispatch_semaphore_t sem = dispatch_semaphore_create(0);
  MTLSharedEventListener *listener = [[MTLSharedEventListener alloc] init];
  [sharedEvent.get()
      notifyListener:listener
             atValue:targetValue
               block:^(id<MTLSharedEvent> event, uint64_t value) {
                 dispatch_semaphore_signal(sem);
               }];
  dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
}

bool EventMetal::isComplete() const {
  return sharedEvent.get().signaledValue >= targetValue;
}

// ---------------------------------------------------------------------------
// MetalEncoder
// ---------------------------------------------------------------------------

id<MTLBlitCommandEncoder> MetalEncoder::getBlitEncoder() {
  begin();
  if (blitEncoder.get())
    return blitEncoder.get();
  bool hadPriorEncoder = pendingFenceWait;
  if (computeEncoder.get()) {
    [computeEncoder.get() updateFence:fence.get()];
    [computeEncoder.get() endEncoding];
    computeEncoder = objc::ptr<id<MTLComputeCommandEncoder>>();
    hadPriorEncoder = true;
  }
  blitEncoder = [commandBuffer.get() blitCommandEncoder];
  if (hadPriorEncoder)
    [blitEncoder.get() waitForFence:fence.get()];
  pendingFenceWait = false;
  return blitEncoder.get();
}

id<MTLComputeCommandEncoder> MetalEncoder::getComputeEncoder() {
  begin();
  if (computeEncoder.get())
    return computeEncoder.get();
  bool hadPriorEncoder = pendingFenceWait;
  if (blitEncoder.get()) {
    [blitEncoder.get() updateFence:fence.get()];
    [blitEncoder.get() endEncoding];
    blitEncoder = objc::ptr<id<MTLBlitCommandEncoder>>();
    hadPriorEncoder = true;
  }
  if (concurrent) {
    if (@available(macOS 10.14, iOS 12.0, tvOS 12.0, macCatalyst 13.0, *)) {
      computeEncoder = [commandBuffer.get()
          computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    }
  }
  if (!computeEncoder.get()) {
    computeEncoder = [commandBuffer.get() computeCommandEncoder];
  }
  if (hadPriorEncoder)
    [computeEncoder.get() waitForFence:fence.get()];
  pendingFenceWait = false;
  return computeEncoder.get();
}

void MetalEncoder::endEncoding() {
  // Resources are MTLHazardTrackingModeUntracked, so updating the fence on
  // the encoder being closed is the only way to make a subsequent encoder
  // wait on its writes. Without this, BarrierCmd between two same-type
  // (e.g. compute -> compute) encoders would race: the second encoder
  // creates fresh without observing the first's fence signal.
  if (blitEncoder.get()) {
    [blitEncoder.get() updateFence:fence.get()];
    [blitEncoder.get() endEncoding];
    blitEncoder = objc::ptr<id<MTLBlitCommandEncoder>>();
    pendingFenceWait = true;
  }
  if (computeEncoder.get()) {
    [computeEncoder.get() updateFence:fence.get()];
    [computeEncoder.get() endEncoding];
    computeEncoder = objc::ptr<id<MTLComputeCommandEncoder>>();
    pendingFenceWait = true;
  }
}

MetalEncoder &metalEncoder(const ghost::Encoder &s) {
  auto *p = dynamic_cast<MetalEncoder *>(s.impl().get());
  if (!p)
    throw ghost::unsupported_error();
  return *p;
}

// ---------------------------------------------------------------------------
// StreamMetal
// ---------------------------------------------------------------------------

StreamMetal::StreamMetal(objc::ptr<id<MTLCommandQueue>> queue_,
                         const StreamOptions &options)
    : queue(queue_) {
  concurrent = options.concurrent;
  if (queue.get()) {
    syncEvent = [queue.get().device newEvent];
  }
}

StreamMetal::StreamMetal(id<MTLDevice> dev, const StreamOptions &options) {
  concurrent = options.concurrent;
  queue = [dev newCommandQueue];
  checkExists(queue);
  syncEvent = [dev newEvent];
}

void StreamMetal::begin() {
  if (commandBuffer.get())
    return;
  commandBuffer = [queue.get() commandBuffer];
  commandBuffer.get().label = @"Ghost";
  fence = [queue.get().device newFence];
  if (syncCounter > 0) {
    [commandBuffer.get() encodeWaitForEvent:syncEvent.get() value:syncCounter];
  }
}

void StreamMetal::commit() {
  if (!commandBuffer.get())
    return;
  endEncoding();
  syncCounter++;
  [commandBuffer.get() encodeSignalEvent:syncEvent.get() value:syncCounter];
  [commandBuffer.get() commit];
  _lastCommitted = commandBuffer;
  commandBuffer = objc::ptr<id<MTLCommandBuffer>>();
  fence = objc::ptr<id<MTLFence>>();
}

void StreamMetal::attachCommitted(objc::ptr<id<MTLCommandBuffer>> cb) {
  _lastCommitted = std::move(cb);
}

void StreamMetal::sync() {
  if (commandBuffer.get())
    commit();
  if (_lastCommitted.get()) {
    [_lastCommitted.get() waitUntilCompleted];
    _lastCommitted = objc::ptr<id<MTLCommandBuffer>>();
  }
}

std::shared_ptr<Event> StreamMetal::record() {
  begin();
  endEncoding();
  id<MTLDevice> device = queue.get().device;
  objc::ptr<id<MTLSharedEvent>> sharedEvent =
      objc::ptr<id<MTLSharedEvent>>([device newSharedEvent]);
  uint64_t value = 1;
  [commandBuffer.get() encodeSignalEvent:sharedEvent.get() value:value];
  commit();
  return std::make_shared<EventMetal>(sharedEvent, value);
}

void StreamMetal::waitForEvent(const std::shared_ptr<Event> &e) {
  auto eventMetal = static_cast<EventMetal *>(e.get());
  begin();
  endEncoding();
  [commandBuffer.get() encodeWaitForEvent:eventMetal->sharedEvent.get()
                                    value:eventMetal->targetValue];
}

BufferMetal::BufferMetal(objc::ptr<id<MTLBuffer>> mem_, size_t bytes)
    : mem(mem_), _size(bytes) {}

BufferMetal::~BufferMetal() {
  if (_allocator) {
    void *handle = GHOST_OBJC_BRIDGE_RETAINED(void *, mem.release());
    _allocator->freeBuffer(handle, _size);
  }
  // else: objc::ptr destructor releases mem normally
}

BufferMetal::BufferMetal(const DeviceMetal &dev, size_t bytes,
                         const BufferOptions &opts)
    : _size(bytes) {
  MTLResourceOptions options = MTLResourceCPUCacheModeDefaultCache |
                               MTLResourceHazardTrackingModeUntracked;
  // Staging hint routes to a host-visible shared buffer. Other hints
  // currently fall through to private storage (the default fast path).
  if (opts.hint == AllocHint::Staging) {
    options |= MTLResourceStorageModeShared;
    if (opts.access == Access::ReadOnly) {
      // Host writes, kernel reads — write-combined is efficient for upload.
      options |= MTLResourceCPUCacheModeWriteCombined;
    }
  } else {
    options |= MTLResourceStorageModePrivate;
  }
  mem = [dev.dev newBufferWithLength:bytes options:options];
  checkExists(mem);
}

size_t BufferMetal::size() const { return _size; }

void BufferMetal::copy(const ghost::Encoder &s, const ghost::Buffer &src,
                       size_t bytes) {
  copy(s, src, 0, 0, bytes);
}

void BufferMetal::copy(const ghost::Encoder &s, const void *src, size_t bytes) {
  copy(s, src, 0, bytes);
}

void BufferMetal::copyTo(const ghost::Encoder &s, void *dst,
                         size_t bytes) const {
  copyTo(s, dst, 0, bytes);
}

void BufferMetal::copy(const ghost::Encoder &s, const ghost::Buffer &src,
                       size_t srcOffset, size_t dstOffset, size_t bytes) {
  auto &enc = metalEncoder(s);
  auto src_impl = static_cast<implementation::BufferMetal *>(src.impl().get());
  size_t effectiveSrcOff = src_impl->baseOffset() + srcOffset;
  size_t effectiveDstOff = baseOffset() + dstOffset;

  // Metal's copyFromBuffer has undefined behavior when source and destination
  // regions overlap within the same buffer. Use a staging buffer in that case.
  bool sameBuffer = (src_impl->mem.get() == mem.get());
  bool overlaps = sameBuffer && (effectiveSrcOff < effectiveDstOff + bytes) &&
                  (effectiveDstOff < effectiveSrcOff + bytes);

  id<MTLBlitCommandEncoder> blit = enc.getBlitEncoder();
  if (overlaps) {
    id<MTLDevice> device = mem.get().device;
    id<MTLBuffer> staging =
        [device newBufferWithLength:bytes options:mem.get().resourceOptions];
    [blit copyFromBuffer:src_impl->mem.get()
             sourceOffset:effectiveSrcOff
                 toBuffer:staging
        destinationOffset:0
                     size:bytes];
    [blit copyFromBuffer:staging
             sourceOffset:0
                 toBuffer:mem.get()
        destinationOffset:effectiveDstOff
                     size:bytes];
  } else {
    [blit copyFromBuffer:src_impl->mem.get()
             sourceOffset:effectiveSrcOff
                 toBuffer:mem.get()
        destinationOffset:effectiveDstOff
                     size:bytes];
  }
}

void BufferMetal::copy(const ghost::Encoder &s, const void *src,
                       size_t dstOffset, size_t bytes) {
  size_t effectiveDstOff = baseOffset() + dstOffset;
  if (IsPrivate(mem)) {
    // For private buffers, use a temporary shared buffer and blit
    auto &enc = metalEncoder(s);
    id<MTLBuffer> staging =
        [mem.get().device newBufferWithBytes:src
                                      length:bytes
                                     options:MTLResourceStorageModeShared];
    id<MTLBlitCommandEncoder> blit = enc.getBlitEncoder();
    [blit copyFromBuffer:staging
             sourceOffset:0
                 toBuffer:mem.get()
        destinationOffset:effectiveDstOff
                     size:bytes];
  } else {
    void *dst = static_cast<uint8_t *>([mem contents]) + effectiveDstOff;
    memcpy(dst, src, bytes);
    if (mem.get().storageMode == MTLStorageModeManaged)
      [mem didModifyRange:NSMakeRange(effectiveDstOff, bytes)];
  }
}

void BufferMetal::copyTo(const ghost::Encoder &s, void *dst, size_t srcOffset,
                         size_t bytes) const {
  size_t effectiveSrcOff = baseOffset() + srcOffset;
  if (mem.get().storageMode == MTLStorageModePrivate) {
    // Private buffers: blit to a shared staging buffer, drain, then memcpy.
    auto *stream_impl =
        dynamic_cast<implementation::StreamMetal *>(s.impl().get());
    if (!stream_impl)
      throw ghost::unsupported_error();
    id<MTLBuffer> staging =
        [mem.get().device newBufferWithLength:bytes
                                      options:MTLResourceStorageModeShared];
    id<MTLBlitCommandEncoder> blit = stream_impl->getBlitEncoder();
    [blit copyFromBuffer:mem.get()
             sourceOffset:effectiveSrcOff
                 toBuffer:staging
        destinationOffset:0
                     size:bytes];
    stream_impl->sync();
    memcpy(dst, [staging contents], bytes);
  } else if (mem.get().storageMode == MTLStorageModeManaged) {
    auto *stream_impl =
        dynamic_cast<implementation::StreamMetal *>(s.impl().get());
    if (!stream_impl)
      throw ghost::unsupported_error();
    id<MTLBlitCommandEncoder> blit = stream_impl->getBlitEncoder();
    [blit synchronizeResource:mem];
    stream_impl->sync();
    const void *src =
        static_cast<const uint8_t *>([mem contents]) + effectiveSrcOff;
    memcpy(dst, src, bytes);
  } else {
    const void *src =
        static_cast<const uint8_t *>([mem contents]) + effectiveSrcOff;
    memcpy(dst, src, bytes);
  }
}

void BufferMetal::fill(const ghost::Encoder &s, size_t offset, size_t size,
                       uint8_t value) {
  auto &enc = metalEncoder(s);
  size_t effectiveOff = baseOffset() + offset;
  id<MTLBlitCommandEncoder> blit = enc.getBlitEncoder();
  [blit fillBuffer:mem.get() range:NSMakeRange(effectiveOff, size) value:value];
}

std::shared_ptr<Buffer>
BufferMetal::createSubBuffer(const std::shared_ptr<Buffer> &self, size_t offset,
                             size_t size) {
  return std::make_shared<SubBufferMetal>(self, mem, baseOffset() + offset,
                                          size);
}

SubBufferMetal::SubBufferMetal(std::shared_ptr<Buffer> parent,
                               objc::ptr<id<MTLBuffer>> mem_, size_t offset,
                               size_t size)
    : BufferMetal(mem_, size), _parent(parent), _offset(offset) {}

size_t SubBufferMetal::baseOffset() const { return _offset; }

void SubBufferMetal::copy(const ghost::Encoder &s, const ghost::Buffer &src,
                          size_t bytes) {
  BufferMetal::copy(s, src, 0, 0, bytes);
}

void SubBufferMetal::copy(const ghost::Encoder &s, const void *src,
                          size_t bytes) {
  BufferMetal::copy(s, src, 0, bytes);
}

void SubBufferMetal::copyTo(const ghost::Encoder &s, void *dst,
                            size_t bytes) const {
  BufferMetal::copyTo(s, dst, 0, bytes);
}

void BufferMetal::fill(const ghost::Encoder &s, size_t offset, size_t size,
                       const void *pattern, size_t patternSize) {
  // Metal only supports single-byte fill natively.
  // For multi-byte patterns, build a staging buffer then blit.
  if (patternSize == 1) {
    fill(s, offset, size, *static_cast<const uint8_t *>(pattern));
    return;
  }
  auto &enc = metalEncoder(s);
  size_t effectiveOff = baseOffset() + offset;
  id<MTLBuffer> staging =
      [mem.get().device newBufferWithLength:size
                                    options:MTLResourceStorageModeShared];
  uint8_t *dst = static_cast<uint8_t *>([staging contents]);
  for (size_t i = 0; i < size; i += patternSize) {
    size_t n = std::min(patternSize, size - i);
    memcpy(dst + i, pattern, n);
  }
  id<MTLBlitCommandEncoder> blit = enc.getBlitEncoder();
  [blit copyFromBuffer:staging
           sourceOffset:0
               toBuffer:mem.get()
      destinationOffset:effectiveOff
                   size:size];
}

MappedBufferMetal::MappedBufferMetal(objc::ptr<id<MTLBuffer>> mem_,
                                     size_t bytes)
    : BufferMetal(mem_, bytes), length(bytes) {}

MappedBufferMetal::~MappedBufferMetal() {
  if (_allocator) {
    void *handle = GHOST_OBJC_BRIDGE_RETAINED(void *, mem.release());
    _allocator->freeMappedBuffer(handle, _size);
    _allocator = nullptr; // suppress base BufferMetal destructor
  }
}

MappedBufferMetal::MappedBufferMetal(const DeviceMetal &dev, size_t bytes,
                                     const BufferOptions &opts)
    : BufferMetal(objc::ptr<id<MTLBuffer>>(), bytes), length(bytes) {
  MTLResourceOptions options = MTLResourceCPUCacheModeDefaultCache |
                               MTLResourceHazardTrackingModeUntracked |
                               MTLResourceStorageModeShared;
  // WriteOnly / Staging+ReadOnly both mean host-write / kernel-read: use
  // write-combined cache mode for faster upload.
  if (opts.access == Access::WriteOnly ||
      (opts.hint == AllocHint::Staging && opts.access == Access::ReadOnly))
    options |= MTLResourceCPUCacheModeWriteCombined;
  mem = [dev.dev newBufferWithLength:bytes options:options];
  checkExists(mem);
}

void *MappedBufferMetal::map(const ghost::Encoder &s, Access access,
                             bool sync) {
  if (sync) {
    auto *stream_impl =
        dynamic_cast<implementation::StreamMetal *>(s.impl().get());
    if (!stream_impl)
      throw ghost::unsupported_error();
    stream_impl->sync();
  }
  return [mem contents];
}

void MappedBufferMetal::unmap(const ghost::Encoder &s) {
  [mem didModifyRange:NSMakeRange(0, length)];
}

ImageMetal::ImageMetal(objc::ptr<id<MTLTexture>> mem_,
                       const ImageDescription &descr_)
    : mem(mem_), descr(descr_) {}

ImageMetal::~ImageMetal() {
  if (_allocator) {
    void *handle = GHOST_OBJC_BRIDGE_RETAINED(void *, mem.release());
    _allocator->freeImage(handle, descr);
  }
}

ImageMetal::ImageMetal(const DeviceMetal &dev, const ImageDescription &descr_)
    : descr(descr_) {
  objc::ptr<MTLTextureDescriptor *> textureDescriptor(
      getTextureDescriptor(descr));
  mem = [dev.dev newTextureWithDescriptor:textureDescriptor.get()];
}

ImageMetal::ImageMetal(const DeviceMetal &dev, const ImageDescription &descr_,
                       BufferMetal &buffer)
    : descr(descr_) {
  // newTextureWithDescriptor:offset:bytesPerRow: is not supported on buffers
  // allocated from an MTLHeap. Allocate the buffer with AllocHint::Shared to
  // bypass the heap.
  if ([buffer.mem.get() heap] != nil) {
    throw ghost::unsupported_error(
        "Cannot create shared image from a heap-allocated buffer. "
        "Use AllocHint::Shared when allocating the buffer.");
  }

  // Validate stride: bytesPerRow must be at least width * pixelSize and
  // aligned to the device's minimum linear texture alignment.
  size_t minRowBytes = descr.size.x * descr.pixelSize();
  size_t bytesPerRow = static_cast<size_t>(descr.stride.x);
  if (bytesPerRow == 0) {
    bytesPerRow = minRowBytes;
  }
  if (bytesPerRow < minRowBytes) {
    throw std::invalid_argument("bytesPerRow (" + std::to_string(bytesPerRow) +
                                ") is less than width * pixelSize (" +
                                std::to_string(minRowBytes) + ")");
  }
  size_t requiredBytes = bytesPerRow * descr.size.y;
  if (buffer.size() < requiredBytes) {
    throw std::invalid_argument(
        "buffer size (" + std::to_string(buffer.size()) +
        ") is less than required (" + std::to_string(requiredBytes) + ")");
  }

  objc::ptr<MTLTextureDescriptor *> textureDescriptor(
      getTextureDescriptor(buffer.mem.get(), descr));
  mem = [buffer.mem.get() newTextureWithDescriptor:textureDescriptor.get()
                                            offset:0
                                       bytesPerRow:bytesPerRow];
}

ImageMetal::ImageMetal(const DeviceMetal &dev, const ImageDescription &descr_,
                       ImageMetal &image)
    : mem(image.mem), descr(descr_) {
  mem = [image.mem.get() newTextureViewWithPixelFormat:getFormat(descr)];
}

void ImageMetal::copy(const ghost::Encoder &s, const ghost::Image &src) {
  auto &enc = metalEncoder(s);
  auto src_impl = static_cast<implementation::ImageMetal *>(src.impl().get());
  MTLRegion region = {{0, 0, 0}, {descr.size.x, descr.size.y, descr.size.z}};
  MTLOrigin dst_origin = {0, 0, 0};
  id<MTLBlitCommandEncoder> blit = enc.getBlitEncoder();
  [blit copyFromTexture:src_impl->mem.get()
            sourceSlice:0
            sourceLevel:0
           sourceOrigin:region.origin
             sourceSize:region.size
              toTexture:mem.get()
       destinationSlice:0
       destinationLevel:0
      destinationOrigin:dst_origin];
}

void ImageMetal::copy(const ghost::Encoder &s, const ghost::Buffer &src,
                      const BufferLayout &layout) {
  auto &enc = metalEncoder(s);
  auto src_impl = static_cast<implementation::BufferMetal *>(src.impl().get());
  size_t rowStride = layout.rowBytes(descr.pixelSize());
  size_t sliceStride = layout.sliceBytes(rowStride);
  MTLRegion region = {{0, 0, 0}, {descr.size.x, descr.size.y, descr.size.z}};
  MTLOrigin dst_origin = {0, 0, 0};
  id<MTLBlitCommandEncoder> blit = enc.getBlitEncoder();
  [blit copyFromBuffer:src_impl->mem.get()
             sourceOffset:src_impl->baseOffset()
        sourceBytesPerRow:rowStride
      sourceBytesPerImage:sliceStride
               sourceSize:region.size
                toTexture:mem.get()
         destinationSlice:0
         destinationLevel:0
        destinationOrigin:dst_origin];
}

void ImageMetal::copy(const ghost::Encoder &s, const void *src,
                      const BufferLayout &layout) {
  size_t rowStride = layout.rowBytes(descr.pixelSize());
  size_t sliceStride = layout.sliceBytes(rowStride);
  if (!IsPrivate(mem.get())) {
    MTLRegion region = {{0, 0, 0},
                        {layout.size.x, layout.size.y, layout.size.z}};
    [mem.get() replaceRegion:region
                 mipmapLevel:0
                   withBytes:src
                 bytesPerRow:rowStride];
    return;
  }
  // Private texture: upload via staging buffer + blit
  auto &enc = metalEncoder(s);
  size_t dataSize = sliceStride * layout.size.z;
  id<MTLBuffer> staging =
      [mem.get().device newBufferWithBytes:src
                                    length:dataSize
                                   options:MTLResourceStorageModeShared];
  MTLRegion region = {{0, 0, 0}, {layout.size.x, layout.size.y, layout.size.z}};
  MTLOrigin dst_origin = {0, 0, 0};
  id<MTLBlitCommandEncoder> blit = enc.getBlitEncoder();
  [blit copyFromBuffer:staging
             sourceOffset:0
        sourceBytesPerRow:rowStride
      sourceBytesPerImage:sliceStride
               sourceSize:region.size
                toTexture:mem.get()
         destinationSlice:0
         destinationLevel:0
        destinationOrigin:dst_origin];
}

void ImageMetal::copyTo(const ghost::Encoder &s, ghost::Buffer &dst,
                        const BufferLayout &layout) const {
  auto &enc = metalEncoder(s);
  auto dst_impl = static_cast<implementation::BufferMetal *>(dst.impl().get());
  size_t rowStride = layout.rowBytes(descr.pixelSize());
  size_t sliceStride = layout.sliceBytes(rowStride);
  MTLRegion region = {{0, 0, 0}, {descr.size.x, descr.size.y, descr.size.z}};
  id<MTLBlitCommandEncoder> blit = enc.getBlitEncoder();
  [blit copyFromTexture:mem.get()
                   sourceSlice:0
                   sourceLevel:0
                  sourceOrigin:region.origin
                    sourceSize:region.size
                      toBuffer:dst_impl->mem.get()
             destinationOffset:dst_impl->baseOffset()
        destinationBytesPerRow:rowStride
      destinationBytesPerImage:sliceStride];
}

void ImageMetal::copyTo(const ghost::Encoder &s, void *dst,
                        const BufferLayout &layout) const {
  size_t rowStride = layout.rowBytes(descr.pixelSize());
  size_t sliceStride = layout.sliceBytes(rowStride);
  // For non-private textures accessed from CPU, we need to drain pending GPU
  // work.
  if (!IsPrivate(mem.get())) {
    auto *stream_impl =
        dynamic_cast<implementation::StreamMetal *>(s.impl().get());
    if (!stream_impl)
      throw ghost::unsupported_error();
    stream_impl->sync();
    // Shared/Managed: read directly from texture
    MTLRegion region = {{0, 0, 0},
                        {layout.size.x, layout.size.y, layout.size.z}};
    [mem.get() getBytes:dst
            bytesPerRow:rowStride
             fromRegion:region
            mipmapLevel:0];
    return;
  }
  // Private texture: blit to staging buffer, drain stream, then memcpy
  auto *stream_impl =
      dynamic_cast<implementation::StreamMetal *>(s.impl().get());
  if (!stream_impl)
    throw ghost::unsupported_error();
  size_t dataSize = sliceStride * layout.size.z;
  id<MTLBuffer> staging =
      [mem.get().device newBufferWithLength:dataSize
                                    options:MTLResourceStorageModeShared];
  MTLRegion region = {{0, 0, 0}, {layout.size.x, layout.size.y, layout.size.z}};
  id<MTLBlitCommandEncoder> blit = stream_impl->getBlitEncoder();
  [blit copyFromTexture:mem.get()
                   sourceSlice:0
                   sourceLevel:0
                  sourceOrigin:region.origin
                    sourceSize:region.size
                      toBuffer:staging
             destinationOffset:0
        destinationBytesPerRow:rowStride
      destinationBytesPerImage:sliceStride];
  stream_impl->sync();
  memcpy(dst, [staging contents], dataSize);
}

void ImageMetal::copy(const ghost::Encoder &s, const ghost::Buffer &src,
                      const BufferLayout &layout, const Origin3 &imageOrigin) {
  auto &enc = metalEncoder(s);
  auto src_impl = static_cast<implementation::BufferMetal *>(src.impl().get());
  size_t rowStride = layout.rowBytes(descr.pixelSize());
  size_t sliceStride = layout.sliceBytes(rowStride);
  MTLRegion region = {{0, 0, 0}, {layout.size.x, layout.size.y, layout.size.z}};
  MTLOrigin dst_origin = {imageOrigin.x, imageOrigin.y, imageOrigin.z};
  id<MTLBlitCommandEncoder> blit = enc.getBlitEncoder();
  [blit copyFromBuffer:src_impl->mem.get()
             sourceOffset:src_impl->baseOffset()
        sourceBytesPerRow:rowStride
      sourceBytesPerImage:sliceStride
               sourceSize:region.size
                toTexture:mem.get()
         destinationSlice:0
         destinationLevel:0
        destinationOrigin:dst_origin];
}

void ImageMetal::copyTo(const ghost::Encoder &s, ghost::Buffer &dst,
                        const BufferLayout &layout,
                        const Origin3 &imageOrigin) const {
  auto &enc = metalEncoder(s);
  auto dst_impl = static_cast<implementation::BufferMetal *>(dst.impl().get());
  size_t rowStride = layout.rowBytes(descr.pixelSize());
  size_t sliceStride = layout.sliceBytes(rowStride);
  MTLRegion region = {{imageOrigin.x, imageOrigin.y, imageOrigin.z},
                      {layout.size.x, layout.size.y, layout.size.z}};
  id<MTLBlitCommandEncoder> blit = enc.getBlitEncoder();
  [blit copyFromTexture:mem.get()
                   sourceSlice:0
                   sourceLevel:0
                  sourceOrigin:region.origin
                    sourceSize:region.size
                      toBuffer:dst_impl->mem.get()
             destinationOffset:dst_impl->baseOffset()
        destinationBytesPerRow:rowStride
      destinationBytesPerImage:sliceStride];
}

void ImageMetal::copy(const ghost::Encoder &s, const ghost::Image &src,
                      const Size3 &region, const Origin3 &srcOrigin,
                      const Origin3 &dstOrigin) {
  auto &enc = metalEncoder(s);
  auto src_impl = static_cast<implementation::ImageMetal *>(src.impl().get());
  MTLRegion srcRegion = {{srcOrigin.x, srcOrigin.y, srcOrigin.z},
                         {region.x, region.y, region.z}};
  MTLOrigin dst = {dstOrigin.x, dstOrigin.y, dstOrigin.z};
  id<MTLBlitCommandEncoder> blit = enc.getBlitEncoder();
  [blit copyFromTexture:src_impl->mem.get()
            sourceSlice:0
            sourceLevel:0
           sourceOrigin:srcRegion.origin
             sourceSize:srcRegion.size
              toTexture:mem.get()
       destinationSlice:0
       destinationLevel:0
      destinationOrigin:dst];
}

// ---------------------------------------------------------------------------
// CommandBufferMetal
// ---------------------------------------------------------------------------

CommandBufferMetal::CommandBufferMetal(const DeviceMetal &dev_,
                                       const CommandBufferOptions &options)
    : _dev(dev_) {
  concurrent = options.concurrent;
}

CommandBufferMetal::~CommandBufferMetal() {
  try {
    waitForCompletion();
  } catch (...) {
  }
}

void CommandBufferMetal::waitForCompletion() {
  if (_submittedCB.get()) {
    [_submittedCB.get() waitUntilCompleted];
    _submittedCB = objc::ptr<id<MTLCommandBuffer>>();
  }
}

void CommandBufferMetal::begin() {
  // submit() pre-allocates commandBuffer before triggering the replay, so
  // begin() is a no-op in normal use. Throw if called outside that context.
  if (!commandBuffer.get())
    throw ghost::unsupported_error();
}

void CommandBufferMetal::submit(const ghost::Stream &stream) {
  waitForCompletion();

  auto *streamMetal =
      dynamic_cast<implementation::StreamMetal *>(stream.impl().get());
  if (!streamMetal)
    throw ghost::unsupported_error();

  // Drain any pending work on the stream's transient cb first so this cb
  // submits in correct FIFO order on the queue.
  if (streamMetal->commandBuffer.get()) {
    streamMetal->commit();
  }

  // Allocate our cb on the target stream's queue so queue FIFO ordering
  // applies between this cb and any subsequent stream ops.
  commandBuffer = [streamMetal->queue.get() commandBuffer];
  commandBuffer.get().label = @"Ghost CommandBuffer";
  fence = [streamMetal->queue.get().device newFence];
  // Resources are hazard-untracked; chain via the stream's sync event so
  // prior stream-submitted work happens-before our cb.
  if (streamMetal->syncCounter > 0) {
    [commandBuffer.get() encodeWaitForEvent:streamMetal->syncEvent.get()
                                      value:streamMetal->syncCounter];
  }

  // Replay variants against *this* — per-op functions cross-cast through
  // metalEncoder() and find our commandBuffer / encoder state.
  ghost::Encoder enc(std::shared_ptr<implementation::Encoder>(
      static_cast<implementation::Encoder *>(this), [](auto *) {}));

  ghost::Buffer srcWrap(nullptr);
  ghost::Buffer dstWrap(nullptr);
  ghost::Image srcImgWrap(nullptr);
  ghost::Image dstImgWrap(nullptr);

  // Deferred D2H finalizers. Built during ReadBufferCmd replay; consumed in
  // the cb completion handler so memcpys land before stream.sync() returns.
  struct ReadFinalizer {
    objc::ptr<id<MTLBuffer>> staging; // non-null for Private path
    std::shared_ptr<const implementation::Buffer> srcAlive; // for non-Private
    size_t srcOff;
    void *dst;
    size_t bytes;
  };
  auto readFinalizers = std::make_shared<std::vector<ReadFinalizer>>();

  // Per-resource barrier tracking. A compute->compute barrier only needs to
  // order the resources dispatches actually touched since the last barrier,
  // not every buffer+texture the concurrent encoder can reach. Passing that
  // exact set to memoryBarrierWithResources: preserves more intra-encoder
  // overlap than the coarse memoryBarrierWithScope: on barrier-dense graphs.
  // Ghost binds every resource directly (setBuffer/setTexture, no indirect
  // residency), so the args fully enumerate what a dispatch touches.
  // Resources stay alive via the recorded commands for the duration of
  // submit(), so unretained pointers are safe.
  // Above this many distinct resources in one barrier interval the per-resource
  // list stops being worth it (memoryBarrierWithResources: is O(N) and the
  // linear dedup below is O(N^2)); fall back to the O(1) coarse scope barrier.
  constexpr size_t kBarrierResourceMax = 32;
  std::vector<__unsafe_unretained id<MTLResource>> barrierResources;
  barrierResources.reserve(kBarrierResourceMax);
  bool barrierOverflow = false;
  // Small linear dedup: per-barrier-interval sets are tiny (dense graphs have a
  // run length near 1), so a vector scan beats hashing + node allocation.
  auto addBarrierResource = [&](id<MTLResource> r) {
    if (!r || barrierOverflow)
      return;
    for (id<MTLResource> existing : barrierResources)
      if (existing == r)
        return;
    if (barrierResources.size() >= kBarrierResourceMax) {
      barrierOverflow = true;
      return;
    }
    barrierResources.push_back(r);
  };
  // Resolve a kernel-argument Attribute to its backing MTLResource (nil for
  // scalars, samplers, and inline-struct argument buffers).
  auto resourceOf = [](const Attribute &a) -> id<MTLResource> {
    switch (a.type()) {
    case Attribute::Type_Buffer:
      return static_cast<implementation::BufferMetal *>(a.bufferImpl().get())
          ->mem.get();
    case Attribute::Type_Image:
      return static_cast<implementation::ImageMetal *>(a.imageImpl().get())
          ->mem.get();
    case Attribute::Type_ArgumentBuffer: {
      auto ab = a.argumentBuffer();
      if (!ab->isStruct())
        return static_cast<implementation::BufferMetal *>(
                   ab->bufferImpl().get())
            ->mem.get();
      return nil;
    }
    default:
      return nil;
    }
  };
  // A resource only needs compute->compute barrier ordering if some dispatch in
  // this cb *writes* it. Resources that are read-only across every dispatch
  // (e.g. weights/inputs) can never be a hazard source, so they are dropped
  // from every per-resource barrier. The filter is "written by some dispatch",
  // not "written by this dispatch": a resource this dispatch only reads but a
  // later dispatch overwrites still needs ordering (write-after-read), so per-
  // dispatch write-only tracking would be unsafe. Which args count as writes is
  // resolved by Function::writtenArgs (the owning Library's WriteDefault plus
  // any ghost::write()/read()/writes() overrides). Copy/blit-written resources
  // are ordered by the compute<->blit encoder fence, not this barrier, so only
  // dispatch writes feed the set.
  std::unordered_set<void *> writtenByDispatch;
  for (auto &command : commands) {
    auto collect = [&](const std::shared_ptr<implementation::Function> &fn,
                       const std::vector<Attribute> &args) {
      std::vector<bool> written = fn->writtenArgs(args);
      for (size_t k = 0; k < args.size(); ++k)
        if (written[k])
          if (id<MTLResource> r = resourceOf(args[k]))
            writtenByDispatch.insert((__bridge void *)r);
    };
    if (auto *d = std::get_if<DispatchCmd>(&command))
      collect(d->function, d->args);
    else if (auto *di = std::get_if<DispatchIndirectCmd>(&command))
      collect(di->function, di->args);
  }
  auto addBarrierArgs = [&](const std::vector<Attribute> &args) {
    for (auto &a : args)
      if (id<MTLResource> r = resourceOf(a))
        if (writtenByDispatch.count((__bridge void *)r))
          addBarrierResource(r);
  };
  auto clearBarrierResources = [&]() {
    barrierResources.clear();
    barrierOverflow = false;
  };

  for (auto &command : commands) {
    std::visit(
        [&](auto &cmd) {
          using T = std::decay_t<decltype(cmd)>;
          if constexpr (std::is_same_v<T, DispatchCmd>) {
            cmd.function->execute(enc, cmd.launchArgs, cmd.args);
            addBarrierArgs(cmd.args);
          } else if constexpr (std::is_same_v<T, DispatchIndirectCmd>) {
            cmd.function->executeIndirect(enc, cmd.indirectBuffer,
                                          cmd.indirectOffset, cmd.args);
            addBarrierArgs(cmd.args);
            // The indirect-args buffer is read here; order it only if a
            // dispatch writes it elsewhere (e.g. a prior dispatch computed the
            // counts), matching the writtenByDispatch filter.
            id<MTLResource> ind = static_cast<implementation::BufferMetal *>(
                                      cmd.indirectBuffer.get())
                                      ->mem.get();
            if (ind && writtenByDispatch.count((__bridge void *)ind))
              addBarrierResource(ind);
          } else if constexpr (std::is_same_v<T, CopyBufferCmd>) {
            dstWrap.impl() = cmd.dst;
            srcWrap.impl() =
                std::const_pointer_cast<implementation::Buffer>(cmd.src);
            cmd.dst->copy(enc, srcWrap, cmd.srcOffset, cmd.dstOffset,
                          cmd.bytes);
          } else if constexpr (std::is_same_v<T, CopyBufferRawCmd>) {
            cmd.dst->copy(enc, cmd.src.data(), cmd.dstOffset, cmd.bytes);
          } else if constexpr (std::is_same_v<T, ReadBufferCmd>) {
            // BufferMetal::copyTo's stream-only path can't run on a cb
            // encoder (it would do a synchronous sync()). Encode the blit
            // here and queue a finalizer that memcpys to the user's host
            // dst inside the cb completion handler.
            auto *srcImpl =
                static_cast<const implementation::BufferMetal *>(cmd.src.get());
            size_t effOff = srcImpl->baseOffset() + cmd.srcOffset;
            MTLStorageMode mode = srcImpl->mem.get().storageMode;
            ReadFinalizer fin;
            fin.dst = cmd.dst;
            fin.bytes = cmd.bytes;
            if (mode == MTLStorageModePrivate) {
              fin.staging = [srcImpl->mem.get().device
                  newBufferWithLength:cmd.bytes
                              options:MTLResourceStorageModeShared];
              fin.srcOff = 0;
              id<MTLBlitCommandEncoder> blit = getBlitEncoder();
              [blit copyFromBuffer:srcImpl->mem.get()
                       sourceOffset:effOff
                           toBuffer:fin.staging.get()
                  destinationOffset:0
                               size:cmd.bytes];
            } else {
              if (mode == MTLStorageModeManaged) {
                id<MTLBlitCommandEncoder> blit = getBlitEncoder();
                [blit synchronizeResource:srcImpl->mem.get()];
              }
              fin.srcAlive = cmd.src;
              fin.srcOff = effOff;
            }
            readFinalizers->push_back(std::move(fin));
          } else if constexpr (std::is_same_v<T, FillBufferCmd>) {
            cmd.dst->fill(enc, cmd.offset, cmd.size, cmd.value);
          } else if constexpr (std::is_same_v<T, FillBufferPatternCmd>) {
            cmd.dst->fill(enc, cmd.offset, cmd.size, cmd.pattern.data(),
                          cmd.pattern.size());
          } else if constexpr (std::is_same_v<T, CopyImageCmd>) {
            dstImgWrap.impl() =
                std::const_pointer_cast<implementation::Image>(cmd.dst);
            srcImgWrap.impl() =
                std::const_pointer_cast<implementation::Image>(cmd.src);
            cmd.dst->copy(enc, srcImgWrap);
          } else if constexpr (std::is_same_v<T, CopyImageFromBufferCmd>) {
            srcWrap.impl() = cmd.src;
            cmd.dst->copy(enc, srcWrap, cmd.layout);
          } else if constexpr (std::is_same_v<T, CopyImageFromHostCmd>) {
            cmd.dst->copy(enc, cmd.src.data(), cmd.layout);
          } else if constexpr (std::is_same_v<T, CopyImageToBufferCmd>) {
            dstWrap.impl() = cmd.dst;
            cmd.src->copyTo(enc, dstWrap, cmd.layout);
          } else if constexpr (std::is_same_v<T, CopyImageToHostCmd>) {
            cmd.src->copyTo(enc, cmd.dst, cmd.layout);
          } else if constexpr (std::is_same_v<T, BarrierCmd>) {
            // Compute->compute: order just the resources dispatches touched
            // since the last barrier (memoryBarrierWithResources:), which
            // keeps more of the concurrent encoder's overlap than the coarse
            // memoryBarrierWithScope:(Buffers|Textures) on barrier-dense
            // graphs. The tracked set covers both buffers and textures
            // (Ghost Images), preserving barrier()'s order-everything
            // semantics. An empty set means nothing was dispatched since the
            // last barrier, so there is nothing to order. A pending
            // compute<->blit transition still needs the cross-encoder fence;
            // fall back to endEncoding() when no compute encoder is open or
            // the API is unavailable.
            if (computeEncoder.get() && !blitEncoder.get()) {
              if (!concurrent) {
                // Serial encoder (MTLDispatchTypeSerial): consecutive
                // dispatches already execute in order, so an intra-encoder
                // barrier is redundant. (A compute<->blit transition still
                // needs the cross-encoder fence, handled by the else branch.)
                clearBarrierResources();
              } else if (@available(macOS 10.14, iOS 12.0, tvOS 12.0,
                                    macCatalyst 13.0, *)) {
                if (barrierOverflow) {
                  // Too many resources to order individually; the coarse scope
                  // barrier is O(1) and strictly broader, so still correct.
                  [computeEncoder.get()
                      memoryBarrierWithScope:MTLBarrierScopeBuffers |
                                             MTLBarrierScopeTextures];
                } else if (!barrierResources.empty()) {
                  [computeEncoder.get()
                      memoryBarrierWithResources:barrierResources.data()
                                           count:barrierResources.size()];
                }
                clearBarrierResources();
              } else {
                endEncoding();
              }
            } else {
              endEncoding();
            }
          } else if constexpr (std::is_same_v<T, WaitEventCmd>) {
            throw ghost::unsupported_error();
          } else if constexpr (std::is_same_v<T, RecordEventCmd>) {
            throw ghost::unsupported_error();
          }
        },
        command);
    // A blit transition or endEncoding() ends the compute encoder and fences
    // its writes, so resources tracked for the per-resource barrier are now
    // ordered by that fence. Reset the running set when no compute encoder is
    // open (the per-resource barrier branch leaves it open and self-clears).
    if (!computeEncoder.get())
      clearBarrierResources();
  }

  endEncoding();
  // Bump the stream's sync chain so subsequent stream cbs (or another
  // CommandBuffer submit) wait for us.
  streamMetal->syncCounter++;
  [commandBuffer.get() encodeSignalEvent:streamMetal->syncEvent.get()
                                   value:streamMetal->syncCounter];

  // Attach a single completion handler that runs D2H finalizers first (so
  // host dst pointers are filled before user callbacks observe them) and
  // then any user-registered onCompletion handlers. Move both into the
  // block so the handler is self-sufficient even if *this is destroyed
  // before completion fires.
  if (!readFinalizers->empty() || !pendingCompletionHandlers.empty()) {
    auto userHandlers = std::make_shared<std::vector<std::function<void()>>>();
    *userHandlers = std::move(pendingCompletionHandlers);
    pendingCompletionHandlers.clear();
    [commandBuffer.get() addCompletedHandler:^(id<MTLCommandBuffer>) {
      for (auto &f : *readFinalizers) {
        const void *src;
        if (f.staging.get()) {
          src = [f.staging.get() contents];
        } else {
          auto *bm = static_cast<const implementation::BufferMetal *>(
              f.srcAlive.get());
          src =
              static_cast<const uint8_t *>([bm->mem.get() contents]) + f.srcOff;
        }
        memcpy(f.dst, src, f.bytes);
      }
      for (auto &h : *userHandlers)
        h();
    }];
  }

  [commandBuffer.get() commit];
  _submittedCB = commandBuffer;
  // Register with the stream so stream.sync() waits for our cb. Queue
  // FIFO on the stream's queue then handles ordering with subsequent
  // stream-submitted ops.
  streamMetal->attachCommitted(commandBuffer);
  commandBuffer = objc::ptr<id<MTLCommandBuffer>>();
  fence = objc::ptr<id<MTLFence>>();
}

void CommandBufferMetal::reset() {
  waitForCompletion();
  commands.clear();
  // Drop any handlers that were registered but not yet submitted. Handlers
  // attached to a prior submit() are already owned by the cb's completion
  // block and fire independently of the next recording.
  pendingCompletionHandlers.clear();
}

void CommandBufferMetal::onCompletion(std::function<void()> handler) {
  // Use the RecordedCommandBuffer field so submit() can hand it off uniformly
  // with the rest of the variant-replay machinery.
  pendingCompletionHandlers.push_back(std::move(handler));
}

// ---------------------------------------------------------------------------
// DeviceMetal (implementation)
// ---------------------------------------------------------------------------

DeviceMetal::DeviceMetal(const SharedContext &share) {
  if (share.device) {
#if __has_feature(objc_arc)
    dev = (__bridge id<MTLDevice>)share.device;
#else
    dev = objc::ptr<id<MTLDevice>>(
        reinterpret_cast<id<MTLDevice>>(share.device), true);
#endif
  }
  if (share.queue) {
#if __has_feature(objc_arc)
    queue = (__bridge id<MTLCommandQueue>)share.queue;
#else
    queue = objc::ptr<id<MTLCommandQueue>>(
        reinterpret_cast<id<MTLCommandQueue>>(share.queue), true);
#endif
  }
  if (!dev.get() && queue.get()) {
    dev = queue.get().device;
  }
  if (!dev) {
    dev = MTLCreateSystemDefaultDevice();
    checkExists(dev);
  }
  if (!queue) {
    queue = [dev.get() newCommandQueue];
    checkExists(queue);
  }
}

DeviceMetal::DeviceMetal(const GpuInfo &info) {
  @autoreleasepool {
#if TARGET_OS_OSX
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    if (info.index >= 0 && info.index < (int)devices.count) {
      dev = objc::ptr<id<MTLDevice>>(devices[info.index], true);
      queue = [dev.get() newCommandQueue];
      checkExists(queue);
      return;
    }
#endif
    // Fallback: default device
    dev = MTLCreateSystemDefaultDevice();
    checkExists(dev);
    queue = [dev.get() newCommandQueue];
    checkExists(queue);
  }
}

DeviceMetal::DeviceMetal(id<MTLDevice> device) {
  dev = objc::ptr<id<MTLDevice>>(device, true);
  queue = [dev.get() newCommandQueue];
  checkExists(queue);
}

ghost::Library DeviceMetal::loadLibraryFromText(const std::string &text,
                                                const CompilerOptions &options,
                                                bool retainBinary) const {
  auto ptr =
      std::make_shared<implementation::LibraryMetal>(*this, retainBinary);
  ptr->loadFromText(text, options);
  return ghost::Library(ptr);
}

ghost::Library DeviceMetal::loadLibraryFromData(const void *data, size_t len,
                                                const CompilerOptions &options,
                                                bool retainBinary) const {
  auto ptr =
      std::make_shared<implementation::LibraryMetal>(*this, retainBinary);
  ptr->loadFromData(data, len, options);
  return ghost::Library(ptr);
}

SharedContext DeviceMetal::shareContext() const {
  SharedContext c;
#if __has_feature(objc_arc)
  c.device = (__bridge void *)dev.get();
  c.queue = (__bridge void *)queue.get();
#else
  c.device = dev.get();
  c.queue = queue.get();
#endif
  return c;
}

ghost::Stream DeviceMetal::createStream(const StreamOptions &options) const {
  auto ptr = std::make_shared<implementation::StreamMetal>(dev.get(), options);
  return ghost::Stream(ptr);
}

std::shared_ptr<CommandBuffer>
DeviceMetal::createCommandBuffer(const CommandBufferOptions &options) const {
  return std::make_shared<CommandBufferMetal>(*this, options);
}

size_t DeviceMetal::getMemoryPoolSize() const {
  if (heap)
    return heap.get().size;
  return Device::getMemoryPoolSize();
}

void DeviceMetal::setMemoryPoolSize(size_t bytes) {
  Device::setMemoryPoolSize(bytes);
  if (bytes > 0) {
    MTLHeapDescriptor *descriptor = [[MTLHeapDescriptor alloc] init];
    descriptor.size = bytes;
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.cpuCacheMode = MTLCPUCacheModeDefaultCache;
    descriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    heap = [dev.get() newHeapWithDescriptor:descriptor];
  } else {
    heap = objc::ptr<id<MTLHeap>>();
  }
}

ghost::Buffer DeviceMetal::allocateBuffer(size_t bytes,
                                          const BufferOptions &opts) const {
  // Use the MTLHeap for Default/Transient (private storage). Staging and
  // Persistent bypass the heap — Staging needs host-visible memory, and
  // Persistent is long-lived so heap fragmentation is undesirable.
  if (heap && opts.hint != AllocHint::Staging &&
      opts.hint != AllocHint::Persistent && opts.hint != AllocHint::Shared) {
    MTLResourceOptions options = MTLResourceCPUCacheModeDefaultCache |
                                 MTLResourceHazardTrackingModeUntracked |
                                 MTLResourceStorageModePrivate;
    objc::ptr<id<MTLBuffer>> buf([heap.get() newBufferWithLength:bytes
                                                         options:options]);
    if (buf) {
      auto ptr = std::make_shared<implementation::BufferMetal>(buf, bytes);
      return ghost::Buffer(ptr);
    }
    // Heap full — fall through to external allocator / individual allocation
  }
  if (auto *a = allocator()) {
    if (void *handle = a->allocateBuffer(bytes, opts)) {
      id<MTLBuffer> mtlBuf = GHOST_OBJC_BRIDGE_TRANSFER(id<MTLBuffer>, handle);
      // retainObject=true means "I'm taking ownership, don't retain again"
      objc::ptr<id<MTLBuffer>> buf(mtlBuf, /*retainObject=*/true);
      auto ptr = std::make_shared<implementation::BufferMetal>(buf, bytes);
      ptr->setAllocator(a);
      return ghost::Buffer(ptr);
    }
  }
  auto ptr = std::make_shared<implementation::BufferMetal>(*this, bytes, opts);
  return ghost::Buffer(ptr);
}

ghost::MappedBuffer
DeviceMetal::allocateMappedBuffer(size_t bytes,
                                  const BufferOptions &opts) const {
  if (auto *a = allocator()) {
    if (void *handle = a->allocateMappedBuffer(bytes, opts)) {
      id<MTLBuffer> mtlBuf = GHOST_OBJC_BRIDGE_TRANSFER(id<MTLBuffer>, handle);
      objc::ptr<id<MTLBuffer>> buf(mtlBuf, /*retainObject=*/true);
      auto ptr =
          std::make_shared<implementation::MappedBufferMetal>(buf, bytes);
      ptr->setAllocator(a);
      return ghost::MappedBuffer(ptr);
    }
  }
  auto ptr =
      std::make_shared<implementation::MappedBufferMetal>(*this, bytes, opts);
  return ghost::MappedBuffer(ptr);
}

ghost::Image DeviceMetal::allocateImage(const ImageDescription &descr) const {
  if (heap) {
    objc::ptr<MTLTextureDescriptor *> textureDescriptor(
        getTextureDescriptor(descr));
    objc::ptr<id<MTLTexture>> tex(
        [heap.get() newTextureWithDescriptor:textureDescriptor.get()]);
    if (tex) {
      auto ptr = std::make_shared<implementation::ImageMetal>(tex, descr);
      return ghost::Image(ptr);
    }
    // Heap full — fall through to external allocator / individual allocation
  }
  if (auto *a = allocator()) {
    if (void *handle = a->allocateImage(descr)) {
      id<MTLTexture> mtlTex =
          GHOST_OBJC_BRIDGE_TRANSFER(id<MTLTexture>, handle);
      objc::ptr<id<MTLTexture>> tex(mtlTex, /*retainObject=*/true);
      auto ptr = std::make_shared<implementation::ImageMetal>(tex, descr);
      ptr->setAllocator(a);
      return ghost::Image(ptr);
    }
  }
  auto ptr = std::make_shared<implementation::ImageMetal>(*this, descr);
  return ghost::Image(ptr);
}

ghost::Image DeviceMetal::sharedImage(const ImageDescription &descr,
                                      ghost::Buffer &buffer) const {
  auto b = static_cast<implementation::BufferMetal *>(buffer.impl().get());
  auto ptr = std::make_shared<implementation::ImageMetal>(*this, descr, *b);
  return ghost::Image(ptr);
}

ghost::Image DeviceMetal::sharedImage(const ImageDescription &descr,
                                      ghost::Image &image) const {
  auto i = static_cast<implementation::ImageMetal *>(image.impl().get());
  auto ptr = std::make_shared<implementation::ImageMetal>(*this, descr, *i);
  return ghost::Image(ptr);
}

ghost::Buffer DeviceMetal::wrapBuffer(const SharedBuffer &shared) const {
  id<MTLBuffer> buf = (__bridge id<MTLBuffer>)shared.handle;
  // retainObject=false: ARC's strong-ivar assign in objc::ptr balances the
  // dealloc release. Host's +1 is unaffected.
  objc::ptr<id<MTLBuffer>> p(buf, /*retainObject=*/false);
  auto ptr = std::make_shared<implementation::BufferMetal>(p, shared.bytes);
  return ghost::Buffer(ptr);
}

ghost::Image DeviceMetal::wrapImage(const SharedImage &shared) const {
  id<MTLTexture> tex = (__bridge id<MTLTexture>)shared.handle;
  objc::ptr<id<MTLTexture>> p(tex, /*retainObject=*/false);
  auto ptr = std::make_shared<implementation::ImageMetal>(p, shared.descr);
  return ghost::Image(ptr);
}

namespace {
const char *getMetalVersion() {
#if defined(MAC_OS_VERSION_26_0)
  if (@available(macOS 26, iOS 26.0, tvOS 26.0, macCatalyst 26.0,
                 visionOS 26.0, *)) {
    return "4.0";
  }
#endif
#if defined(MAC_OS_VERSION_15_0)
  if (@available(macOS 15, iOS 18.0, tvOS 18.0, macCatalyst 18.0, *)) {
    return "3.2";
  }
#endif
#if defined(MAC_OS_VERSION_14_0)
  if (@available(macOS 14, iOS 17.0, tvOS 17.0, macCatalyst 17.0, *)) {
    return "3.1";
  }
#endif
#if defined(MAC_OS_VERSION_13_0)
  if (@available(macOS 13, iOS 16.0, tvOS 16.0, macCatalyst 16.0, *)) {
    return "3.0";
  }
#endif
#if defined(MAC_OS_VERSION_12_0)
  if (@available(macOS 12, iOS 15.0, tvOS 15.0, macCatalyst 15.0, *)) {
    return "2.4";
  }
#endif
#if defined(MAC_OS_VERSION_11_0)
  if (@available(macOS 11, iOS 14.0, tvOS 14.0, macCatalyst 14.0, *)) {
    return "2.3";
  }
#endif
  if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, macCatalyst 13.0, *)) {
    return "2.2";
  }
  if (@available(macOS 10.14, iOS 12.0, tvOS 12.0, macCatalyst 12.0, *)) {
    return "2.1";
  }
  if (@available(macOS 10.13, iOS 11.0, tvOS 11.0, macCatalyst 11.0, *)) {
    return "2.0";
  }
  if (@available(macOS 10.12, iOS 10.0, tvOS 10.0, macCatalyst 10.0, *)) {
    return "1.2";
  }
  if (@available(macOS 10.11, iOS 9.0, tvOS 9.0, macCatalyst 9.0, *)) {
    return "1.1";
  }
  if (@available(iOS 8.0, *)) {
    return "1.0";
  }
  return "";
}

std::string getOSRelease() {
  int mib[2];
  mib[0] = CTL_KERN;
  mib[1] = KERN_OSRELEASE;
  size_t len;
  sysctl(mib, 2, nullptr, &len, nullptr, 0);
  char *osrelease = (char *)malloc(len);
  sysctl(mib, 2, osrelease, &len, nullptr, 0);
  std::string s = osrelease;
  free(osrelease);
  return s;
}
} // namespace

Attribute DeviceMetal::getAttribute(DeviceAttributeId what) const {
  switch (what) {
  case kDeviceImplementation:
    return "Metal";
  case kDeviceName:
    return [[dev.get() name] UTF8String];
  case kDeviceVendor:
    return "Apple";
  case kDeviceDriverVersion:
    return getMetalVersion();
  case kDeviceFamily:
    if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, macCatalyst 13.1, *)) {
      if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
        if ([dev.get() supportsFamily:MTLGPUFamilyApple10])
          return "Apple10";
      }
      if ([dev.get() supportsFamily:MTLGPUFamilyApple9])
        return "Apple9";
      if ([dev.get() supportsFamily:MTLGPUFamilyApple8])
        return "Apple8";
      if ([dev.get() supportsFamily:MTLGPUFamilyApple7])
        return "Apple7";
      if ([dev.get() supportsFamily:MTLGPUFamilyApple6])
        return "Apple6";
      if ([dev.get() supportsFamily:MTLGPUFamilyApple5])
        return "Apple5";
      if ([dev.get() supportsFamily:MTLGPUFamilyApple4])
        return "Apple4";
      if ([dev.get() supportsFamily:MTLGPUFamilyApple3])
        return "Apple3";
      if ([dev.get() supportsFamily:MTLGPUFamilyApple2])
        return "Apple2";
      if ([dev.get() supportsFamily:MTLGPUFamilyApple1])
        return "Apple1";
      if ([dev.get() supportsFamily:MTLGPUFamilyMac2])
        return "Mac2";
      if ([dev.get() supportsFamily:MTLGPUFamilyMac1])
        return "Mac1";
    }
    return "Unknown";
  case kDeviceCount:
    return 1;
  case kDeviceProcessorCount:
    return 1;
  case kDeviceUnifiedMemory:
    return (bool)dev.get().hasUnifiedMemory;
  case kDeviceMemory:
    return (uint64_t)dev.get().recommendedMaxWorkingSetSize;
  case kDeviceLocalMemory:
    return (uint64_t)dev.get().maxThreadgroupMemoryLength;
  case kDeviceMaxThreads: {
    MTLSize size = dev.get().maxThreadsPerThreadgroup;
    return uint64_t(size.width * size.height * size.depth);
  }
  case kDeviceMaxWorkSize: {
    MTLSize size = dev.get().maxThreadsPerThreadgroup;
    return Attribute((uint32_t)size.width, (uint32_t)size.height,
                     (uint32_t)size.depth);
  }
  case kDeviceMaxRegisters:
    return 0;
  case kDeviceMaxImageSize1:
  case kDeviceMaxImageSize2: {
    int32_t v = 16384;
    if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, macCatalyst 13.1, *)) {
      if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
        if ([dev.get() supportsFamily:MTLGPUFamilyApple10]) {
          v = 32786;
        }
      } else if (![dev.get() supportsFamily:MTLGPUFamilyApple3]) {
        v = 8192;
      }
    } else {
#if TARGET_OS_IPHONE
      if (![dev.get() supportsFeatureSet:MTLFeatureSet_iOS_GPUFamily3_v1]) {
        v = 8192;
      }
#endif
    }
    if (what == kDeviceMaxImageSize1) {
      return v;
    }
    return Attribute(v, v);
  }
  case kDeviceMaxImageSize3: {
    const int32_t v = 2048;
    return Attribute(v, v, v);
  }
  case kDeviceMaxImageAlignment: {
    ImageDescription descr(Size3(16, 16, 1), PixelOrder_RGBA, DataType_Float,
                           Stride2(0, 0));
    size_t size = 256;
    if (@available(macOS 10.13, iOS 11.0, tvOS 11.0, macCatalyst 13.1, *)) {
      size = [dev.get()
          minimumLinearTextureAlignmentForPixelFormat:getFormat(descr)];
    }
    return uint32_t(size);
  }
  case kDeviceSupportsImageIntegerFiltering:
    return true;
  case kDeviceSupportsImageFloatFiltering:
    if (@available(macOS 11, iOS 14.0, tvOS 16.0, macCatalyst 14.0, *)) {
      return dev.get().supports32BitFloatFiltering != NO;
    }
    return false;
  case kDeviceSupportsMappedBuffer:
    return true;
  case kDeviceSupportsProgramConstants:
    return true;
  case kDeviceSupportsProgramGlobals:
    return false;
  case kDeviceSupportsSubgroup:
  case kDeviceSupportsSubgroupShuffle:
    if (@available(macOS 10.13, iOS 11.0, tvOS 11.0, macCatalyst 11.0, *)) {
      return true;
    }
    return false;
  case kDeviceSubgroupWidth:
    return 32;
  case kDeviceMaxComputeUnits:
    return 1;
  case kDeviceMemoryAlignment:
    return (uint32_t)256;
  case kDeviceBufferAlignment:
    return (uint32_t)256;
  case kDeviceMaxBufferSize:
    return (uint64_t)dev.get().maxBufferLength;
  case kDeviceMaxConstantBufferSize:
    return (uint64_t)dev.get().maxBufferLength;
  case kDeviceTimestampPeriod:
    return 0.0f;
  case kDeviceSupportsProfilingTimer:
    return false;
  case kDeviceSupportsCooperativeMatrix:
    if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, macCatalyst 13.1, *)) {
      return (bool)[dev.get() supportsFamily:MTLGPUFamilyApple7];
    }
    return false;
  default:
    return Attribute();
  }
}

size_t DeviceMetal::imageAlignment(const ImageDescription &descr) const {
  size_t size = 256;
  if (@available(macOS 10.13, iOS 11.0, tvOS 11.0, macCatalyst 13.1, *)) {
    size = [dev.get()
        minimumLinearTextureAlignmentForPixelFormat:getFormat(descr)];
  }
  return size > 0 ? size : 1;
}
} // namespace implementation

DeviceMetal::DeviceMetal(const SharedContext &share)
    : Device(std::make_shared<implementation::DeviceMetal>(share)) {
  auto metal = static_cast<implementation::DeviceMetal *>(impl().get());
  setDefaultStream(std::make_shared<implementation::StreamMetal>(metal->queue));
}

DeviceMetal::DeviceMetal(const GpuInfo &info)
    : Device(std::make_shared<implementation::DeviceMetal>(info)) {
  auto metal = static_cast<implementation::DeviceMetal *>(impl().get());
  setDefaultStream(std::make_shared<implementation::StreamMetal>(metal->queue));
}

std::vector<GpuInfo> DeviceMetal::enumerateDevices() {
  std::vector<GpuInfo> result;
  @autoreleasepool {
#if TARGET_OS_OSX
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    for (NSUInteger i = 0; i < devices.count; i++) {
      id<MTLDevice> d = devices[i];
      GpuInfo info;
      info.name = [[d name] UTF8String];
      info.vendor = "Apple";
      info.implementation = "Metal";
      info.memory = (uint64_t)d.recommendedMaxWorkingSetSize;
      info.unifiedMemory = d.hasUnifiedMemory;
      info.index = (int)i;
      result.push_back(info);
    }
#else
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    if (d) {
      GpuInfo info;
      info.name = [[d name] UTF8String];
      info.vendor = "Apple";
      info.implementation = "Metal";
      info.memory = (uint64_t)d.recommendedMaxWorkingSetSize;
      info.unifiedMemory = d.hasUnifiedMemory;
      info.index = 0;
      result.push_back(info);
    }
#endif
  }
  return result;
}
} // namespace ghost
#endif

// vim: ts=2:sw=2:et:ft=mm
// -*- mode: objective-c++; indent-tabs-mode: nil; tab-width: 2 -*-
// code: language=objective-c++ insertSpaces=true tabSize=2
