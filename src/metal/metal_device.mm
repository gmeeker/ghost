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

#include <ghost/metal/device.h>
#include <ghost/metal/exception.h>
#include <ghost/metal/impl_device.h>
#include <ghost/metal/impl_function.h>

#include <sys/sysctl.h>
#include <sys/types.h>

#include <TargetConditionals.h>

namespace ghost {
namespace implementation {
using namespace metal;

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
  switch (descr.access) {
  case Access_ReadOnly:
    d.get().usage = MTLTextureUsageShaderRead;
    break;
  case Access_WriteOnly:
    d.get().usage = MTLTextureUsageShaderWrite;
    break;
  case Access_ReadWrite:
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
  switch (descr.access) {
  case Access_ReadOnly:
    d.get().usage = MTLTextureUsageShaderRead;
    break;
  case Access_WriteOnly:
    d.get().usage = MTLTextureUsageShaderWrite;
    break;
  case Access_ReadWrite:
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

StreamMetal::StreamMetal(objc::ptr<id<MTLCommandQueue>> queue_)
    : queue(queue_) {
  if (queue.get()) {
    syncEvent = [queue.get().device newEvent];
  }
}
StreamMetal::StreamMetal(id<MTLDevice> dev) {
  queue = [dev newCommandQueue];
  checkExists(queue);
  syncEvent = [dev newEvent];
}

void StreamMetal::commitAndTrack(id<MTLCommandBuffer> cb) {
  syncCounter++;
  [cb encodeSignalEvent:syncEvent.get() value:syncCounter];
  [cb commit];
  lastCommandBuffer = objc::ptr<id<MTLCommandBuffer>>(cb, true);
}

void StreamMetal::encodeWait(id<MTLCommandBuffer> cb) {
  if (syncCounter > 0) {
    [cb encodeWaitForEvent:syncEvent.get() value:syncCounter];
  }
}

void StreamMetal::sync() {
  if (lastCommandBuffer.get()) {
    [lastCommandBuffer.get() waitUntilCompleted];
    lastCommandBuffer = objc::ptr<id<MTLCommandBuffer>>();
  }
}

std::shared_ptr<Event> StreamMetal::record() {
  id<MTLDevice> device = queue.get().device;
  objc::ptr<id<MTLSharedEvent>> sharedEvent =
      objc::ptr<id<MTLSharedEvent>>([device newSharedEvent]);
  uint64_t value = 1;
  id<MTLCommandBuffer> commandBuffer = [queue.get() commandBuffer];
  commandBuffer.label = @"Ghost Event Record";
  [commandBuffer encodeSignalEvent:sharedEvent.get() value:value];
  commitAndTrack(commandBuffer);
  return std::make_shared<EventMetal>(sharedEvent, value);
}

void StreamMetal::waitForEvent(const std::shared_ptr<Event> &e) {
  auto eventMetal = static_cast<EventMetal *>(e.get());
  id<MTLCommandBuffer> commandBuffer = [queue.get() commandBuffer];
  commandBuffer.label = @"Ghost Event Wait";
  [commandBuffer encodeWaitForEvent:eventMetal->sharedEvent.get()
                              value:eventMetal->targetValue];
  commitAndTrack(commandBuffer);
}

BufferMetal::BufferMetal(objc::ptr<id<MTLBuffer>> mem_, size_t bytes)
    : mem(mem_), _size(bytes) {}

BufferMetal::BufferMetal(const DeviceMetal &dev, size_t bytes, Access access)
    : _size(bytes) {
  MTLResourceOptions options = MTLResourceCPUCacheModeDefaultCache |
                               MTLResourceHazardTrackingModeUntracked |
                               MTLResourceStorageModePrivate;
  mem = [dev.dev newBufferWithLength:bytes options:options];
  checkExists(mem);
}

size_t BufferMetal::size() const { return _size; }

void BufferMetal::copy(const ghost::Stream &s, const ghost::Buffer &src,
                       size_t bytes) {
  copy(s, src, 0, 0, bytes);
}

void BufferMetal::copy(const ghost::Stream &s, const void *src, size_t bytes) {
  copy(s, src, 0, bytes);
}

void BufferMetal::copyTo(const ghost::Stream &s, void *dst,
                         size_t bytes) const {
  copyTo(s, dst, 0, bytes);
}

void BufferMetal::copy(const ghost::Stream &s, const ghost::Buffer &src,
                       size_t srcOffset, size_t dstOffset, size_t bytes) {
  auto stream_impl = static_cast<implementation::StreamMetal *>(s.impl().get());
  auto src_impl = static_cast<implementation::BufferMetal *>(src.impl().get());
  size_t effectiveSrcOff = src_impl->baseOffset() + srcOffset;
  size_t effectiveDstOff = baseOffset() + dstOffset;

  // Metal's copyFromBuffer has undefined behavior when source and destination
  // regions overlap within the same buffer. Use a staging buffer in that case.
  bool sameBuffer = (src_impl->mem.get() == mem.get());
  bool overlaps = sameBuffer && (effectiveSrcOff < effectiveDstOff + bytes) &&
                  (effectiveDstOff < effectiveSrcOff + bytes);

  if (overlaps) {
    id<MTLDevice> device = mem.get().device;
    id<MTLBuffer> staging =
        [device newBufferWithLength:bytes options:mem.get().resourceOptions];
    id<MTLCommandBuffer> commandBuffer =
        [stream_impl->queue.get() commandBuffer];
    commandBuffer.label = @"Ghost";
    stream_impl->encodeWait(commandBuffer);
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
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
    [blit endEncoding];
    stream_impl->commitAndTrack(commandBuffer);
  } else {
    id<MTLCommandBuffer> commandBuffer =
        [stream_impl->queue.get() commandBuffer];
    commandBuffer.label = @"Ghost";
    stream_impl->encodeWait(commandBuffer);
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    [blit copyFromBuffer:src_impl->mem.get()
             sourceOffset:effectiveSrcOff
                 toBuffer:mem.get()
        destinationOffset:effectiveDstOff
                     size:bytes];
    [blit endEncoding];
    stream_impl->commitAndTrack(commandBuffer);
  }
}

void BufferMetal::copy(const ghost::Stream &s, const void *src,
                       size_t dstOffset, size_t bytes) {
  auto stream_impl = static_cast<implementation::StreamMetal *>(s.impl().get());
  size_t effectiveDstOff = baseOffset() + dstOffset;
  if (IsPrivate(mem)) {
    // For private buffers, use a temporary shared buffer and blit
    id<MTLBuffer> staging =
        [mem.get().device newBufferWithBytes:src
                                      length:bytes
                                     options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> commandBuffer =
        [stream_impl->queue.get() commandBuffer];
    commandBuffer.label = @"Ghost";
    stream_impl->encodeWait(commandBuffer);
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    [blit copyFromBuffer:staging
             sourceOffset:0
                 toBuffer:mem.get()
        destinationOffset:effectiveDstOff
                     size:bytes];
    [blit endEncoding];
    stream_impl->commitAndTrack(commandBuffer);
  } else {
    void *dst = static_cast<uint8_t *>([mem contents]) + effectiveDstOff;
    memcpy(dst, src, bytes);
    if (mem.get().storageMode == MTLStorageModeManaged)
      [mem didModifyRange:NSMakeRange(effectiveDstOff, bytes)];
  }
}

void BufferMetal::copyTo(const ghost::Stream &s, void *dst, size_t srcOffset,
                         size_t bytes) const {
  auto stream_impl = static_cast<implementation::StreamMetal *>(s.impl().get());
  size_t effectiveSrcOff = baseOffset() + srcOffset;
  if (mem.get().storageMode == MTLStorageModePrivate) {
    // Private buffers: blit to a shared staging buffer, wait, then memcpy.
    id<MTLBuffer> staging =
        [mem.get().device newBufferWithLength:bytes
                                      options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> commandBuffer =
        [stream_impl->queue.get() commandBuffer];
    commandBuffer.label = @"Ghost";
    stream_impl->encodeWait(commandBuffer);
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    [blit copyFromBuffer:mem.get()
             sourceOffset:effectiveSrcOff
                 toBuffer:staging
        destinationOffset:0
                     size:bytes];
    [blit endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    memcpy(dst, [staging contents], bytes);
  } else if (mem.get().storageMode == MTLStorageModeManaged) {
    id<MTLCommandBuffer> commandBuffer =
        [stream_impl->queue.get() commandBuffer];
    commandBuffer.label = @"Ghost";
    stream_impl->encodeWait(commandBuffer);
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    [blit synchronizeResource:mem];
    [blit endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    const void *src =
        static_cast<const uint8_t *>([mem contents]) + effectiveSrcOff;
    memcpy(dst, src, bytes);
  } else {
    const void *src =
        static_cast<const uint8_t *>([mem contents]) + effectiveSrcOff;
    memcpy(dst, src, bytes);
  }
}

void BufferMetal::fill(const ghost::Stream &s, size_t offset, size_t size,
                       uint8_t value) {
  auto stream_impl = static_cast<implementation::StreamMetal *>(s.impl().get());
  size_t effectiveOff = baseOffset() + offset;
  id<MTLCommandBuffer> commandBuffer = [stream_impl->queue.get() commandBuffer];
  commandBuffer.label = @"Ghost";
  stream_impl->encodeWait(commandBuffer);
  id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
  [blit fillBuffer:mem.get() range:NSMakeRange(effectiveOff, size) value:value];
  [blit endEncoding];
  stream_impl->commitAndTrack(commandBuffer);
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

void SubBufferMetal::copy(const ghost::Stream &s, const ghost::Buffer &src,
                          size_t bytes) {
  BufferMetal::copy(s, src, 0, 0, bytes);
}

void SubBufferMetal::copy(const ghost::Stream &s, const void *src,
                          size_t bytes) {
  BufferMetal::copy(s, src, 0, bytes);
}

void SubBufferMetal::copyTo(const ghost::Stream &s, void *dst,
                            size_t bytes) const {
  BufferMetal::copyTo(s, dst, 0, bytes);
}

void BufferMetal::fill(const ghost::Stream &s, size_t offset, size_t size,
                       const void *pattern, size_t patternSize) {
  // Metal only supports single-byte fill natively.
  // For multi-byte patterns, fill on CPU side if accessible, otherwise use
  // staging.
  if (patternSize == 1) {
    fill(s, offset, size, *static_cast<const uint8_t *>(pattern));
    return;
  }
  auto stream_impl = static_cast<implementation::StreamMetal *>(s.impl().get());
  size_t effectiveOff = baseOffset() + offset;
  // Build the fill pattern in a staging buffer
  id<MTLBuffer> staging =
      [mem.get().device newBufferWithLength:size
                                    options:MTLResourceStorageModeShared];
  uint8_t *dst = static_cast<uint8_t *>([staging contents]);
  for (size_t i = 0; i < size; i += patternSize) {
    size_t n = std::min(patternSize, size - i);
    memcpy(dst + i, pattern, n);
  }
  id<MTLCommandBuffer> commandBuffer = [stream_impl->queue.get() commandBuffer];
  commandBuffer.label = @"Ghost";
  stream_impl->encodeWait(commandBuffer);
  id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
  [blit copyFromBuffer:staging
           sourceOffset:0
               toBuffer:mem.get()
      destinationOffset:effectiveOff
                   size:size];
  [blit endEncoding];
  stream_impl->commitAndTrack(commandBuffer);
}

MappedBufferMetal::MappedBufferMetal(objc::ptr<id<MTLBuffer>> mem_,
                                     size_t bytes)
    : BufferMetal(mem_, bytes), length(bytes) {}

MappedBufferMetal::MappedBufferMetal(const DeviceMetal &dev, size_t bytes,
                                     Access access)
    : BufferMetal(objc::ptr<id<MTLBuffer>>(), bytes), length(bytes) {
  MTLResourceOptions options = MTLResourceCPUCacheModeDefaultCache |
                               MTLResourceHazardTrackingModeUntracked |
                               MTLResourceStorageModeShared;
  if (access == Access_WriteOnly)
    options |= MTLResourceCPUCacheModeWriteCombined;
  mem = [dev.dev newBufferWithLength:bytes options:options];
  checkExists(mem);
}

void *MappedBufferMetal::map(const ghost::Stream &s, Access access, bool sync) {
  if (sync) {
    auto stream_impl =
        static_cast<implementation::StreamMetal *>(s.impl().get());
    stream_impl->sync();
  }
  return [mem contents];
}

void MappedBufferMetal::unmap(const ghost::Stream &s) {
  [mem didModifyRange:NSMakeRange(0, length)];
}

ImageMetal::ImageMetal(objc::ptr<id<MTLTexture>> mem_,
                       const ImageDescription &descr_)
    : mem(mem_), descr(descr_) {}

ImageMetal::ImageMetal(const DeviceMetal &dev, const ImageDescription &descr_)
    : descr(descr_) {
  objc::ptr<MTLTextureDescriptor *> textureDescriptor(
      getTextureDescriptor(descr));
  mem = [dev.dev newTextureWithDescriptor:textureDescriptor.get()];
}

ImageMetal::ImageMetal(const DeviceMetal &dev, const ImageDescription &descr_,
                       BufferMetal &buffer)
    : descr(descr_) {
  objc::ptr<MTLTextureDescriptor *> textureDescriptor(
      getTextureDescriptor(buffer.mem.get(), descr));
  mem = [buffer.mem.get() newTextureWithDescriptor:textureDescriptor.get()
                                            offset:0
                                       bytesPerRow:descr.stride.x];
}

ImageMetal::ImageMetal(const DeviceMetal &dev, const ImageDescription &descr_,
                       ImageMetal &image)
    : mem(image.mem), descr(descr_) {
  mem = [image.mem.get() newTextureViewWithPixelFormat:getFormat(descr)];
}

void ImageMetal::copy(const ghost::Stream &s, const ghost::Image &src) {
  auto stream_impl = static_cast<implementation::StreamMetal *>(s.impl().get());
  auto src_impl = static_cast<implementation::ImageMetal *>(src.impl().get());
  id<MTLCommandBuffer> commandBuffer = [stream_impl->queue.get() commandBuffer];
  commandBuffer.label = @"Ghost";
  stream_impl->encodeWait(commandBuffer);
  MTLRegion region = {{0, 0, 0}, {descr.size.x, descr.size.y, descr.size.z}};
  MTLOrigin dst_origin = {0, 0, 0};
  id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
  [blit copyFromTexture:src_impl->mem.get()
            sourceSlice:0
            sourceLevel:0
           sourceOrigin:region.origin
             sourceSize:region.size
              toTexture:mem.get()
       destinationSlice:0
       destinationLevel:0
      destinationOrigin:dst_origin];
  [blit endEncoding];
  stream_impl->commitAndTrack(commandBuffer);
}

void ImageMetal::copy(const ghost::Stream &s, const ghost::Buffer &src,
                      const ImageDescription &descr_) {
  auto stream_impl = static_cast<implementation::StreamMetal *>(s.impl().get());
  auto src_impl = static_cast<implementation::BufferMetal *>(src.impl().get());
  id<MTLCommandBuffer> commandBuffer = [stream_impl->queue.get() commandBuffer];
  commandBuffer.label = @"Ghost";
  stream_impl->encodeWait(commandBuffer);
  MTLRegion region = {{0, 0, 0}, {descr.size.x, descr.size.y, descr.size.z}};
  MTLOrigin dst_origin = {0, 0, 0};
  id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
  [blit copyFromBuffer:src_impl->mem.get()
             sourceOffset:0
        sourceBytesPerRow:descr_.stride.x
      sourceBytesPerImage:descr_.stride.y
               sourceSize:region.size
                toTexture:mem.get()
         destinationSlice:0
         destinationLevel:0
        destinationOrigin:dst_origin];
  [blit endEncoding];
  stream_impl->commitAndTrack(commandBuffer);
}

void ImageMetal::copy(const ghost::Stream &s, const void *src,
                      const ImageDescription &descr_) {
  auto stream_impl = static_cast<implementation::StreamMetal *>(s.impl().get());
  if (!IsPrivate(mem.get())) {
    MTLRegion region = {{0, 0, 0},
                        {descr_.size.x, descr_.size.y, descr_.size.z}};
    [mem.get() replaceRegion:region
                 mipmapLevel:0
                   withBytes:src
                 bytesPerRow:descr_.stride.x];
    return;
  }
  // Private texture: upload via staging buffer + blit
  size_t dataSize = descr_.stride.x * descr_.size.y;
  id<MTLBuffer> staging =
      [mem.get().device newBufferWithBytes:src
                                    length:dataSize
                                   options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> commandBuffer = [stream_impl->queue.get() commandBuffer];
  commandBuffer.label = @"Ghost";
  stream_impl->encodeWait(commandBuffer);
  MTLRegion region = {{0, 0, 0}, {descr_.size.x, descr_.size.y, descr_.size.z}};
  MTLOrigin dst_origin = {0, 0, 0};
  id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
  [blit copyFromBuffer:staging
             sourceOffset:0
        sourceBytesPerRow:descr_.stride.x
      sourceBytesPerImage:dataSize
               sourceSize:region.size
                toTexture:mem.get()
         destinationSlice:0
         destinationLevel:0
        destinationOrigin:dst_origin];
  [blit endEncoding];
  stream_impl->commitAndTrack(commandBuffer);
}

void ImageMetal::copyTo(const ghost::Stream &s, ghost::Buffer &dst,
                        const ImageDescription &descr_) const {
  auto stream_impl = static_cast<implementation::StreamMetal *>(s.impl().get());
  auto dst_impl = static_cast<implementation::BufferMetal *>(dst.impl().get());
  id<MTLCommandBuffer> commandBuffer = [stream_impl->queue.get() commandBuffer];
  commandBuffer.label = @"Ghost";
  stream_impl->encodeWait(commandBuffer);
  MTLRegion region = {{0, 0, 0}, {descr.size.x, descr.size.y, descr.size.z}};
  MTLOrigin dst_origin = {0, 0, 0};
  id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
  [blit copyFromTexture:mem.get()
                   sourceSlice:0
                   sourceLevel:0
                  sourceOrigin:region.origin
                    sourceSize:region.size
                      toBuffer:dst_impl->mem.get()
             destinationOffset:0
        destinationBytesPerRow:descr_.stride.x
      destinationBytesPerImage:descr_.stride.y];
  [blit endEncoding];
  stream_impl->commitAndTrack(commandBuffer);
}

void ImageMetal::copyTo(const ghost::Stream &s, void *dst,
                        const ImageDescription &descr_) const {
  auto stream_impl = static_cast<implementation::StreamMetal *>(s.impl().get());
  // For non-private textures accessed from CPU, we need to sync pending GPU
  // work.
  if (!IsPrivate(mem.get())) {
    stream_impl->sync();
    // Shared/Managed: read directly from texture
    MTLRegion region = {{0, 0, 0},
                        {descr_.size.x, descr_.size.y, descr_.size.z}};
    [mem.get() getBytes:dst
            bytesPerRow:descr_.stride.x
             fromRegion:region
            mipmapLevel:0];
    return;
  }
  // Private texture: blit to staging buffer, wait, then memcpy
  size_t dataSize = descr_.stride.x * descr_.size.y;
  id<MTLBuffer> staging =
      [mem.get().device newBufferWithLength:dataSize
                                    options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> commandBuffer = [stream_impl->queue.get() commandBuffer];
  commandBuffer.label = @"Ghost";
  stream_impl->encodeWait(commandBuffer);
  MTLRegion region = {{0, 0, 0}, {descr_.size.x, descr_.size.y, descr_.size.z}};
  id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
  [blit copyFromTexture:mem.get()
                   sourceSlice:0
                   sourceLevel:0
                  sourceOrigin:region.origin
                    sourceSize:region.size
                      toBuffer:staging
             destinationOffset:0
        destinationBytesPerRow:descr_.stride.x
      destinationBytesPerImage:dataSize];
  [blit endEncoding];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
  memcpy(dst, [staging contents], dataSize);
}

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
                                                const std::string &options,
                                                bool retainBinary) const {
  auto ptr =
      std::make_shared<implementation::LibraryMetal>(*this, retainBinary);
  ptr->loadFromText(text, options);
  return ghost::Library(ptr);
}

ghost::Library DeviceMetal::loadLibraryFromData(const void *data, size_t len,
                                                const std::string &options,
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

ghost::Stream DeviceMetal::createStream() const {
  auto ptr = std::make_shared<implementation::StreamMetal>(dev.get());
  return ghost::Stream(ptr);
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

ghost::Buffer DeviceMetal::allocateBuffer(size_t bytes, Access access) const {
  if (heap) {
    MTLResourceOptions options = MTLResourceCPUCacheModeDefaultCache |
                                 MTLResourceHazardTrackingModeUntracked |
                                 MTLResourceStorageModePrivate;
    objc::ptr<id<MTLBuffer>> buf([heap.get() newBufferWithLength:bytes
                                                         options:options]);
    if (buf) {
      auto ptr = std::make_shared<implementation::BufferMetal>(buf, bytes);
      return ghost::Buffer(ptr);
    }
    // Heap full — fall through to individual allocation
  }
  auto ptr =
      std::make_shared<implementation::BufferMetal>(*this, bytes, access);
  return ghost::Buffer(ptr);
}

ghost::MappedBuffer DeviceMetal::allocateMappedBuffer(size_t bytes,
                                                      Access access) const {
  auto ptr =
      std::make_shared<implementation::MappedBufferMetal>(*this, bytes, access);
  return ghost::MappedBuffer(ptr);
}

ghost::Image DeviceMetal::allocateImage(const ImageDescription &descr) const {
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
  case kDeviceCount:
    return 1;
  case kDeviceProcessorCount:
    return 1;
  case kDeviceUnifiedMemory:
    return dev.get().hasUnifiedMemory;
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
      if ([dev.get() supportsFamily:MTLGPUFamilyApple10]) {
        v = 32786;
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
  case kDeviceImageAlignment: {
    ImageDescription descr(Size3(16, 16, 1), PixelOrder_RGBA, DataType_Float,
                           Stride2(0, 0));
    size_t size = 256;
    if (@available(macOS 10.13, iOS 11.0, tvOS 11.0, macCatalyst 13.1, *)) {
      size = [dev.get()
          minimumLinearTextureAlignmentForPixelFormat:getFormat(descr)];
    }
    return uint32_t(size / descr.pixelSize());
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
  default:
    return Attribute();
  }
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
