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

StreamMetal::StreamMetal(objc::ptr<id<MTLCommandQueue>> queue_)
    : queue(queue_) {}
StreamMetal::StreamMetal(id<MTLDevice> dev) {
  queue = [dev newCommandQueue];
  checkExists(queue);
}

void StreamMetal::sync() {}

BufferMetal::BufferMetal(objc::ptr<id<MTLBuffer>> mem_) : mem(mem_) {}

BufferMetal::BufferMetal(const DeviceMetal &dev, size_t bytes, Access access) {
  MTLResourceOptions options = MTLResourceCPUCacheModeDefaultCache |
                               MTLResourceHazardTrackingModeUntracked |
                               MTLResourceStorageModePrivate;
  mem = [dev.dev newBufferWithLength:bytes options:options];
  checkExists(mem);
}

void BufferMetal::copy(const ghost::Stream &s, const ghost::Buffer &src,
                       size_t bytes) {
  auto stream_impl = static_cast<implementation::StreamMetal *>(s.impl().get());
  auto src_impl = static_cast<implementation::BufferMetal *>(src.impl().get());
  id<MTLCommandBuffer> commandBuffer = nil;
  commandBuffer = [stream_impl->queue.get() commandBuffer];
  commandBuffer.label = @"Ghost";
  id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
  [blit copyFromBuffer:src_impl->mem.get()
           sourceOffset:0
               toBuffer:mem.get()
      destinationOffset:0
                   size:bytes];
  [blit endEncoding];
}

void BufferMetal::copy(const ghost::Stream &s, const void *src, size_t bytes) {
  auto stream_impl = static_cast<implementation::StreamMetal *>(s.impl().get());
  if (IsPrivate(mem)) {
  } else {
    void *dst = [mem contents];
    memcpy(dst, src, bytes);
    if (mem.get().storageMode == MTLStorageModeManaged)
      [mem didModifyRange:NSMakeRange(0, bytes)];
  }
}

void BufferMetal::copyTo(const ghost::Stream &s, void *dst,
                         size_t bytes) const {
  auto stream_impl = static_cast<implementation::StreamMetal *>(s.impl().get());
  if (IsPrivate(mem)) {
    if (mem.get().storageMode == MTLStorageModeManaged) {
      id<MTLCommandBuffer> commandBuffer = nil;
      commandBuffer = [stream_impl->queue.get() commandBuffer];
      commandBuffer.label = @"Ghost";
      id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
      [blit synchronizeResource:mem];
      [blit endEncoding];
    }
  } else {
    const void *src = [mem contents];
    memcpy(dst, src, bytes);
  }
}

MappedBufferMetal::MappedBufferMetal(objc::ptr<id<MTLBuffer>> mem_,
                                     size_t bytes)
    : BufferMetal(mem_), length(bytes) {}

MappedBufferMetal::MappedBufferMetal(const DeviceMetal &dev, size_t bytes,
                                     Access access)
    : BufferMetal(objc::ptr<id<MTLBuffer>>()), length(bytes) {
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
    // TODO
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
  id<MTLCommandBuffer> commandBuffer = nil;
  commandBuffer = [stream_impl->queue.get() commandBuffer];
  commandBuffer.label = @"Ghost";
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
}

void ImageMetal::copy(const ghost::Stream &s, const ghost::Buffer &src,
                      const ImageDescription &descr_) {
  auto stream_impl = static_cast<implementation::StreamMetal *>(s.impl().get());
  auto src_impl = static_cast<implementation::BufferMetal *>(src.impl().get());
  id<MTLCommandBuffer> commandBuffer = nil;
  commandBuffer = [stream_impl->queue.get() commandBuffer];
  commandBuffer.label = @"Ghost";
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
}

void ImageMetal::copy(const ghost::Stream &s, const void *src,
                      const ImageDescription &descr_) {
  if (!IsPrivate(mem.get())) {
    // d->wait(*ctx);
    MTLRegion region = {{0, 0, 0},
                        {descr_.size.x, descr_.size.y, descr_.size.z}};
    [mem.get() replaceRegion:region
                 mipmapLevel:0
                   withBytes:src
                 bytesPerRow:descr_.stride.x];
    // d->dirty = false; // This updates the CPU side first.
    return;
  }
}

void ImageMetal::copyTo(const ghost::Stream &s, ghost::Buffer &dst,
                        const ImageDescription &descr_) const {
  auto stream_impl = static_cast<implementation::StreamMetal *>(s.impl().get());
  auto dst_impl = static_cast<implementation::BufferMetal *>(dst.impl().get());
  id<MTLCommandBuffer> commandBuffer = nil;
  commandBuffer = [stream_impl->queue.get() commandBuffer];
  commandBuffer.label = @"Ghost";
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
}

void ImageMetal::copyTo(const ghost::Stream &s, void *dst,
                        const ImageDescription &descr_) const {}

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

ghost::Library
DeviceMetal::loadLibraryFromText(const std::string &text,
                                 const std::string &options) const {
  auto ptr = std::make_shared<implementation::LibraryMetal>(*this);
  ptr->loadFromText(text, options);
  return ghost::Library(ptr);
}

ghost::Library
DeviceMetal::loadLibraryFromData(const void *data, size_t len,
                                 const std::string &options) const {
  auto ptr = std::make_shared<implementation::LibraryMetal>(*this);
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

size_t DeviceMetal::getMemoryPoolSize() const {}

void DeviceMetal::setMemoryPoolSize(size_t bytes) {}

ghost::Buffer DeviceMetal::allocateBuffer(size_t bytes, Access access) const {
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
  if (@available(macOS 26, iOS 26.0, *)) {
    return "4.0";
  }
#endif
#if defined(MAC_OS_VERSION_15_0)
  if (@available(macOS 15, iOS 18.0, *)) {
    return "3.2";
  }
#endif
#if defined(MAC_OS_VERSION_14_0)
  if (@available(macOS 14, iOS 17.0, *)) {
    return "3.1";
  }
#endif
#if defined(MAC_OS_VERSION_13_0)
  if (@available(macOS 13, iOS 16.0, *)) {
    return "3.0";
  }
#endif
#if defined(MAC_OS_VERSION_12_0)
  if (@available(macOS 12, iOS 15.0, *)) {
    return "2.4";
  }
#endif
#if defined(MAC_OS_VERSION_11_0)
  if (@available(macOS 11, iOS 14.0, *)) {
    return "2.3";
  }
#endif
  if (@available(macOS 10.15, iOS 13.0, *)) {
    return "2.2";
  }
  if (@available(macOS 10.14, iOS 12.0, *)) {
    return "2.1";
  }
  if (@available(macOS 10.13, iOS 11.0, *)) {
    return "2.0";
  }
  if (@available(macOS 10.12, iOS 10.0, *)) {
    return "1.2";
  }
  if (@available(macOS 10.11, iOS 9.0, *)) {
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
    return Attribute("Metal");
  case kDeviceName:
    return Attribute([[dev.get() name] UTF8String]);
  case kDeviceVendor:
    return Attribute("Apple");
  case kDeviceDriverVersion:
    return Attribute(getMetalVersion());
  case kDeviceCount:
    return Attribute(1);
  case kDeviceSupportsMappedBuffer:
    return Attribute(true);
  case kDeviceSupportsProgramConstants:
    return Attribute(true);
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
} // namespace ghost
#endif

// vim: ts=2:sw=2:et:ft=mm
// -*- mode: objective-c++; indent-tabs-mode: nil; tab-width: 2 -*-
// code: language=objective-c++ insertSpaces=true tabSize=2
