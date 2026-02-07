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

#ifndef GHOST_IMPL_DEVICE_H
#define GHOST_IMPL_DEVICE_H

#include <ghost/attribute.h>
#include <ghost/binary_cache.h>
#include <ghost/image.h>
#include <ghost/implementation/impl_function.h>
#include <stdlib.h>

#include <memory>
#include <string>

namespace ghost {
enum DeviceAttributeId {
  kDeviceImplementation,
  kDeviceName,
  kDeviceVendor,
  kDeviceDriverVersion,
  kDeviceCount,
  kDeviceSupportsMappedBuffer,
  kDeviceSupportsProgramConstants,
};

class SharedContext {
 public:
  void* context;
  void* queue;
  void* device;
  void* platform;

  SharedContext(void* context_ = nullptr, void* queue_ = nullptr,
                void* device_ = nullptr, void* platform_ = nullptr)
      : context(context_),
        queue(queue_),
        device(device_),
        platform(platform_) {}
};

class Function;
class Library;
class Stream;
class Buffer;
class MappedBuffer;
class Image;

namespace implementation {
class Stream {
 protected:
  Stream() {}

  Stream(const Stream& rhs) = delete;

  virtual ~Stream() {}

  Stream& operator=(const Stream& rhs) = delete;

 public:
  virtual void sync() = 0;
};

class Buffer {
 protected:
  Buffer() {}

  Buffer(const Buffer& rhs) = delete;

  virtual ~Buffer() {}

  Buffer& operator=(const Buffer& rhs) = delete;

 public:
  virtual void copy(const ghost::Stream& s, const ghost::Buffer& src,
                    size_t bytes) = 0;
  virtual void copy(const ghost::Stream& s, const void* src, size_t bytes) = 0;
  virtual void copyTo(const ghost::Stream& s, void* dst,
                      size_t bytes) const = 0;
  virtual void* map(const ghost::Stream& s, Access access, bool sync = true);
  virtual void unmap(const ghost::Stream& s);
};

class Image {
 protected:
  Image() {}

  Image(const Image& rhs) = delete;

  virtual ~Image() {}

  Image& operator=(const Image& rhs) = delete;

 public:
  virtual void copy(const ghost::Stream& s, const ghost::Image& src) = 0;
  virtual void copy(const ghost::Stream& s, const ghost::Buffer& src,
                    const ImageDescription& descr) = 0;
  virtual void copy(const ghost::Stream& s, const void* src,
                    const ImageDescription& descr) = 0;
  virtual void copyTo(const ghost::Stream& s, ghost::Buffer& dst,
                      const ImageDescription& descr) const = 0;
  virtual void copyTo(const ghost::Stream& s, void* dst,
                      const ImageDescription& descr) const = 0;
};

class Device {
 protected:
  Device() : _poolSize(0) {}

  Device(const Device& rhs) = delete;

  virtual ~Device() {}

  Device& operator=(const Device& rhs) = delete;

 public:
  static BinaryCache& binaryCache();
  virtual ghost::Library loadLibraryFromText(
      const std::string& text, const std::string& options = "") const = 0;
  virtual ghost::Library loadLibraryFromData(
      const void* data, size_t len, const std::string& options = "") const = 0;
  virtual ghost::Library loadLibraryFromFile(const std::string& filename) const;

  virtual SharedContext shareContext() const = 0;

  virtual ghost::Stream createStream() const = 0;

  virtual size_t getMemoryPoolSize() const;
  virtual void setMemoryPoolSize(size_t bytes);
  virtual void* allocateHostMemory(size_t bytes) const;
  virtual void freeHostMemory(void* ptr) const;
  virtual ghost::Buffer allocateBuffer(
      size_t bytes, Access access = Access_ReadWrite) const = 0;
  virtual ghost::MappedBuffer allocateMappedBuffer(
      size_t bytes, Access access = Access_ReadWrite) const = 0;
  virtual ghost::Image allocateImage(const ImageDescription& descr) const = 0;
  virtual ghost::Image sharedImage(const ImageDescription& descr,
                                   ghost::Buffer& buffer) const = 0;
  virtual ghost::Image sharedImage(const ImageDescription& descr,
                                   ghost::Image& image) const = 0;

  virtual Attribute getAttribute(DeviceAttributeId what) const = 0;

 private:
  size_t _poolSize;
  static BinaryCache _cache;
};
}  // namespace implementation
}  // namespace ghost

#endif
