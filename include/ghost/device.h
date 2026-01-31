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

#ifndef GHOST_DEVICE_H
#define GHOST_DEVICE_H

#include <ghost/function.h>
#include <ghost/image.h>
#include <ghost/implementation/impl_device.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace ghost {
class Stream {
 public:
  Stream(std::shared_ptr<implementation::Stream> impl);

  std::shared_ptr<implementation::Stream> impl() const { return _impl; }

  std::shared_ptr<implementation::Stream>& impl() { return _impl; }

  void sync();

 private:
  std::shared_ptr<implementation::Stream> _impl;
};

class Buffer {
 public:
  Buffer(std::shared_ptr<implementation::Buffer> impl);

  std::shared_ptr<implementation::Buffer> impl() const { return _impl; }

  std::shared_ptr<implementation::Buffer>& impl() { return _impl; }

  void copy(const Stream& s, const Buffer& src, size_t bytes);
  void copy(const Stream& s, const void* src, size_t bytes);
  void copyTo(const Stream& s, void* dst, size_t bytes) const;

 private:
  std::shared_ptr<implementation::Buffer> _impl;
};

class MappedBuffer : public Buffer {
 public:
  MappedBuffer(std::shared_ptr<implementation::Buffer> impl);

  void* map(const Stream& s, Access access, bool sync = true);
  void unmap(const Stream& s);
};

class Image {
 public:
  Image(std::shared_ptr<implementation::Image> impl);

  std::shared_ptr<implementation::Image> impl() const { return _impl; }

  std::shared_ptr<implementation::Image>& impl() { return _impl; }

  void copy(const Stream& s, const Image& src);
  void copy(const Stream& s, const Buffer& src, const ImageDescription& descr);
  void copy(const Stream& s, const void* src, const ImageDescription& descr);
  void copyTo(const Stream& s, Buffer& dst,
              const ImageDescription& descr) const;
  void copyTo(const Stream& s, void* dst, const ImageDescription& descr) const;

 private:
  std::shared_ptr<implementation::Image> _impl;
};

class unsupported_error : public std::runtime_error {
 public:
  unsupported_error() : std::runtime_error("unsupported") {}
};

class Device {
 protected:
  Device(std::shared_ptr<implementation::Device> impl);

  void setDefaultStream(std::shared_ptr<implementation::Stream> stream);

 public:
  SharedContext shareContext() const;

  static BinaryCache& binaryCache();
  void purgeBinaries(int days = 30);
  Library loadLibraryFromFile(const std::string& filename);
  Library loadLibraryFromText(const std::string& text,
                              const std::string& options = "") const;
  Library loadLibraryFromData(const void* data, size_t len,
                              const std::string& options = "") const;

  Stream createStream() const;
  Stream defaultStream() const;

  size_t getMemoryPoolSize() const;
  void setMemoryPoolSize(size_t bytes) const;
  void* allocateHostMemory(size_t bytes) const;
  void freeHostMemory(void* ptr) const;
  Buffer allocateBuffer(size_t bytes, Access access = Access_ReadWrite) const;
  MappedBuffer allocateMappedBuffer(size_t bytes,
                                    Access access = Access_ReadWrite) const;
  Image allocateImage(const ImageDescription& descr) const;
  Image sharedImage(const ImageDescription& descr, Buffer& buffer) const;
  Image sharedImage(const ImageDescription& descr, Image& image) const;

  Attribute getAttribute(DeviceAttributeId what) const;

  std::shared_ptr<implementation::Device> impl() const { return _impl; }

  std::shared_ptr<implementation::Device>& impl() { return _impl; }

 private:
  std::shared_ptr<implementation::Device> _impl;
  Stream _stream;
};
}  // namespace ghost

#endif