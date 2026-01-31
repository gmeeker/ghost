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

#include <ghost/device.h>

#include <memory>
#include <string>

namespace ghost {
namespace implementation {
void* Buffer::map(const ghost::Stream&, Access, bool) {
  throw ghost::unsupported_error();
}

void Buffer::unmap(const ghost::Stream&) { throw ghost::unsupported_error(); }

BinaryCache Device::_cache;

BinaryCache& Device::binaryCache() { return _cache; }

ghost::Library Device::loadLibraryFromFile(const std::string& filename) const {
  throw ghost::unsupported_error();
}

void* Device::allocateHostMemory(size_t bytes) const { return ::malloc(bytes); }

void Device::freeHostMemory(void* ptr) const {
  if (ptr) ::free(ptr);
}

size_t Device::getMemoryPoolSize() const { return _poolSize; }

void Device::setMemoryPoolSize(size_t bytes) { _poolSize = bytes; }
}  // namespace implementation

Stream::Stream(std::shared_ptr<implementation::Stream> impl) : _impl(impl) {}

void Stream::sync() { impl()->sync(); }

Buffer::Buffer(std::shared_ptr<implementation::Buffer> impl) : _impl(impl) {}

void Buffer::copy(const Stream& s, const Buffer& src, size_t bytes) {
  _impl->copy(s, src, bytes);
}

void Buffer::copy(const Stream& s, const void* src, size_t bytes) {
  _impl->copy(s, src, bytes);
}

void Buffer::copyTo(const Stream& s, void* dst, size_t bytes) const {
  _impl->copyTo(s, dst, bytes);
}

MappedBuffer::MappedBuffer(std::shared_ptr<implementation::Buffer> impl)
    : Buffer(impl) {}

void* MappedBuffer::map(const Stream& s, Access access, bool sync) {
  return impl()->map(s, access, sync);
}

void MappedBuffer::unmap(const Stream& s) { impl()->unmap(s); }

Image::Image(std::shared_ptr<implementation::Image> impl) : _impl(impl) {}

void Image::copy(const Stream& s, const Image& src) { _impl->copy(s, src); }

void Image::copy(const Stream& s, const Buffer& src,
                 const ImageDescription& descr) {
  _impl->copy(s, src, descr);
}

void Image::copy(const Stream& s, const void* src,
                 const ImageDescription& descr) {
  _impl->copy(s, src, descr);
}

void Image::copyTo(const Stream& s, Buffer& dst,
                   const ImageDescription& descr) const {
  _impl->copyTo(s, dst, descr);
}

void Image::copyTo(const Stream& s, void* dst,
                   const ImageDescription& descr) const {
  _impl->copyTo(s, dst, descr);
}

Device::Device(std::shared_ptr<implementation::Device> impl)
    : _impl(impl), _stream(nullptr) {}

void Device::setDefaultStream(std::shared_ptr<implementation::Stream> stream) {
  _stream.impl() = stream;
}

BinaryCache& Device::binaryCache() {
  return implementation::Device::binaryCache();
}

void Device::purgeBinaries(int days) {
  binaryCache().purgeBinaries(*_impl, days);
}

Library Device::loadLibraryFromFile(const std::string& filename) {
  return _impl->loadLibraryFromFile(filename);
}

Library Device::loadLibraryFromText(const std::string& text,
                                    const std::string& options) const {
  return _impl->loadLibraryFromText(text, options);
}

Library Device::loadLibraryFromData(const void* data, size_t len,
                                    const std::string& options) const {
  return _impl->loadLibraryFromData(data, len, options);
}

SharedContext Device::shareContext() const { return _impl->shareContext(); }

Stream Device::createStream() const { return _impl->createStream(); }

Stream Device::defaultStream() const { return _stream; }

size_t Device::getMemoryPoolSize() const { return _impl->getMemoryPoolSize(); }

void Device::setMemoryPoolSize(size_t bytes) const {
  _impl->setMemoryPoolSize(bytes);
}

void* Device::allocateHostMemory(size_t bytes) const {
  return _impl->allocateHostMemory(bytes);
}

void Device::freeHostMemory(void* ptr) const { _impl->freeHostMemory(ptr); }

Buffer Device::allocateBuffer(size_t bytes, Access access) const {
  return _impl->allocateBuffer(bytes, access);
}

MappedBuffer Device::allocateMappedBuffer(size_t bytes, Access access) const {
  return _impl->allocateMappedBuffer(bytes, access);
}

Image Device::allocateImage(const ImageDescription& descr) const {
  return _impl->allocateImage(descr);
}

Image Device::sharedImage(const ImageDescription& descr, Buffer& buffer) const {
  return _impl->sharedImage(descr, buffer);
}

Image Device::sharedImage(const ImageDescription& descr, Image& image) const {
  return _impl->sharedImage(descr, image);
}

Attribute Device::getAttribute(DeviceAttributeId what) const {
  return _impl->getAttribute(what);
}

}  // namespace ghost
