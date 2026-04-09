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

#include <cstring>
#include <memory>
#include <string>

namespace ghost {
namespace implementation {
double Event::timestamp() const { return 0.0; }

double Event::elapsed(const Event&) const { return 0.0; }

std::shared_ptr<Event> Stream::record() { throw ghost::unsupported_error(); }

void Stream::waitForEvent(const std::shared_ptr<Event>&) {
  throw ghost::unsupported_error();
}

size_t Buffer::baseOffset() const { return 0; }

std::shared_ptr<Buffer> Buffer::createSubBuffer(const std::shared_ptr<Buffer>&,
                                                size_t, size_t) {
  throw ghost::unsupported_error();
}

void* Buffer::map(const ghost::Stream&, Access, bool) {
  throw ghost::unsupported_error();
}

void Buffer::unmap(const ghost::Stream&) { throw ghost::unsupported_error(); }

void Buffer::copy(const ghost::Stream&, const ghost::Buffer&, size_t, size_t,
                  size_t) {
  throw ghost::unsupported_error();
}

void Buffer::copy(const ghost::Stream&, const void*, size_t, size_t) {
  throw ghost::unsupported_error();
}

void Buffer::copyTo(const ghost::Stream&, void*, size_t, size_t) const {
  throw ghost::unsupported_error();
}

void Buffer::fill(const ghost::Stream&, size_t, size_t, uint8_t) {
  throw ghost::unsupported_error();
}

void Buffer::fill(const ghost::Stream&, size_t, size_t, const void*, size_t) {
  throw ghost::unsupported_error();
}

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

void Image::copy(const ghost::Stream& s, const ghost::Buffer& src,
                 const BufferLayout& layout, const Origin3& imageOrigin) {
  throw ghost::unsupported_error();
}

void Image::copyTo(const ghost::Stream& s, ghost::Buffer& dst,
                   const BufferLayout& layout,
                   const Origin3& imageOrigin) const {
  throw ghost::unsupported_error();
}

void Image::copy(const ghost::Stream& s, const ghost::Image& src,
                 const Size3& region, const Origin3& srcOrigin,
                 const Origin3& dstOrigin) {
  throw ghost::unsupported_error();
}
}  // namespace implementation

Event::Event(std::shared_ptr<implementation::Event> impl) : _impl(impl) {}

void Event::wait() { _impl->wait(); }

bool Event::isComplete() const { return _impl->isComplete(); }

double Event::timestamp() const { return _impl->timestamp(); }

double Event::elapsed(const Event& start, const Event& end) {
  return start._impl->elapsed(*end._impl);
}

Stream::Stream(std::shared_ptr<implementation::Stream> impl) : _impl(impl) {}

void Stream::sync() { impl()->sync(); }

Event Stream::record() { return Event(_impl->record()); }

void Stream::waitForEvent(const Event& e) { _impl->waitForEvent(e.impl()); }

Buffer::Buffer(std::shared_ptr<implementation::Buffer> impl) : _impl(impl) {}

size_t Buffer::size() const { return _impl->size(); }

void Buffer::copy(const Stream& s, const Buffer& src, size_t bytes) {
  _impl->copy(s, src, bytes);
}

void Buffer::copy(const Stream& s, const void* src, size_t bytes) {
  _impl->copy(s, src, bytes);
}

void Buffer::copyTo(const Stream& s, void* dst, size_t bytes) const {
  _impl->copyTo(s, dst, bytes);
}

void Buffer::copy(const Stream& s, const Buffer& src, size_t srcOffset,
                  size_t dstOffset, size_t bytes) {
  _impl->copy(s, src, srcOffset, dstOffset, bytes);
}

void Buffer::copy(const Stream& s, const void* src, size_t dstOffset,
                  size_t bytes) {
  _impl->copy(s, src, dstOffset, bytes);
}

void Buffer::copyTo(const Stream& s, void* dst, size_t srcOffset,
                    size_t bytes) const {
  _impl->copyTo(s, dst, srcOffset, bytes);
}

void Buffer::fill(const Stream& s, size_t offset, size_t size, uint8_t value) {
  _impl->fill(s, offset, size, value);
}

void Buffer::fill(const Stream& s, size_t offset, size_t size, uint32_t value) {
  _impl->fill(s, offset, size, &value, sizeof(value));
}

void Buffer::fill(const Stream& s, size_t offset, size_t size,
                  const void* pattern, size_t patternSize) {
  _impl->fill(s, offset, size, pattern, patternSize);
}

Buffer Buffer::createSubBuffer(size_t offset, size_t size) {
  return Buffer(_impl->createSubBuffer(_impl, offset, size));
}

MappedBuffer::MappedBuffer(std::shared_ptr<implementation::Buffer> impl)
    : Buffer(impl) {}

void* MappedBuffer::map(const Stream& s, Access access, bool sync) {
  return impl()->map(s, access, sync);
}

void MappedBuffer::unmap(const Stream& s) { impl()->unmap(s); }

Image::Image(std::shared_ptr<implementation::Image> impl) : _impl(impl) {}

const ImageDescription& Image::description() const {
  return _impl->description();
}

void Image::copy(const Stream& s, const Image& src) { _impl->copy(s, src); }

void Image::copy(const Stream& s, const Buffer& src) {
  _impl->copy(s, src, BufferLayout(description().size));
}

void Image::copy(const Stream& s, const void* src) {
  _impl->copy(s, src, BufferLayout(description().size));
}

void Image::copyTo(const Stream& s, Buffer& dst) const {
  _impl->copyTo(s, dst, BufferLayout(description().size));
}

void Image::copyTo(const Stream& s, void* dst) const {
  _impl->copyTo(s, dst, BufferLayout(description().size));
}

void Image::copy(const Stream& s, const Buffer& src,
                 const BufferLayout& layout) {
  _impl->copy(s, src, layout);
}

void Image::copy(const Stream& s, const void* src, const BufferLayout& layout) {
  _impl->copy(s, src, layout);
}

void Image::copyTo(const Stream& s, Buffer& dst,
                   const BufferLayout& layout) const {
  _impl->copyTo(s, dst, layout);
}

void Image::copyTo(const Stream& s, void* dst,
                   const BufferLayout& layout) const {
  _impl->copyTo(s, dst, layout);
}

void Image::copy(const Stream& s, const Buffer& src, const BufferLayout& layout,
                 const Origin3& imageOrigin) {
  _impl->copy(s, src, layout, imageOrigin);
}

void Image::copyTo(const Stream& s, Buffer& dst, const BufferLayout& layout,
                   const Origin3& imageOrigin) const {
  _impl->copyTo(s, dst, layout, imageOrigin);
}

void Image::copy(const Stream& s, const Image& src, const Size3& region,
                 const Origin3& srcOrigin, const Origin3& dstOrigin) {
  _impl->copy(s, src, region, srcOrigin, dstOrigin);
}

Device::Device(std::shared_ptr<implementation::Device> impl) : _impl(impl) {}

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
                                    const CompilerOptions& options,
                                    bool retainBinary) const {
  return _impl->loadLibraryFromText(text, options, retainBinary);
}

Library Device::loadLibraryFromData(const void* data, size_t len,
                                    const CompilerOptions& options,
                                    bool retainBinary) const {
  return _impl->loadLibraryFromData(data, len, options, retainBinary);
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

Buffer Device::allocateBuffer(size_t bytes, BufferOptions opts) const {
  return _impl->allocateBuffer(bytes, opts);
}

MappedBuffer Device::allocateMappedBuffer(size_t bytes,
                                          BufferOptions opts) const {
  return _impl->allocateMappedBuffer(bytes, opts);
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
