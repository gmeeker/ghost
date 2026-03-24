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

#include <ghost/argument_buffer.h>

namespace ghost {

ArgumentBuffer::ArgumentBuffer() : _gpuBuffer(nullptr) {}

void ArgumentBuffer::reset() {
  _data.clear();
  _gpuBuffer = Buffer(nullptr);
}

size_t ArgumentBuffer::size() const { return _data.size(); }

const void* ArgumentBuffer::data() const {
  return _data.empty() ? nullptr : _data.data();
}

void ArgumentBuffer::ensureSize(size_t minSize) {
  if (_data.size() < minSize) _data.resize(minSize, 0);
}

void ArgumentBuffer::upload(const Device& device, const Stream& stream) {
  if (_data.empty()) return;
  if (!_gpuBuffer.impl() || _gpuBuffer.size() < _data.size()) {
    _gpuBuffer = device.allocateBuffer(_data.size());
  }
  _gpuBuffer.copy(stream, _data.data(), _data.size());
}

bool ArgumentBuffer::isStruct() const { return !_gpuBuffer.impl(); }

std::shared_ptr<implementation::Buffer> ArgumentBuffer::bufferImpl() const {
  return _gpuBuffer.impl();
}

}  // namespace ghost
