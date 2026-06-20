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

#include <ghost/exception.h>
#include <ghost/executable.h>
#include <ghost/implementation/executable.h>
#include <ghost/implementation/recorded_command_buffer.h>

namespace ghost {

Executable::Executable(std::shared_ptr<implementation::Executable> impl)
    : _impl(std::move(impl)) {}

void Executable::submit(const Stream& stream) {
  if (!_impl) throw unsupported_error();
  _impl->submit(stream);
}

void Executable::update(const CommandBuffer& cb) {
  if (!_impl) throw unsupported_error();
  auto* cmdBuf = cb.impl()->asCommandBuffer();
  if (!cmdBuf) throw unsupported_error();
  // Every backend command buffer derives from RecordedCommandBuffer, which
  // owns the recorded command snapshot the Executable rebinds from.
  auto* recorded = static_cast<implementation::RecordedCommandBuffer*>(cmdBuf);
  _impl->update(recorded->commands);
}

bool Executable::accelerated() const { return _impl && _impl->accelerated(); }

bool Executable::lastUpdatePatched() const {
  return _impl && _impl->lastUpdatePatched();
}

}  // namespace ghost
