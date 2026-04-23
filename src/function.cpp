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

#include <ghost/command_buffer.h>
#include <ghost/device.h>
#include <ghost/exception.h>
#include <ghost/function.h>

#include <memory>
#include <string>

namespace ghost {
namespace implementation {
ghost::Function Library::specializeFunction(
    const std::string& name, const std::vector<Attribute>& args) const {
  throw ghost::unsupported_error();
}

ghost::Function Library::specializeFunctionNamed(
    const std::string& name,
    const std::vector<std::pair<std::string, Attribute>>& constants) const {
  throw ghost::unsupported_error();
}

void Library::setGlobals(
    const std::vector<std::pair<std::string, Attribute>>& globals) {
  throw ghost::unsupported_error();
}

std::vector<uint8_t> Library::getBinary() const { return {}; }

void Function::executeIndirect(const ghost::Encoder& s,
                               const std::shared_ptr<Buffer>& indirectBuffer,
                               size_t indirectOffset,
                               const std::vector<Attribute>& args) {
  // Default fallback: sync, read workgroup counts, dispatch
  auto* stream = static_cast<implementation::Stream*>(s.impl().get());
  stream->sync();
  uint32_t counts[3];
  indirectBuffer->copyTo(s, counts, indirectOffset, sizeof(counts));
  stream->sync();
  LaunchArgs la;
  la.global_size(counts[0], counts[1], counts[2]);
  execute(s, la, args);
}

uint32_t Function::preferredSubgroupSize() const {
  return (uint32_t)getAttribute(kFunctionThreadWidth).asInt();
}
}  // namespace implementation

Function::Function(std::shared_ptr<implementation::Function> impl)
    : _impl(impl) {}

Function::BoundFunction::BoundFunction(
    std::shared_ptr<implementation::Function> impl,
    const LaunchArgs& launchArgs, const Encoder& encoder)
    : _impl(impl), _launchArgs(launchArgs), _encoder(encoder) {}

void Function::BoundFunction::dispatch(const std::vector<Attribute>& args) {
  auto* cb = _encoder.impl()->asCommandBuffer();
  if (cb)
    cb->dispatch(_impl, _launchArgs, args);
  else
    _impl->execute(_encoder, _launchArgs, args);
}

void Function::execute(const Encoder& s, const LaunchArgs& launchArgs,
                       const std::vector<Attribute>& args) {
  auto* cb = s.impl()->asCommandBuffer();
  if (cb)
    cb->dispatch(_impl, launchArgs, args);
  else
    _impl->execute(s, launchArgs, args);
}

Attribute Function::getAttribute(FunctionAttributeId what) const {
  return _impl->getAttribute(what);
}

uint32_t Function::preferredSubgroupSize() const {
  return _impl->preferredSubgroupSize();
}

Library::Library(std::shared_ptr<implementation::Library> impl) : _impl(impl) {}

Function Library::lookupFunction(const std::string& name) const {
  Function fn = _impl->lookupFunction(name);
  fn._parent = _impl;
  return fn;
}

void Library::setGlobals(
    const std::vector<std::pair<std::string, Attribute>>& globals) {
  _impl->setGlobals(globals);
}

std::vector<uint8_t> Library::getBinary() const { return _impl->getBinary(); }
}  // namespace ghost
