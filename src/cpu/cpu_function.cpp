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

#include <ghost/cpu/device.h>
#include <ghost/cpu/impl_device.h>
#include <ghost/cpu/impl_function.h>
#include <ghost/digest.h>
#include <ghost/function.h>
#include <ghost/io.h>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <vector>

#if WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ghost {
namespace implementation {

FunctionCPU::FunctionCPU(const DeviceCPU& dev, Type f)
    : function(f), _dev(dev) {}

void FunctionCPU::execute(const ghost::Encoder& s, const LaunchArgs& launchArgs,
                          const std::vector<Attribute>& args) {
  if (launchArgs.requiredSubgroupSize() != 0 &&
      launchArgs.requiredSubgroupSize() != 1) {
    throw std::invalid_argument("CPU backend: requiredSubgroupSize must be 1");
  }
  size_t count = launchArgs.count();
  count = std::min(count, _dev.cores);
  auto stream = static_cast<const StreamCPU*>(s.impl().get());
  FunctionCPU::Type fn = function;
  stream->pool->parallel(count,
                         [fn, &args](size_t i, size_t n) { fn(i, n, args); });
}

Attribute FunctionCPU::getAttribute(FunctionAttributeId what) const {
  switch (what) {
    case kFunctionLocalMemory:
      return 0;
    case kFunctionMaxLocalMemory:
      return 0;
    case kFunctionThreadWidth:
      return 1;
    case kFunctionMaxThreads:
      return 1024;
    case kFunctionRequiredWorkSize:
      return Attribute(0, 0, 0);
    case kFunctionPreferredWorkMultiple:
      return 1;
    case kFunctionNumRegisters:
      return 0;
    case kFunctionPrivateMemory:
      return 0;
    default:
      return Attribute();
  }
}

LibraryCPU::LibraryCPU(const DeviceCPU& dev) : _dev(dev), _module(nullptr) {}

LibraryCPU::~LibraryCPU() {
  if (_module) {
#if WIN32
    FreeLibrary((HMODULE)_module);
#else
    dlclose(_module);
#endif
    _module = nullptr;
  }
}

void LibraryCPU::loadFromFile(const std::filesystem::path& filename) {
#if WIN32
  _module = (HMODULE)LoadLibraryA(filename.string().c_str());
#else
  _module = dlopen(filename.string().c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
}

ghost::Function LibraryCPU::lookupFunction(const std::string& name) const {
#if WIN32
  auto f = std::make_shared<FunctionCPU>(
      _dev, reinterpret_cast<FunctionCPU::Type>(
                GetProcAddress((HMODULE)_module, name.c_str())));
  return ghost::Function(f);
#else
  auto f = std::make_shared<FunctionCPU>(
      _dev, reinterpret_cast<FunctionCPU::Type>(dlsym(_module, name.c_str())));
  return ghost::Function(f);
#endif
}

InlineLibraryCPU::InlineLibraryCPU(const DeviceCPU& dev) : _dev(dev) {}

void InlineLibraryCPU::addFunction(const std::string& name,
                                   FunctionCPU::Type fn) {
  _functions[name] = fn;
}

ghost::Function InlineLibraryCPU::lookupFunction(
    const std::string& name) const {
  auto it = _functions.find(name);
  if (it == _functions.end()) {
    throw std::runtime_error("Function not found: " + name);
  }
  return ghost::Function(std::make_shared<FunctionCPU>(_dev, it->second));
}
}  // namespace implementation
}  // namespace ghost
