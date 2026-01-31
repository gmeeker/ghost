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
#include <vector>

#if WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ghost {
namespace implementation {

FunctionCPU::FunctionCPU(const DeviceCPU& dev, Type f)
    : function(f), _dev(dev) {}

void FunctionCPU::execute(const ghost::Stream& s, const LaunchArgs& launchArgs,
                          const std::vector<Attribute>& args) {
  size_t count = launchArgs.count();
  count = std::min(count, _dev.cores);
  auto stream = static_cast<const StreamCPU*>(s.impl().get());
  stream->pool->thread(count, function, args);
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

void LibraryCPU::loadFromFile(const std::string& filename) {
#if WIN32
  _module = (HMODULE)LoadLibraryA(filename.c_str());
#else
  _module = dlopen(filename.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
}

ghost::Function LibraryCPU::lookupFunction(const std::string& name) const {
#if WIN32
  auto f = std::make_shared<FunctionCPU>(reinterpret_cast<FunctionCPU::Type>(
      _dev, GetProcAddress((HMODULE)_module, name.c_str())));
  return ghost::Function(f);
#else
  auto f = std::make_shared<FunctionCPU>(
      _dev, reinterpret_cast<FunctionCPU::Type>(dlsym(_module, name.c_str())));
  return ghost::Function(f);
#endif
}
}  // namespace implementation
}  // namespace ghost
