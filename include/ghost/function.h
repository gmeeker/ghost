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

#ifndef GHOST_FUNCTION_H
#define GHOST_FUNCTION_H

#include <ghost/implementation/impl_function.h>

#include <memory>
#include <string>

namespace ghost {
class LaunchArgs {
 private:
  uint32_t _dims;
  uint32_t _global_size[3];
  uint32_t _local_size[3];
  bool _local_defined;

 public:
  uint32_t dims() const { return _dims; }

  const uint32_t* global_size() const { return _global_size; }

  const uint32_t* local_size() const { return _local_size; }

  bool is_local_defined() const { return _local_defined; }

  size_t count(uint32_t i) const {
    return size_t((global_size()[i] + local_size()[i] - 1) / local_size()[i]);
  }

  size_t count() const {
    size_t v = 1;
    for (uint32_t i = 0; i < dims(); i++) {
      v *= count(i);
    }
    return v;
  }

  LaunchArgs() : _dims(0), _local_defined(false) {
    _global_size[0] = 1;
    _global_size[1] = 1;
    _global_size[2] = 1;
    _local_size[0] = 1;
    _local_size[1] = 1;
    _local_size[2] = 1;
  }

  LaunchArgs& global_size(uint32_t v0) {
    _dims = 1;
    _global_size[0] = v0;
    return *this;
  }

  LaunchArgs& global_size(uint32_t v0, uint32_t v1) {
    _dims = 2;
    _global_size[0] = v0;
    _global_size[1] = v1;
    return *this;
  }

  LaunchArgs& global_size(uint32_t v0, uint32_t v1, uint32_t v2) {
    _dims = 3;
    _global_size[0] = v0;
    _global_size[1] = v1;
    _global_size[2] = v2;
    return *this;
  }

  LaunchArgs& local_size(uint32_t v0) {
    _dims = 1;
    _local_size[0] = v0;
    _local_defined = true;
    return *this;
  }

  LaunchArgs& local_size(uint32_t v0, uint32_t v1) {
    _dims = 2;
    _local_size[0] = v0;
    _local_size[1] = v1;
    _local_defined = true;
    return *this;
  }

  LaunchArgs& local_size(uint32_t v0, uint32_t v1, uint32_t v2) {
    _dims = 3;
    _local_size[0] = v0;
    _local_size[1] = v1;
    _local_size[2] = v2;
    _local_defined = true;
    return *this;
  }
};

class Function {
 public:
  Function(std::shared_ptr<implementation::Function> impl);

  std::shared_ptr<implementation::Function> impl() const { return _impl; }

  std::shared_ptr<implementation::Function>& impl() { return _impl; }

  template <typename... ARGS>
  void operator()(const Stream& s, const LaunchArgs& launchArgs,
                  ARGS&&... args) {
    (*_impl)(s, launchArgs, std::forward<ARGS>(args)...);
  }

  Attribute getAttribute(FunctionAttributeId what) const;

 private:
  std::shared_ptr<implementation::Function> _impl;
};

class Library {
 public:
  Library(std::shared_ptr<implementation::Library> impl);

  Function lookupFunction(const std::string& name) const;

  template <typename... ARGS>
  Function lookupFunction(const std::string& name, ARGS&&... args) {
    return _impl->lookupFunction(name, std::forward<ARGS>(args)...);
  }

  template <typename... ARGS>
  Function lookupSpecializedFunction(const std::string& name, ARGS&&... tail) {
    std::vector<Attribute> args;
    implementation::Function::addArgs(args, std::forward<ARGS>(tail)...);
    return _impl->specializeFunction(name, args);
  }

 protected:
  std::shared_ptr<implementation::Library> impl() const { return _impl; }

  std::shared_ptr<implementation::Library>& impl() { return _impl; }

 private:
  std::shared_ptr<implementation::Library> _impl;
};
}  // namespace ghost

#endif