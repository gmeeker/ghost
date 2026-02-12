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

#ifndef GHOST_CPU_IMPL_FUNCTION_H
#define GHOST_CPU_IMPL_FUNCTION_H

#include <ghost/implementation/impl_function.h>

namespace ghost {
namespace implementation {
class DeviceCPU;

class FunctionCPU : public Function {
 public:
  typedef void (*Type)(size_t i, size_t n, const std::vector<Attribute>& args);

  Type function;

  FunctionCPU(const DeviceCPU& dev, Type f);

  virtual void execute(const ghost::Stream& s, const LaunchArgs& launchArgs,
                       const std::vector<Attribute>& args) override;

  virtual Attribute getAttribute(FunctionAttributeId what) const override;

 private:
  const DeviceCPU& _dev;
};

class LibraryCPU : public Library {
 public:
  LibraryCPU(const DeviceCPU& dev);
  ~LibraryCPU();

  void loadFromFile(const std::string& filename);
  virtual ghost::Function lookupFunction(
      const std::string& name) const override;

 private:
  const DeviceCPU& _dev;
  void* _module;
};
}  // namespace implementation
}  // namespace ghost

#endif
