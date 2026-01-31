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

#ifndef GHOST_CUDA_IMPL_FUNCTION_H
#define GHOST_CUDA_IMPL_FUNCTION_H

#include <ghost/implementation/impl_function.h>

#include "cu_ptr.h"

namespace ghost {
namespace implementation {
class DeviceCUDA;

class FunctionCUDA : public Function {
 public:
  CUfunction kernel;

  FunctionCUDA(CUfunction k);

  virtual void execute(const ghost::Stream& s, const LaunchArgs& launchArgs,
                       const std::vector<Attribute>& args) override;
};

class LibraryCUDA : public Library {
 public:
  cu::ptr<CUmodule> program;

  LibraryCUDA(const DeviceCUDA& dev);

  void loadFromText(const std::string& text, const std::string& options);
  void loadFromData(const void* data, size_t len, const std::string& options);
  void loadFromBinary(void* binary);
  virtual ghost::Function lookupFunction(
      const std::string& name) const override;

 private:
  void loadFromCache(const void* data, size_t length,
                     const std::string& options);
  void saveToCache(void* binary, size_t binarySize, const void* data,
                   size_t length, const std::string& options) const;
  const DeviceCUDA& _dev;
};
}  // namespace implementation
}  // namespace ghost

#endif
