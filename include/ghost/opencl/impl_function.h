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

#ifndef GHOST_OPENCL_IMPL_FUNCTION_H
#define GHOST_OPENCL_IMPL_FUNCTION_H

#include <ghost/implementation/impl_function.h>

#include "ptr.h"

namespace ghost {
namespace implementation {
class DeviceOpenCL;

class FunctionOpenCL : public Function {
 public:
  opencl::ptr<cl_kernel> kernel;

  FunctionOpenCL(opencl::ptr<cl_kernel> k);

  virtual void execute(const ghost::Stream& s, const LaunchArgs& launchArgs,
                       const std::vector<Attribute>& args) override;
};

class LibraryOpenCL : public Library {
 public:
  opencl::ptr<cl_program> program;

  LibraryOpenCL(const DeviceOpenCL& dev);

  void loadFromText(const std::string& text, const std::string& options);
  void loadFromData(const void* data, size_t len, const std::string& options);
  void loadFromBinaries(const size_t* lengths, const unsigned char** binaries,
                        const std::string& options);
  virtual ghost::Function lookupFunction(
      const std::string& name) const override;

 private:
  void checkBuildLog(cl_int err0);
  void loadFromCache(const void* data, size_t length,
                     const std::string& options);
  void saveToCache(const void* data, size_t length,
                   const std::string& options) const;
  const DeviceOpenCL& _dev;
};
}  // namespace implementation
}  // namespace ghost

#endif
