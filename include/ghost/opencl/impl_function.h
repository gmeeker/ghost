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
#include <ghost/opencl/ptr.h>

namespace ghost {
namespace implementation {
class DeviceOpenCL;

class FunctionOpenCL : public Function {
 public:
  opencl::ptr<cl_kernel> kernel;

  FunctionOpenCL(const DeviceOpenCL& dev, opencl::ptr<cl_kernel> k);

  virtual void execute(const ghost::Encoder& s, const LaunchArgs& launchArgs,
                       const std::vector<Attribute>& args) override;

  virtual Attribute getAttribute(FunctionAttributeId what) const override;

  virtual uint32_t preferredSubgroupSize() const override;

 private:
  const DeviceOpenCL& _dev;
};

class LibraryOpenCL : public Library {
 public:
  opencl::ptr<cl_program> program;

  LibraryOpenCL(const DeviceOpenCL& dev);

  void loadFromText(const std::string& text, const CompilerOptions& options);
  void loadFromData(const void* data, size_t len,
                    const CompilerOptions& options);
  void loadFromBinaries(const size_t* lengths, const unsigned char** binaries,
                        const CompilerOptions& options);
  virtual ghost::Function lookupFunction(
      const std::string& name) const override;
  virtual void setGlobals(
      const std::vector<std::pair<std::string, Attribute>>& globals) override;
  virtual std::vector<uint8_t> getBinary() const override;

 protected:
  opencl::ptr<cl_context> context;

 private:
  void checkBuildLog(cl_int err0);
  void loadFromCache(const void* data, size_t length,
                     const CompilerOptions& options);
  void saveToCache(const void* data, size_t length,
                   const CompilerOptions& options) const;
  const DeviceOpenCL& _dev;
  std::string _sourceText;
  CompilerOptions _originalOptions;
  bool _hasSource = false;
};
}  // namespace implementation
}  // namespace ghost

#endif
