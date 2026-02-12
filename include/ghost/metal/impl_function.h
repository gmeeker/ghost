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

#ifndef GHOST_METAL_IMPL_FUNCTION_H
#define GHOST_METAL_IMPL_FUNCTION_H

#import <Metal/Metal.h>
#include <ghost/implementation/impl_function.h>
#include <ghost/objc/ptr.h>

namespace ghost {
namespace implementation {
class DeviceMetal;

class FunctionMetal : public Function {
 public:
  objc::ptr<id<MTLFunction>> function;
  objc::ptr<id<MTLComputePipelineState>> pipeline;

  FunctionMetal(id<MTLLibrary> library, const std::string& name);
  FunctionMetal(id<MTLLibrary> library, const std::string& name,
                const std::vector<Attribute>& args);

  virtual void execute(const ghost::Stream& s, const LaunchArgs& launchArgs,
                       const std::vector<Attribute>& args) override;

  virtual Attribute getAttribute(FunctionAttributeId what) const override;
};

class LibraryMetal : public Library {
 public:
  objc::ptr<id<MTLLibrary>> library;

  LibraryMetal(const DeviceMetal& dev);

  void loadFromText(const std::string& text, const std::string& options);
  void loadFromData(const void* data, size_t len, const std::string& options);
  virtual ghost::Function lookupFunction(
      const std::string& name) const override;
  virtual ghost::Function specializeFunction(
      const std::string& name,
      const std::vector<Attribute>& args) const override;

 private:
  const DeviceMetal& _dev;
};
}  // namespace implementation
}  // namespace ghost

#endif

// vim: ts=2:sw=2:et:ft=mm
// -*- mode: objective-c++; indent-tabs-mode: nil; tab-width: 2 -*-
// code: language=objective-c++ insertSpaces=true tabSize=2
