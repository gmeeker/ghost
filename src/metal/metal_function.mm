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

#if WITH_METAL

#include <ghost/function.h>
#include <ghost/metal/impl_device.h>
#include <ghost/metal/impl_function.h>

#include <algorithm>

namespace ghost {
namespace implementation {
namespace {
class ProgramParams {
private:
  std::vector<uint8_t> _data;
  static constexpr size_t Align = 16;

public:
  bool empty() const { return _data.empty(); }
  size_t size() const { return _data.size(); }

  template <typename T> void push_back(const T *v, size_t count) {
    size_t i = _data.size();
    _data.resize(i + std::max(Align, sizeof(T) * count));
    memset(&_data[i], 0, Align);
    memcpy(&_data[i], v, sizeof(T) * count);
  }
  template <typename T> void push_back(T const &v) { push_back(&v, 1); }
  const void *get() const { return _data.empty() ? nullptr : &_data[0]; }
};

} // namespace

FunctionMetal::FunctionMetal(id<MTLLibrary> library, const std::string &name) {
  NSError *error;
  function = [library
      newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
  pipeline = [library.device newComputePipelineStateWithFunction:function.get()
                                                           error:&error];
}

FunctionMetal::FunctionMetal(id<MTLLibrary> library, const std::string &name,
                             const std::vector<Attribute> &args) {
  NSError *error;
  MTLFunctionConstantValues *constantValues = [MTLFunctionConstantValues new];
  size_t j = 0;
  for (auto arg : args) {
    if (arg.type() == Attribute::Type_Bool)
      [constantValues setConstantValue:arg.boolArray()
                                  type:MTLDataTypeBool
                               atIndex:j++];
    else if (arg.type() == Attribute::Type_Int)
      [constantValues setConstantValue:arg.intArray()
                                  type:MTLDataTypeInt
                               atIndex:j++];
    else if (arg.type() == Attribute::Type_Float)
      [constantValues setConstantValue:arg.floatArray()
                                  type:MTLDataTypeFloat
                               atIndex:j++];
  }
  function =
      [library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]
                    constantValues:constantValues
                             error:&error];
  pipeline = [library.device newComputePipelineStateWithFunction:function.get()

                                                           error:&error];
}

void FunctionMetal::execute(const ghost::Stream &s,
                            const LaunchArgs &launchArgs,
                            const std::vector<Attribute> &args) {
  auto stream_impl = static_cast<implementation::StreamMetal *>(s.impl().get());
  id<MTLCommandBuffer> commandBuffer = nil;
  commandBuffer = [stream_impl->queue.get() commandBuffer];
  commandBuffer.label = @"Ghost";
  id<MTLComputeCommandEncoder> computeEncoder = nil;
  if (@available(macOS 10.14, iOS 12.0, tvOS 12.0, macCatalyst 13.0, *)) {
    computeEncoder = [commandBuffer
        computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
  }
  if (!computeEncoder) {
    computeEncoder = [commandBuffer computeCommandEncoder];
  }

  ProgramParams params;
  size_t bufferIndex = 0;
  size_t textureIndex = 0;
  size_t sharedIndex = 0;
  for (auto i = args.begin(); i != args.end(); ++i) {
    switch (i->type()) {
    case Attribute::Type_Float: {
      const float *v = i->floatArray();
      size_t count = i->count();
      params.push_back(v, count);
      break;
    }
    case Attribute::Type_Int: {
      const int32_t *v = i->intArray();
      size_t count = i->count();
      params.push_back(v, count);
      break;
    }
    case Attribute::Type_Bool: {
      const bool *v = i->boolArray();
      size_t count = i->count();
      params.push_back(v, count);
      break;
    }
    case Attribute::Type_Buffer: {
      auto metal = static_cast<implementation::BufferMetal *>(
          i->asBuffer()->impl().get());
      [computeEncoder setBuffer:metal->mem.get()
                         offset:0
                        atIndex:bufferIndex++];
      break;
    }
    case Attribute::Type_Image: {
      auto metal =
          static_cast<implementation::ImageMetal *>(i->asImage()->impl().get());
      [computeEncoder setTexture:metal->mem.get() atIndex:textureIndex++];
      break;
    }
    case Attribute::Type_LocalMem:
      [computeEncoder setThreadgroupMemoryLength:(size_t)i->asUInt()
                                         atIndex:sharedIndex++];
      break;
    default:
      break;
    }
  }
  if (!params.empty()) {
    [computeEncoder setBytes:params.get()
                      length:params.size()
                     atIndex:bufferIndex++];
  }

  MTLSize threadgroupCount = {launchArgs.global_size()[0],
                              launchArgs.global_size()[1],
                              launchArgs.global_size()[2]};
  MTLSize threadgroupSize = {launchArgs.local_size()[0],
                             launchArgs.local_size()[1],
                             launchArgs.local_size()[2]};
  [computeEncoder dispatchThreadgroups:threadgroupCount
                 threadsPerThreadgroup:threadgroupSize];

  [computeEncoder endEncoding];
}

LibraryMetal::LibraryMetal(const DeviceMetal &dev) : _dev(dev) {}

void LibraryMetal::loadFromText(const std::string &source,
                                const std::string &) {
  NSError *err;
  MTLCompileOptions *compileOptions = [MTLCompileOptions new];
#if !__has_feature(objc_arc)
  [compileOptions autorelease];
#endif
  library = [_dev.dev.get()
      newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                   options:compileOptions
                     error:&err];
}

void LibraryMetal::loadFromData(const void *data, size_t len,
                                const std::string &) {
  if (data == nullptr) {
    library = [_dev.dev.get() newDefaultLibrary];
  } else {
    NSError *err = nil;
    dispatch_data_t d;
    d = dispatch_data_create(data, len, nullptr,
                             DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    library = [_dev.dev.get() newLibraryWithData:d error:&err];
#if !__has_feature(objc_arc)
    dispatch_release(d);
#endif
  }
}

ghost::Function LibraryMetal::lookupFunction(const std::string &name) const {
  auto f = std::make_shared<FunctionMetal>(library.get(), name);
  return ghost::Function(f);
}

ghost::Function
LibraryMetal::specializeFunction(const std::string &name,
                                 const std::vector<Attribute> &args) const {
  auto f = std::make_shared<FunctionMetal>(library.get(), name, args);
  return ghost::Function(f);
}
} // namespace implementation
} // namespace ghost
#endif

// vim: ts=2:sw=2:et:ft=mm
// -*- mode: objective-c++; indent-tabs-mode: nil; tab-width: 2 -*-
// code: language=objective-c++ insertSpaces=true tabSize=2
