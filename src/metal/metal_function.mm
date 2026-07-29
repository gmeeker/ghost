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

#include <ghost/argument_buffer.h>
#include <ghost/device.h>
#include <ghost/function.h>
#include <ghost/metal/impl_device.h>
#include <ghost/metal/impl_function.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ghost {
namespace implementation {
namespace {
class ProgramParams {
private:
  std::vector<uint8_t> _data;

public:
  bool empty() const { return _data.empty(); }
  size_t size() const { return _data.size(); }

  template <typename T> void push_back(const T *v, size_t count) {
    size_t i = _data.size();
    size_t bytes = sizeof(T) * count;
    _data.resize(i + bytes);
    memcpy(&_data[i], v, bytes);
  }
  template <typename T> void push_back(T const &v) { push_back(&v, 1); }
  const void *get() const { return _data.empty() ? nullptr : &_data[0]; }
};

} // namespace

FunctionMetal::FunctionMetal(id<MTLLibrary> library, const std::string &name) {
  NSError *error;
  function = [library
      newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
  if (!function.get()) {
    throw std::runtime_error("Metal: function not found: " + name);
  }
  if (function.get().functionType != MTLFunctionTypeKernel) {
    throw std::runtime_error("Metal: function is not a kernel: " + name);
  }
  pipeline = [library.device newComputePipelineStateWithFunction:function.get()
                                                           error:&error];
}

FunctionMetal::FunctionMetal(id<MTLLibrary> library, const std::string &name,
                             const std::vector<Attribute> &args) {
  NSError *error;
  MTLFunctionConstantValues *constantValues = [MTLFunctionConstantValues new];
#if !__has_feature(objc_arc)
  [constantValues autorelease];
#endif
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
    else if (arg.type() == Attribute::Type_UInt)
      [constantValues setConstantValue:arg.uintArray()
                                  type:MTLDataTypeUInt
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
  if (!function.get()) {
    std::string msg = "Metal: function not found: " + name;
    if (error) {
      msg +=
          " (" + std::string([[error localizedDescription] UTF8String]) + ")";
    }
    throw std::runtime_error(msg);
  }
  if (function.get().functionType != MTLFunctionTypeKernel) {
    throw std::runtime_error("Metal: function is not a kernel: " + name);
  }
  pipeline = [library.device newComputePipelineStateWithFunction:function.get()
                                                           error:&error];
}

static MTLFunctionConstantValues *buildNamedConstants(
    const std::vector<std::pair<std::string, Attribute>> &constants) {
  MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
#if !__has_feature(objc_arc)
  [cv autorelease];
#endif
  for (auto &pair : constants) {
    NSString *nsName = [NSString stringWithUTF8String:pair.first.c_str()];
    auto &attr = pair.second;
    if (attr.type() == Attribute::Type_Bool)
      [cv setConstantValue:attr.boolArray()
                      type:MTLDataTypeBool
                  withName:nsName];
    else if (attr.type() == Attribute::Type_Int)
      [cv setConstantValue:attr.intArray() type:MTLDataTypeInt withName:nsName];
    else if (attr.type() == Attribute::Type_UInt)
      [cv setConstantValue:attr.uintArray()
                      type:MTLDataTypeUInt
                  withName:nsName];
    else if (attr.type() == Attribute::Type_Float)
      [cv setConstantValue:attr.floatArray()
                      type:MTLDataTypeFloat
                  withName:nsName];
  }
  return cv;
}

FunctionMetal::FunctionMetal(
    id<MTLLibrary> library, const std::string &name,
    const std::vector<std::pair<std::string, Attribute>> &namedConstants) {
  NSError *error;
  MTLFunctionConstantValues *constantValues =
      buildNamedConstants(namedConstants);
  function =
      [library newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]
                    constantValues:constantValues
                             error:&error];
  if (!function.get()) {
    std::string msg = "Metal: function not found: " + name;
    if (error) {
      msg +=
          " (" + std::string([[error localizedDescription] UTF8String]) + ")";
    }
    throw std::runtime_error(msg);
  }
  if (function.get().functionType != MTLFunctionTypeKernel) {
    throw std::runtime_error("Metal: function is not a kernel: " + name);
  }
  pipeline = [library.device newComputePipelineStateWithFunction:function.get()
                                                           error:&error];
}

void FunctionMetal::execute(const ghost::Encoder &s,
                            const LaunchArgs &launchArgs,
                            const std::vector<Attribute> &args) {
  auto &enc = metalEncoder(s);
  id<MTLComputeCommandEncoder> computeEncoder = enc.getComputeEncoder();
  [computeEncoder setComputePipelineState:pipeline];

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
    case Attribute::Type_UInt: {
      const uint32_t *v = i->uintArray();
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
      auto metal =
          static_cast<implementation::BufferMetal *>(i->bufferImpl().get());
      [computeEncoder setBuffer:metal->mem.get()
                         offset:metal->baseOffset()
                        atIndex:bufferIndex++];
      break;
    }
    case Attribute::Type_Image: {
      auto metal =
          static_cast<implementation::ImageMetal *>(i->imageImpl().get());
      [computeEncoder setTexture:metal->mem.get() atIndex:textureIndex++];
      // Buffer-backed (aliased) textures: Metal needs to know the GPU is
      // touching the underlying buffer through the texture view. Without
      // useResource:, the buffer's writes (or the texture's caches) can be
      // out of sync with the texture read. See Apple's docs for
      // newTextureWithDescriptor:offset:bytesPerRow:.
      if (metal->aliasedBuffer.get()) {
        MTLResourceUsage usage = MTLResourceUsageRead;
        if (metal->descr.access == Access::WriteOnly ||
            metal->descr.access == Access::ReadWrite) {
          usage |= MTLResourceUsageWrite;
        }
        [computeEncoder useResource:metal->aliasedBuffer.get() usage:usage];
      }
      break;
    }
    case Attribute::Type_ArgumentBuffer: {
      auto ab = i->argumentBuffer();
      if (ab->isStruct()) {
        [computeEncoder setBytes:ab->data()
                          length:ab->size()
                         atIndex:bufferIndex++];
      } else {
        auto metal =
            static_cast<implementation::BufferMetal *>(ab->bufferImpl().get());
        [computeEncoder setBuffer:metal->mem.get()
                           offset:metal->baseOffset()
                          atIndex:bufferIndex++];
      }
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

  if (launchArgs.requiredSubgroupSize() != 0) {
    NSUInteger actual = pipeline.get().threadExecutionWidth;
    if ((NSUInteger)launchArgs.requiredSubgroupSize() != actual) {
      throw std::invalid_argument(
          "Metal: requiredSubgroupSize (" +
          std::to_string(launchArgs.requiredSubgroupSize()) +
          ") does not match pipeline threadExecutionWidth (" +
          std::to_string(actual) + ")");
    }
  }

  MTLSize threadgroupCount = {launchArgs.count(0), launchArgs.count(1),
                              launchArgs.count(2)};
  MTLSize threadgroupSize = {launchArgs.local_size()[0],
                             launchArgs.local_size()[1],
                             launchArgs.local_size()[2]};
  [computeEncoder dispatchThreadgroups:threadgroupCount
                 threadsPerThreadgroup:threadgroupSize];
}

void FunctionMetal::executeIndirect(
    const ghost::Encoder &s, const std::shared_ptr<Buffer> &indirectBuffer,
    size_t indirectOffset, const std::vector<Attribute> &args) {
  auto &enc = metalEncoder(s);
  id<MTLComputeCommandEncoder> computeEncoder = enc.getComputeEncoder();
  [computeEncoder setComputePipelineState:pipeline];

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
    case Attribute::Type_UInt: {
      const uint32_t *v = i->uintArray();
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
      auto metal =
          static_cast<implementation::BufferMetal *>(i->bufferImpl().get());
      [computeEncoder setBuffer:metal->mem.get()
                         offset:metal->baseOffset()
                        atIndex:bufferIndex++];
      break;
    }
    case Attribute::Type_Image: {
      auto metal =
          static_cast<implementation::ImageMetal *>(i->imageImpl().get());
      [computeEncoder setTexture:metal->mem.get() atIndex:textureIndex++];
      // Buffer-backed (aliased) textures: Metal needs to know the GPU is
      // touching the underlying buffer through the texture view. Without
      // useResource:, the buffer's writes (or the texture's caches) can be
      // out of sync with the texture read. See Apple's docs for
      // newTextureWithDescriptor:offset:bytesPerRow:.
      if (metal->aliasedBuffer.get()) {
        MTLResourceUsage usage = MTLResourceUsageRead;
        if (metal->descr.access == Access::WriteOnly ||
            metal->descr.access == Access::ReadWrite) {
          usage |= MTLResourceUsageWrite;
        }
        [computeEncoder useResource:metal->aliasedBuffer.get() usage:usage];
      }
      break;
    }
    case Attribute::Type_ArgumentBuffer: {
      auto ab = i->argumentBuffer();
      if (ab->isStruct()) {
        [computeEncoder setBytes:ab->data()
                          length:ab->size()
                         atIndex:bufferIndex++];
      } else {
        auto metal =
            static_cast<implementation::BufferMetal *>(ab->bufferImpl().get());
        [computeEncoder setBuffer:metal->mem.get()
                           offset:metal->baseOffset()
                          atIndex:bufferIndex++];
      }
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

  auto metalBuf =
      static_cast<implementation::BufferMetal *>(indirectBuffer.get());

  // Determine threads-per-threadgroup from the pipeline.
  // Use requiredThreadsPerThreadgroup if available (macOS 26+),
  // otherwise default to (threadExecutionWidth, 1, 1).
  MTLSize threadgroupSize = {pipeline.get().threadExecutionWidth, 1, 1};
#if defined(MAC_OS_VERSION_26_0)
  if (@available(macOS 26.0, iOS 26.0, tvOS 26.0, macCatalyst 26.0,
                 visionOS 26.0, *)) {
    MTLSize req = pipeline.get().requiredThreadsPerThreadgroup;
    if (req.width > 0 && req.height > 0 && req.depth > 0) {
      threadgroupSize = req;
    }
  }
#endif

  [computeEncoder
      dispatchThreadgroupsWithIndirectBuffer:metalBuf->mem.get()
                        indirectBufferOffset:(NSUInteger)indirectOffset
                       threadsPerThreadgroup:threadgroupSize];
}

Attribute FunctionMetal::getAttribute(FunctionAttributeId what) const {
  switch (what) {
  case kFunctionLocalMemory:
    return (uint32_t)pipeline.get().staticThreadgroupMemoryLength;
  case kFunctionMaxLocalMemory:
    return 0;
  case kFunctionThreadWidth:
    return (uint32_t)pipeline.get().threadExecutionWidth;
  case kFunctionMaxThreads:
    return (uint32_t)pipeline.get().maxTotalThreadsPerThreadgroup;
  case kFunctionRequiredWorkSize: {
#if defined(MAC_OS_VERSION_26_0)
    if (@available(macOS 26.0, iOS 26.0, tvOS 26.0, macCatalyst 26.0,
                   visionOS 26.0, *)) {
      MTLSize s = pipeline.get().requiredThreadsPerThreadgroup;
      return Attribute((uint32_t)s.width, (uint32_t)s.height,
                       (uint32_t)s.depth);
    }
#endif
    return Attribute(0, 0, 0);
  }
  case kFunctionPreferredWorkMultiple:
    return (uint32_t)pipeline.get().threadExecutionWidth;
  case kFunctionNumRegisters:
    return 0;
  case kFunctionPrivateMemory:
    return 0;
  default:
    return Attribute();
  }
}

LibraryMetal::LibraryMetal(const DeviceMetal &dev, bool retainBinary)
    : Library(retainBinary), _dev(dev) {}

void LibraryMetal::loadFromText(const std::string &source,
                                const CompilerOptions &options) {
  NSError *err = nil;
  MTLCompileOptions *compileOptions = [MTLCompileOptions new];
#if !__has_feature(objc_arc)
  [compileOptions autorelease];
#endif
  if (!options.defines.empty()) {
    NSMutableDictionary *macros = [NSMutableDictionary dictionary];
    for (auto &def : options.defines) {
      NSString *key = [NSString stringWithUTF8String:def.first.c_str()];
      NSString *val = [NSString stringWithUTF8String:def.second.c_str()];
      macros[key] = val;
    }
    compileOptions.preprocessorMacros = macros;
  }
  library = [_dev.dev.get()
      newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                   options:compileOptions
                     error:&err];
  if (!library.get()) {
    std::string msg = "Metal: failed to compile source";
    if (err) {
      msg += ": " + std::string([[err localizedDescription] UTF8String]);
    }
    throw std::runtime_error(msg);
  }
}

void LibraryMetal::loadFromData(const void *data, size_t len,
                                const CompilerOptions &options) {
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
    if (retainBinary()) {
      auto bytes = reinterpret_cast<const uint8_t *>(data);
      _binaryData.assign(bytes, bytes + len);
    }
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

ghost::Function LibraryMetal::specializeFunctionNamed(
    const std::string &name,
    const std::vector<std::pair<std::string, Attribute>> &constants) const {
  auto f = std::make_shared<FunctionMetal>(library.get(), name, constants);
  return ghost::Function(f);
}

std::vector<uint8_t> LibraryMetal::getBinary() const { return _binaryData; }
} // namespace implementation
} // namespace ghost
#endif

// vim: ts=2:sw=2:et:ft=mm
// -*- mode: objective-c++; indent-tabs-mode: nil; tab-width: 2 -*-
// code: language=objective-c++ insertSpaces=true tabSize=2
