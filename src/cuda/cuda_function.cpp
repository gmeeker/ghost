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

#if WITH_CUDA

#include <ghost/argument_buffer.h>
#include <ghost/cuda/device.h>
#include <ghost/cuda/exception.h>
#include <ghost/cuda/impl_device.h>
#include <ghost/cuda/impl_function.h>
#include <ghost/digest.h>
#include <ghost/function.h>
#include <ghost/io.h>
#include <string.h>

#if WITH_CUDA_NVRTC
#include <nvrtc.h>
#endif

#include <fstream>
#include <list>
#include <sstream>
#include <vector>

namespace ghost {
#if WITH_CUDA_NVRTC
namespace cu {
template <>
class detail<nvrtcProgram> {
 public:
  static CUresult release(nvrtcProgram v) {
    nvrtcResult err = nvrtcDestroyProgram(&v);
    return (err == NVRTC_SUCCESS) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
  }
};
}  // namespace cu

inline void checkNVRTCError(nvrtcResult err) {
  if (err != NVRTC_SUCCESS) throw std::runtime_error(nvrtcGetErrorString(err));
}
#endif

namespace implementation {
using namespace cu;

FunctionCUDA::FunctionCUDA(const DeviceCUDA& dev, CUfunction k)
    : kernel(k), _dev(dev) {}

void FunctionCUDA::collectParams(const std::vector<Attribute>& args,
                                 std::vector<void*>& params, size_t& localMem) {
  for (auto i = args.begin(); i != args.end(); ++i) {
    switch (i->type()) {
      case Attribute::Type_Float: {
        const float* v = i->floatArray();
        params.push_back(const_cast<float*>(v));
        break;
      }
      case Attribute::Type_Int: {
        const int32_t* v = i->intArray();
        params.push_back(const_cast<int32_t*>(v));
        break;
      }
      case Attribute::Type_UInt: {
        const uint32_t* v = i->uintArray();
        params.push_back(const_cast<uint32_t*>(v));
        break;
      }
      case Attribute::Type_Bool: {
        const bool* v = i->boolArray();
        params.push_back(const_cast<bool*>(v));
        break;
      }
      case Attribute::Type_Buffer: {
        auto cuda =
            static_cast<implementation::BufferCUDA*>(i->bufferImpl().get());
        params.push_back(cuda->mem.handleAddress());
        break;
      }
      case Attribute::Type_Image: {
        auto cuda =
            static_cast<implementation::ImageCUDA*>(i->imageImpl().get());
        CUaddress_mode addressMode = CU_TR_ADDRESS_MODE_CLAMP;
        CUfilter_mode filterMode = CU_TR_FILTER_MODE_POINT;
        bool normalizedCoords = false;
        if (auto& s = i->sampler()) {
          switch (s->filter) {
            case FilterMode::Nearest:
              filterMode = CU_TR_FILTER_MODE_POINT;
              break;
            case FilterMode::Linear:
              filterMode = CU_TR_FILTER_MODE_LINEAR;
              break;
          }
          switch (s->address) {
            case AddressMode::Clamp:
              addressMode = CU_TR_ADDRESS_MODE_CLAMP;
              break;
            case AddressMode::Wrap:
              addressMode = CU_TR_ADDRESS_MODE_WRAP;
              break;
            case AddressMode::Mirror:
              addressMode = CU_TR_ADDRESS_MODE_MIRROR;
              break;
          }
          normalizedCoords = s->normalizedCoords;
        }
        // Cached on the image and reused across launches; the returned
        // address is the handle's stable heap storage, valid until the
        // texture is destroyed (deferred behind in-flight work when the
        // image dies — see ImageCUDA::~ImageCUDA).
        params.push_back(
            cuda->lookupTexture(addressMode, filterMode, normalizedCoords));
        break;
      }
      case Attribute::Type_ArgumentBuffer: {
        auto ab = i->argumentBuffer();
        if (ab->isStruct()) {
          params.push_back(const_cast<void*>(ab->data()));
        } else {
          auto cuda =
              static_cast<implementation::BufferCUDA*>(ab->bufferImpl().get());
          params.push_back(cuda->mem.handleAddress());
        }
        break;
      }
      case Attribute::Type_LocalMem:
        localMem += (size_t)i->asUInt();
        break;
      default:
        break;
    }
  }
}

void FunctionCUDA::buildKernelNodeParams(const LaunchArgs& launchArgs,
                                         const std::vector<Attribute>& args,
                                         std::vector<void*>& paramStorage,
                                         CUDA_KERNEL_NODE_PARAMS& out) {
  size_t local_mem = 0;
  paramStorage.clear();
  collectParams(args, paramStorage, local_mem);

  out = {};
  out.func = kernel;
  out.gridDimX = (unsigned int)launchArgs.count(0);
  out.gridDimY = (unsigned int)launchArgs.count(1);
  out.gridDimZ = (unsigned int)launchArgs.count(2);
  out.blockDimX = (unsigned int)launchArgs.local_size()[0];
  out.blockDimY = (unsigned int)launchArgs.local_size()[1];
  out.blockDimZ = (unsigned int)launchArgs.local_size()[2];
  out.sharedMemBytes = (unsigned int)local_mem;
  out.kernelParams = paramStorage.empty() ? nullptr : paramStorage.data();
  out.extra = nullptr;
  // v2 fields (kern / ctx) left zero: this is a driver-CUfunction kernel node.
}

void FunctionCUDA::execute(const ghost::Encoder& s,
                           const LaunchArgs& launchArgs,
                           const std::vector<Attribute>& args) {
  CUresult err;
  size_t local_mem = 0;
  std::vector<void*> params;
  // Opportunistic destroy of texture objects parked by dead images whose
  // guarding work has since completed.
  _dev.reapDeferredTextures();

  collectParams(args, params, local_mem);
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  // Retain buffer/image impls past kernel completion: if the caller drops
  // their wrapper between dispatch and stream sync, the device pointer must
  // not be freed before the GPU finishes reading it.
  for (auto i = args.begin(); i != args.end(); ++i) {
    switch (i->type()) {
      case Attribute::Type_Buffer: {
        auto buf =
            static_cast<implementation::BufferCUDA*>(i->bufferImpl().get());
        buf->markUsed(*stream_impl);
        break;
      }
      case Attribute::Type_Image: {
        auto img =
            static_cast<implementation::ImageCUDA*>(i->imageImpl().get());
        img->markUsed(*stream_impl);
        break;
      }
      case Attribute::Type_ArgumentBuffer: {
        auto ab = i->argumentBuffer();
        if (!ab->isStruct()) {
          auto buf =
              static_cast<implementation::BufferCUDA*>(ab->bufferImpl().get());
          buf->markUsed(*stream_impl);
        }
        break;
      }
      default:
        break;
    }
  }
  if (launchArgs.requiredSubgroupSize() != 0) {
    int warpSize = 0;
    checkError(cuDeviceGetAttribute(&warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                                    _dev.device));
    if ((int)launchArgs.requiredSubgroupSize() != warpSize) {
      throw std::invalid_argument(
          "CUDA: requiredSubgroupSize must equal warp size (" +
          std::to_string(warpSize) + ")");
    }
  }
  static const char* kDimNames[3] = {"global_size[0]", "global_size[1]",
                                     "global_size[2]"};
  static const char* kLocalNames[3] = {"local_size[0]", "local_size[1]",
                                       "local_size[2]"};
  unsigned int grid_size[3];
  unsigned int local_size[3];
  for (size_t i = 0; i < 3; i++) {
    grid_size[i] = narrowDim(launchArgs.count(i), kDimNames[i]);
    local_size[i] = narrowDim(launchArgs.local_size()[i], kLocalNames[i]);
  }
  // CUDA caps a block's shared memory at 48 KB unless the function explicitly
  // opts into the larger per-block budget (sm_80/90 ~164 KB, sm_86 ~100 KB,
  // sm_100 ~227 KB). Without the opt-in, a launch requesting more than 48 KB of
  // dynamic shared memory fails with CUDA_ERROR_INVALID_VALUE. Do it lazily
  // here so callers only need to pass the byte count via
  // Attribute::localMem(N): when a launch asks for more than the static cap,
  // grant exactly that much. The driver caches the value per CUfunction, so we
  // track a high-water mark to skip redundant cuFuncSetAttribute calls on
  // subsequent launches. Small-shape launches (<= 48 KB) never touch this path.
  // Stealing shared from the unified L1/shared lowers occupancy, so requesting
  // large amounts is the caller's (compute-bound, large-shape) trade-off to
  // make.
  static const size_t kStaticSharedCap = 48 * 1024;
  if (local_mem > kStaticSharedCap &&
      local_mem > _maxDynamicSharedBytes.load(std::memory_order_relaxed)) {
    checkError(cuFuncSetAttribute(
        kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        (int)local_mem));
    _maxDynamicSharedBytes.store(local_mem, std::memory_order_relaxed);
  }
  if (launchArgs.is_cooperative()) {
    err = cuLaunchCooperativeKernel(kernel, grid_size[0], grid_size[1],
                                    grid_size[2], local_size[0], local_size[1],
                                    local_size[2], local_mem,
                                    stream_impl->queue, &params[0]);
  } else {
    err = cuLaunchKernel(kernel, grid_size[0], grid_size[1], grid_size[2],
                         local_size[0], local_size[1], local_size[2], local_mem,
                         stream_impl->queue, &params[0], nullptr);
  }
  checkError(err);
}

Attribute FunctionCUDA::getAttribute(FunctionAttributeId what) const {
  switch (what) {
    case kFunctionLocalMemory: {
      int bytes;
      checkError(cuFuncGetAttribute(&bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                    kernel));
      return bytes;
    }
    case kFunctionMaxLocalMemory: {
      int bytes;
      checkError(cuFuncGetAttribute(
          &bytes, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernel));
      return bytes;
    }
    case kFunctionThreadWidth: {
      int v;
      checkError(
          cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_WARP_SIZE, _dev.device));
      return v;
    }
    case kFunctionMaxThreads: {
      int bytes;
      checkError(cuFuncGetAttribute(
          &bytes, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel));
      return bytes;
    }
    case kFunctionRequiredWorkSize:
      return Attribute(0, 0, 0);
    case kFunctionPreferredWorkMultiple: {
      int v;
      checkError(
          cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_WARP_SIZE, _dev.device));
      return v;
    }
    case kFunctionNumRegisters: {
      int v;
      checkError(cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_NUM_REGS, kernel));
      return v;
    }
    case kFunctionPrivateMemory: {
      int v;
      checkError(
          cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel));
      return v;
    }
    case kFunctionPreferredSharedMemoryCarveout: {
      int v;
      checkError(cuFuncGetAttribute(
          &v, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, kernel));
      return v;
    }
    default:
      return Attribute();
  }
}

void FunctionCUDA::setAttribute(FunctionAttributeId what,
                                const Attribute& value) {
  switch (what) {
    case kFunctionPreferredSharedMemoryCarveout:
      // 0..100 (% of unified L1/shared given to shared), or -1 for the driver
      // default. A hint: the driver may pick a nearby supported split. Unlike
      // the >48 KB opt-in below this does not change what a launch is allowed
      // to request, only the occupancy/L1 trade-off, so there is nothing to
      // track.
      checkError(cuFuncSetAttribute(
          kernel, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
          value.asInt()));
      break;
    case kFunctionMaxLocalMemory: {
      // Explicit form of the lazy opt-in in execute(): grant a dynamic
      // shared-memory budget above the 48 KB static cap. Bump the high-water
      // mark so a later same-or-smaller launch skips the redundant driver call.
      int bytes = value.asInt();
      checkError(cuFuncSetAttribute(
          kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes));
      if (bytes > 0) {
        size_t granted = (size_t)bytes;
        size_t prev = _maxDynamicSharedBytes.load(std::memory_order_relaxed);
        while (granted > prev &&
               !_maxDynamicSharedBytes.compare_exchange_weak(
                   prev, granted, std::memory_order_relaxed)) {
        }
      }
      break;
    }
    default:
      throw ghost::unsupported_error();
  }
}

uint32_t FunctionCUDA::preferredSubgroupSize() const {
  int v = 0;
  checkError(
      cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_WARP_SIZE, _dev.device));
  return (uint32_t)v;
}

LibraryCUDA::LibraryCUDA(const DeviceCUDA& dev, bool retainBinary)
    : Library(retainBinary), program(0), _dev(dev) {}

void LibraryCUDA::loadFromText(const std::string& text,
                               const CompilerOptions& options) {
#if WITH_CUDA_NVRTC
  try {
    loadFromCache(text.c_str(), text.size(), options);
  } catch (...) {
  }
  if (program.get() != nullptr) return;

  // Build header arrays from options.headers for NVRTC.
  std::vector<const char*> headerSources, headerNames;
  for (auto& h : options.headers) {
    headerNames.push_back(h.first.c_str());
    headerSources.push_back(h.second.c_str());
  }
  ptr<nvrtcProgram> prog;
  checkNVRTCError(nvrtcCreateProgram(
      &prog, text.c_str(), "ghost.cu", (int)headerNames.size(),
      headerSources.empty() ? NULL : &headerSources[0],
      headerNames.empty() ? NULL : &headerNames[0]));

  // Build compiler options: hardcoded paths + gpu arch + user arguments +
  // defines.
  std::vector<std::string> optStrings;
#ifdef WITH_CUDA_NVRTC_INCLUDE_PATH
  optStrings.push_back("-I" WITH_CUDA_NVRTC_INCLUDE_PATH);
#endif
#ifdef WITH_CUDA_NVRTC_STD_INCLUDE_PATH
  optStrings.push_back("-I" WITH_CUDA_NVRTC_STD_INCLUDE_PATH);
#endif
  std::stringstream gpu_arch;
  // If we want PTX:
  // gpu_arch << "--gpu-architecture=compute_" << _dev.computeCapability.major
  //          << "0";
  gpu_arch << "--gpu-architecture=sm_" << _dev.computeCapability.major
           << _dev.computeCapability.minor;
  optStrings.push_back(gpu_arch.str());
  for (auto& arg : options.arguments) {
    optStrings.push_back(arg);
  }
  for (auto& def : options.defines) {
    std::string d = "-D" + def.first;
    if (!def.second.empty()) d += "=" + def.second;
    optStrings.push_back(d);
  }
  std::vector<const char*> opts;
  for (auto& s : optStrings) {
    opts.push_back(s.c_str());
  }
  nvrtcResult compileResult =
      nvrtcCompileProgram(prog, (int)opts.size(), opts.data());
  size_t logSize;
  checkNVRTCError(nvrtcGetProgramLogSize(prog, &logSize));
  std::vector<char> log(logSize);
  checkNVRTCError(nvrtcGetProgramLog(prog, &log[0]));
  if (compileResult != NVRTC_SUCCESS) {
    throw std::runtime_error(std::string("CUDA compile error: ") + &log[0]);
  }

  // If we want PTX
  // size_t ptxSize;
  // checkNVRTCError(nvrtcGetPTXSize(prog, &ptxSize));
  // std::vector<char> ptx(ptxSize);
  // checkNVRTCError(nvrtcGetPTX(prog, &ptx[0]));
  // loadFromData(&ptx[0], 0, options);
  size_t cubinSize;
  checkNVRTCError(nvrtcGetCUBINSize(prog, &cubinSize));
  std::vector<char> cuOut(cubinSize);
  checkNVRTCError(nvrtcGetCUBIN(prog, &cuOut[0]));

  // Load resulting cuBin into module
  loadFromBinary(&cuOut[0]);

  if (retainBinary()) {
    _binaryData.assign(cuOut.begin(), cuOut.end());
  }

  try {
    saveToCache(&cuOut[0], cuOut.size(), text.c_str(), text.size(), options);
  } catch (...) {
  }
#else
  (void)text;
  (void)options;
  throw unsupported_error();
#endif
}

void LibraryCUDA::loadFromData(const void* data, size_t len,
                               const CompilerOptions& options) {
  CUjitInputType inputType = CU_JIT_INPUT_FATBINARY;
  if (len == 0) {
    len = strlen(reinterpret_cast<const char*>(data));
    inputType = CU_JIT_INPUT_PTX;
  }
  try {
    loadFromCache(data, len, options);
  } catch (...) {
  }
  if (program.get() != nullptr) return;

  CUjit_option jitOptions[6];
  void* optionVals[6];
  float walltime;
  std::vector<char> error_log, info_log;
  error_log.resize(8192);
  info_log.resize(8192);
  void* cuOut;
  size_t outSize;
  int myErr = 0;

  // Setup linker options
  // Return walltime from JIT compilation
  jitOptions[0] = CU_JIT_WALL_TIME;
  optionVals[0] = (void*)&walltime;
  // Pass a buffer for info messages
  jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
  optionVals[1] = (void*)&info_log[0];
  // Pass the size of the info buffer
  jitOptions[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
  optionVals[2] = (void*)info_log.size();
  // Pass a buffer for error message
  jitOptions[3] = CU_JIT_ERROR_LOG_BUFFER;
  optionVals[3] = (void*)&error_log[0];
  // Pass the size of the error buffer
  jitOptions[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
  optionVals[4] = (void*)error_log.size();
  // Make the linker verbose
  jitOptions[5] = CU_JIT_LOG_VERBOSE;
  optionVals[5] = (void*)1;

  // Create a pending linker invocation
  cu::ptr<CUlinkState> lState;
  checkError(cuLinkCreate(6, jitOptions, optionVals, &lState));

  myErr = cuLinkAddData(lState, inputType, const_cast<void*>(data), len, 0, 0,
                        0, 0);

  if (myErr != CUDA_SUCCESS) {
    throw std::runtime_error("CUDA linker error: " +
                             std::string(&error_log[0]));
  }

  checkError(cuLinkComplete(lState, &cuOut, &outSize));

  // Linker walltime and info_log were requested in options above.
  // printf("CUDA Link Completed in %fms. Linker Output:\n%s\n", walltime,
  // info_log);

  // Load resulting cuBin into module
  loadFromBinary(cuOut);

  try {
    saveToCache(cuOut, outSize, data, len, options);
  } catch (...) {
  }
}

void LibraryCUDA::loadFromBinary(void* binary) {
  CUresult err;
  err = cuModuleLoadData(&program, binary);
  checkError(err);
}

void LibraryCUDA::loadFromCache(const void* data, size_t length,
                                const CompilerOptions& options) {
  std::vector<std::vector<unsigned char>> binaries;
  std::vector<size_t> sizes;
  if (_dev.binaryCache().loadBinaries(binaries, sizes, _dev, data, length,
                                      options)) {
    loadFromBinary(&binaries[0][0]);
  }
}

void LibraryCUDA::saveToCache(void* binary, size_t binarySize, const void* data,
                              size_t length,
                              const CompilerOptions& options) const {
  if (!_dev.binaryCache().isEnabled()) return;
  std::vector<size_t> sizes = {binarySize};
  std::vector<unsigned char*> binaries = {
      reinterpret_cast<unsigned char*>(binary)};

  _dev.binaryCache().saveBinaries(_dev, binaries, sizes, data, length, options);
}

ghost::Function LibraryCUDA::lookupFunction(const std::string& name) const {
  CUresult err;
  CUfunction kernel;
  err = cuModuleGetFunction(&kernel, program.get(), name.c_str());
  checkError(err);
  auto f = std::make_shared<FunctionCUDA>(_dev, kernel);
  return ghost::Function(f);
}

void LibraryCUDA::setGlobals(
    const std::vector<std::pair<std::string, Attribute>>& globals) {
  for (auto& g : globals) {
    CUdeviceptr dptr;
    size_t dsize;
    checkError(
        cuModuleGetGlobal(&dptr, &dsize, program.get(), g.first.c_str()));
    auto& attr = g.second;
    switch (attr.type()) {
      case Attribute::Type_Float: {
        size_t sz = sizeof(float) * attr.count();
        if (sz > dsize)
          throw std::runtime_error("CUDA global '" + g.first +
                                   "': size mismatch");
        checkError(cuMemcpyHtoD(dptr, attr.floatArray(), sz));
        break;
      }
      case Attribute::Type_Int: {
        size_t sz = sizeof(int32_t) * attr.count();
        if (sz > dsize)
          throw std::runtime_error("CUDA global '" + g.first +
                                   "': size mismatch");
        checkError(cuMemcpyHtoD(dptr, attr.intArray(), sz));
        break;
      }
      case Attribute::Type_UInt: {
        size_t sz = sizeof(uint32_t) * attr.count();
        if (sz > dsize)
          throw std::runtime_error("CUDA global '" + g.first +
                                   "': size mismatch");
        checkError(cuMemcpyHtoD(dptr, attr.uintArray(), sz));
        break;
      }
      case Attribute::Type_Bool: {
        // CUDA __constant__ bool may be 1 byte; copy as-is
        size_t sz = sizeof(bool) * attr.count();
        if (sz > dsize)
          throw std::runtime_error("CUDA global '" + g.first +
                                   "': size mismatch");
        checkError(cuMemcpyHtoD(dptr, attr.boolArray(), sz));
        break;
      }
      default:
        throw std::runtime_error("CUDA setGlobals: unsupported attribute type");
    }
  }
}

std::vector<uint8_t> LibraryCUDA::getBinary() const { return _binaryData; }
}  // namespace implementation
}  // namespace ghost
#endif
