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

#if WITH_OPENCL

#include <ghost/digest.h>
#include <ghost/function.h>
#include <ghost/io.h>
#include <ghost/opencl/device.h>
#include <ghost/opencl/exception.h>
#include <ghost/opencl/impl_device.h>
#include <ghost/opencl/impl_function.h>

#include <fstream>
#include <vector>

namespace ghost {
namespace implementation {
using namespace opencl;

FunctionOpenCL::FunctionOpenCL(const DeviceOpenCL& dev,
                               opencl::ptr<cl_kernel> k)
    : kernel(k), _dev(dev) {}

void FunctionOpenCL::execute(const ghost::Stream& s,
                             const LaunchArgs& launchArgs,
                             const std::vector<Attribute>& args) {
  cl_int err;
  cl_uint idx = 0;
  for (auto i = args.begin(); i != args.end(); ++i) {
    switch (i->type()) {
      case Attribute::Type_Float: {
        const float* v = i->floatArray();
        size_t count = i->count();
        // cl_float3 is the same as cl_float4
        if (count == 3) count = 4;
        err = clSetKernelArg(kernel, idx++, sizeof(v[0]) * count, v);
        checkError(err);
        break;
      }
      case Attribute::Type_Int: {
        const int32_t* v = i->intArray();
        size_t count = i->count();
        // cl_int3 is the same as cl_int4
        if (count == 3) count = 4;
        err = clSetKernelArg(kernel, idx++, sizeof(v[0]) * count, v);
        checkError(err);
        break;
      }
      case Attribute::Type_Bool: {
        const bool* v = i->boolArray();
        size_t count = i->count();
        // cl_bool3 is the same as cl_bool4
        if (count == 3) count = 4;
        err = clSetKernelArg(kernel, idx++, sizeof(v[0]) * count, v);
        checkError(err);
        break;
      }
      case Attribute::Type_Buffer: {
        auto opencl = static_cast<implementation::BufferOpenCL*>(
            i->asBuffer()->impl().get());
        cl_mem v = opencl->mem.get();
        err = clSetKernelArg(kernel, idx++, sizeof(v), &v);
        checkError(err);
        break;
      }
      case Attribute::Type_Image: {
        auto opencl = static_cast<implementation::ImageOpenCL*>(
            i->asImage()->impl().get());
        cl_mem v = opencl->mem.get();
        err = clSetKernelArg(kernel, idx++, sizeof(v), &v);
        checkError(err);
        break;
      }
      case Attribute::Type_LocalMem:
        err = clSetKernelArg(kernel, idx++, (size_t)i->asUInt(), nullptr);
        checkError(err);
        break;
      default:
        break;
    }
  }
  auto stream_impl = static_cast<implementation::StreamOpenCL*>(s.impl().get());
  opencl::ptr<cl_event> outEvent;
  std::vector<cl_event> waitEvents;
  size_t global_size[3];
  size_t local_size[3];
  for (size_t i = 0; i < 3; i++) {
    global_size[i] = launchArgs.global_size()[i];
    local_size[i] = launchArgs.local_size()[i];
  }
  err = clEnqueueNDRangeKernel(
      stream_impl->queue, kernel, launchArgs.dims(), NULL, global_size,
      launchArgs.is_local_defined() ? local_size : nullptr, waitEvents.size(),
      waitEvents.empty() ? nullptr : &waitEvents[0], &outEvent);
  checkError(err);
}

Attribute FunctionOpenCL::getAttribute(FunctionAttributeId what) const {
  std::vector<cl_device_id> devices;
  cl_int err;
  size_t numDevs;
  err =
      clGetContextInfo(_dev.context, CL_CONTEXT_DEVICES, 0, nullptr, &numDevs);
  checkError(err);
  numDevs /= sizeof(cl_device_id);
  devices.resize(size_t(numDevs));
  err = clGetContextInfo(_dev.context, CL_CONTEXT_DEVICES,
                         numDevs * sizeof(cl_device_id), &devices[0], nullptr);
  checkError(err);
  switch (what) {
    case kFunctionLocalMemory: {
      cl_ulong bytes;
      checkError(clGetKernelWorkGroupInfo(kernel, devices[0],
                                          CL_KERNEL_LOCAL_MEM_SIZE,
                                          sizeof(bytes), &bytes, nullptr));
      return (uint64_t)bytes;
    }
    case kFunctionMaxLocalMemory:
      return 0;
    case kFunctionThreadWidth: {
      size_t preferredWorkGroupSize;
      checkError(clGetKernelWorkGroupInfo(
          kernel, devices[0], CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
          sizeof(preferredWorkGroupSize), &preferredWorkGroupSize, nullptr));
      return (uint64_t)preferredWorkGroupSize;
    }
    case kFunctionMaxThreads: {
      size_t maxSize;
      checkError(clGetKernelWorkGroupInfo(kernel, devices[0],
                                          CL_KERNEL_WORK_GROUP_SIZE,
                                          sizeof(maxSize), &maxSize, nullptr));
      return (uint64_t)maxSize;
    }
    case kFunctionRequiredWorkSize: {
      size_t workSize[3];
      checkError(clGetKernelWorkGroupInfo(
          kernel, devices[0], CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
          sizeof(workSize[0]) * 3, workSize, nullptr));
      return Attribute((uint64_t)workSize[0], (uint64_t)workSize[1],
                       (uint64_t)workSize[2]);
    }
    default:
      return Attribute();
  }
}

LibraryOpenCL::LibraryOpenCL(const DeviceOpenCL& dev) : program(0), _dev(dev) {}

void LibraryOpenCL::checkBuildLog(cl_int err0) {
  if (err0 != CL_SUCCESS) {
    cl_int err;
    auto devices = _dev.getDevices();
    for (size_t i = 0; i < devices.size(); i++) {
      cl_build_status build_status;
      err = clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_STATUS,
                                  sizeof(build_status), &build_status, nullptr);
      checkError(err);
      if (build_status == CL_BUILD_ERROR) {
        std::vector<char> str;
        size_t logSize;
        err = clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG,
                                    0, nullptr, &logSize);
        checkError(err);
        str.resize(size_t(logSize));
        err = clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG,
                                    logSize, &str[0], nullptr);
        checkError(err);
        throw std::runtime_error(std::string("OpenCL compile error: ") +
                                 &str[0]);
      }
    }
    checkError(err0);
  }
}

void LibraryOpenCL::loadFromText(const std::string& text,
                                 const std::string& options) {
  opencl::ptr<cl_context> context = _dev.context;
  cl_int err;
  try {
    loadFromCache(text.c_str(), text.size(), options);
  } catch (...) {
  }
  if (program.get() != nullptr) return;
  const char* progtext = text.c_str();
  program = opencl::ptr<cl_program>(
      clCreateProgramWithSource(context, 1, &progtext, nullptr, &err));
  checkError(err);
  err = clBuildProgram(program, 0, nullptr, options.c_str(), nullptr, nullptr);
  checkBuildLog(err);
  try {
    saveToCache(text.c_str(), text.size(), options);
  } catch (...) {
  }
}

void LibraryOpenCL::loadFromData(const void* data, size_t len,
                                 const std::string& options) {
#if !defined(CL_VERSION_3_0)
  (void)data;
  (void)len;
  (void)options;
  checkError(CL_COMPILER_NOT_AVAILABLE);
#else
  opencl::ptr<cl_context> context = _dev.context;
  if (!_dev.checkExtension("cl_khr_spir"))
    checkError(CL_COMPILER_NOT_AVAILABLE);
  cl_int err;
  try {
    loadFromCache(data, len, options);
  } catch (...) {
  }
  if (program.get() != nullptr) return;
  program =
      opencl::ptr<cl_program>(clCreateProgramWithIL(context, data, len, &err));
  checkError(err);
  err = clBuildProgram(program, 0, nullptr, options.c_str(), nullptr, nullptr);
  checkBuildLog(err);
  try {
    saveToCache(data, len, options);
  } catch (...) {
  }
#endif
}

void LibraryOpenCL::loadFromBinaries(const size_t* lengths,
                                     const unsigned char** binaries,
                                     const std::string& options) {
  opencl::ptr<cl_context> context = _dev.context;
  cl_int err;
  auto devices = _dev.getDevices();
  program = opencl::ptr<cl_program>(
      clCreateProgramWithBinary(context, (cl_uint)devices.size(), &devices[0],
                                lengths, binaries, nullptr, &err));
  checkError(err);
  err = clBuildProgram(program, 0, nullptr, options.c_str(), nullptr, nullptr);
  checkBuildLog(err);
}

void LibraryOpenCL::loadFromCache(const void* data, size_t length,
                                  const std::string& options) {
  std::vector<std::vector<unsigned char>> binaries;
  std::vector<size_t> sizes;
  if (_dev.binaryCache().loadBinaries(binaries, sizes, _dev, data, length,
                                      options)) {
    std::vector<const unsigned char*> ptrs;
    for (size_t i = 0; i < binaries.size(); i++)
      ptrs.push_back(&binaries[i][0]);
    loadFromBinaries(&sizes[0], &ptrs[0], options);
  }
}

void LibraryOpenCL::saveToCache(const void* data, size_t length,
                                const std::string& options) const {
  if (!_dev.binaryCache().isEnabled()) return;
  auto devices = _dev.getDevices();
  size_t i, numDevs;
  cl_int err;
  numDevs = devices.size();
  std::vector<size_t> sizes;
  std::vector<unsigned char> buffer;
  std::vector<unsigned char*> binaries;
  sizes.resize(numDevs);
  binaries.resize(numDevs);
  err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                         numDevs * sizeof(size_t), &sizes[0], NULL);
  checkError(err);
  size_t total = 0;
  for (i = 0; i < numDevs; i++) {
    total += sizes[i];
  }
  buffer.resize(total);
  total = 0;
  for (i = 0; i < numDevs; i++) {
    binaries[i] = &buffer[total];
    total += sizes[i];
  }
  err = clGetProgramInfo(program, CL_PROGRAM_BINARIES,
                         numDevs * sizeof(unsigned char*), &binaries[0], NULL);
  checkError(err);

  _dev.binaryCache().saveBinaries(_dev, binaries, sizes, data, length, options);
}

ghost::Function LibraryOpenCL::lookupFunction(const std::string& name) const {
  cl_int err;
  opencl::ptr<cl_kernel> kernel(clCreateKernel(program, name.c_str(), &err));
  checkError(err);
  auto f = std::make_shared<FunctionOpenCL>(_dev, kernel);
  return ghost::Function(f);
}
}  // namespace implementation
}  // namespace ghost
#endif
