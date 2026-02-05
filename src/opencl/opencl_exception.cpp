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

#include <ghost/opencl/exception.h>

#if __APPLE_CC__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace ghost {
namespace opencl {
runtime_error::runtime_error(int err)
    : std::runtime_error(errorString(err)), _err(err) {}

int32_t runtime_error::error() const noexcept { return _err; }

#define ERR(a) \
  case a:      \
    return #a;

const char* runtime_error::errorString(int32_t err) {
  switch (err) {
    ERR(CL_DEVICE_NOT_FOUND);
    ERR(CL_DEVICE_NOT_AVAILABLE);
    ERR(CL_COMPILER_NOT_AVAILABLE);
    ERR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
    ERR(CL_OUT_OF_RESOURCES);
    ERR(CL_OUT_OF_HOST_MEMORY);
    ERR(CL_PROFILING_INFO_NOT_AVAILABLE);
    ERR(CL_MEM_COPY_OVERLAP);
    ERR(CL_IMAGE_FORMAT_MISMATCH);
    ERR(CL_IMAGE_FORMAT_NOT_SUPPORTED);
    ERR(CL_BUILD_PROGRAM_FAILURE);
    ERR(CL_MAP_FAILURE);
#ifdef CL_MISALIGNED_SUB_BUFFER_OFFSET
    ERR(CL_MISALIGNED_SUB_BUFFER_OFFSET);
#endif
#ifdef CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST
    ERR(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
#endif
    ERR(CL_INVALID_VALUE);
    ERR(CL_INVALID_DEVICE_TYPE);
    ERR(CL_INVALID_PLATFORM);
    ERR(CL_INVALID_DEVICE);
    ERR(CL_INVALID_CONTEXT);
    ERR(CL_INVALID_QUEUE_PROPERTIES);
    ERR(CL_INVALID_COMMAND_QUEUE);
    ERR(CL_INVALID_HOST_PTR);
    ERR(CL_INVALID_MEM_OBJECT);
    ERR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
    ERR(CL_INVALID_IMAGE_SIZE);
    ERR(CL_INVALID_SAMPLER);
    ERR(CL_INVALID_BINARY);
    ERR(CL_INVALID_BUILD_OPTIONS);
    ERR(CL_INVALID_PROGRAM);
    ERR(CL_INVALID_PROGRAM_EXECUTABLE);
    ERR(CL_INVALID_KERNEL_NAME);
    ERR(CL_INVALID_KERNEL_DEFINITION);
    ERR(CL_INVALID_KERNEL);
    ERR(CL_INVALID_ARG_INDEX);
    ERR(CL_INVALID_ARG_VALUE);
    ERR(CL_INVALID_ARG_SIZE);
    ERR(CL_INVALID_KERNEL_ARGS);
    ERR(CL_INVALID_WORK_DIMENSION);
    ERR(CL_INVALID_WORK_GROUP_SIZE);
    ERR(CL_INVALID_WORK_ITEM_SIZE);
    ERR(CL_INVALID_GLOBAL_OFFSET);
    ERR(CL_INVALID_EVENT_WAIT_LIST);
    ERR(CL_INVALID_EVENT);
    ERR(CL_INVALID_OPERATION);
    ERR(CL_INVALID_GL_OBJECT);
    ERR(CL_INVALID_BUFFER_SIZE);
    ERR(CL_INVALID_MIP_LEVEL);
    ERR(CL_INVALID_GLOBAL_WORK_SIZE);
#ifdef CL_INVALID_PROPERTY
    ERR(CL_INVALID_PROPERTY);
#endif
#ifdef CL_INVALID_IMAGE_DESCRIPTOR
    ERR(CL_INVALID_IMAGE_DESCRIPTOR);
#endif
#ifdef CL_INVALID_COMPILER_OPTIONS
    ERR(CL_INVALID_COMPILER_OPTIONS);
#endif
#ifdef CL_INVALID_LINKER_OPTIONS
    ERR(CL_INVALID_LINKER_OPTIONS);
#endif
#ifdef CL_INVALID_DEVICE_PARTITION_COUNT
    ERR(CL_INVALID_DEVICE_PARTITION_COUNT);
#endif
#if defined(CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR)
    ERR(CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR);
#elif defined(CL_INVALID_GL_CONTEXT_APPLE)
    ERR(CL_INVALID_GL_CONTEXT_APPLE);
#endif
#if defined(CL_PLATFORM_NOT_FOUND_KHR)
    ERR(CL_PLATFORM_NOT_FOUND_KHR);
#endif
#ifdef CL_INVALID_D3D10_DEVICE_KHR
    ERR(CL_INVALID_D3D10_DEVICE_KHR);
#endif
#ifdef CL_INVALID_D3D10_RESOURCE_KHR
    ERR(CL_INVALID_D3D10_RESOURCE_KHR);
#endif
#ifdef CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR
    ERR(CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR);
#endif
#ifdef CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR
    ERR(CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR);
#endif
#ifdef CL_INVALID_PARTITION_COUNT_EXT
    ERR(CL_INVALID_PARTITION_COUNT_EXT);
#endif
#ifdef CL_INVALID_PARTITION_NAME_EXT
    ERR(CL_INVALID_PARTITION_NAME_EXT);
#endif
    case CL_SUCCESS:
      return nullptr;
    default:
      return "Unknown";
  }
}
}  // namespace opencl
}  // namespace ghost

#endif
