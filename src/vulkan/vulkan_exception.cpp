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

#if WITH_VULKAN

#include <ghost/vulkan/exception.h>

#include <string>

namespace ghost {
namespace vk {

runtime_error::runtime_error(int err)
    : std::runtime_error(std::string("Vulkan error: ") + errorString(err)),
      _err(err) {}

int32_t runtime_error::error() const noexcept { return _err; }

const char* runtime_error::errorString(int32_t err) {
  switch (err) {
    case 0:
      return "VK_SUCCESS";
    case 1:
      return "VK_NOT_READY";
    case 2:
      return "VK_TIMEOUT";
    case 3:
      return "VK_EVENT_SET";
    case 4:
      return "VK_EVENT_RESET";
    case 5:
      return "VK_INCOMPLETE";
    case -1:
      return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case -2:
      return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case -3:
      return "VK_ERROR_INITIALIZATION_FAILED";
    case -4:
      return "VK_ERROR_DEVICE_LOST";
    case -5:
      return "VK_ERROR_MEMORY_MAP_FAILED";
    case -6:
      return "VK_ERROR_LAYER_NOT_PRESENT";
    case -7:
      return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case -8:
      return "VK_ERROR_FEATURE_NOT_PRESENT";
    case -9:
      return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case -10:
      return "VK_ERROR_TOO_MANY_OBJECTS";
    case -11:
      return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case -12:
      return "VK_ERROR_FRAGMENTED_POOL";
    case -13:
      return "VK_ERROR_UNKNOWN";
    case -1000069000:
      return "VK_ERROR_OUT_OF_POOL_MEMORY";
    case -1000072003:
      return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
    case -1000161000:
      return "VK_ERROR_FRAGMENTATION";
    case -1000174001:
      return "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS";
    case -1000000000:
      return "VK_ERROR_SURFACE_LOST_KHR";
    case -1000000001:
      return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
    case -1000001003:
      return "VK_ERROR_OUT_OF_DATE_KHR";
    case -1000003001:
      return "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR";
    case -1000011001:
      return "VK_ERROR_VALIDATION_FAILED_EXT";
    default:
      return "VK_ERROR_UNKNOWN";
  }
}

}  // namespace vk
}  // namespace ghost

#endif
