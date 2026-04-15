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

#include <ghost/cpu/device.h>
#include <ghost/device.h>
#include <ghost/exception.h>

#include <algorithm>
#include <optional>

#if WITH_METAL
#include <ghost/metal/device.h>
#endif
#if WITH_CUDA
#include <ghost/cuda/device.h>
#endif
#if WITH_OPENCL
#include <ghost/opencl/device.h>
#endif
#if WITH_VULKAN
#include <ghost/vulkan/device.h>
#endif
#if WITH_DIRECTX
#include <ghost/directx/device.h>
#endif

namespace ghost {

std::string backendName(Backend backend) {
  switch (backend) {
    case Backend::CPU:
      return "CPU";
    case Backend::Metal:
      return "Metal";
    case Backend::OpenCL:
      return "OpenCL";
    case Backend::CUDA:
      return "CUDA";
    case Backend::Vulkan:
      return "Vulkan";
    case Backend::DirectX:
      return "DirectX";
  }
  return "Unknown";
}

std::vector<Backend> availableBackends() {
  return {
      Backend::CPU,
#if WITH_METAL
      Backend::Metal,
#endif
#if WITH_OPENCL
      Backend::OpenCL,
#endif
#if WITH_CUDA
      Backend::CUDA,
#endif
#if WITH_VULKAN
      Backend::Vulkan,
#endif
#if WITH_DIRECTX
      Backend::DirectX,
#endif
  };
}

std::unique_ptr<Device> createDevice(Backend backend) {
  try {
    switch (backend) {
      case Backend::CPU:
        return std::make_unique<DeviceCPU>();
      case Backend::Metal:
#if WITH_METAL
        return std::make_unique<DeviceMetal>();
#else
        return nullptr;
#endif
      case Backend::OpenCL:
#if WITH_OPENCL
        return std::make_unique<DeviceOpenCL>();
#else
        return nullptr;
#endif
      case Backend::CUDA:
#if WITH_CUDA
        return std::make_unique<DeviceCUDA>();
#else
        return nullptr;
#endif
      case Backend::Vulkan:
#if WITH_VULKAN
        return std::make_unique<DeviceVulkan>();
#else
        return nullptr;
#endif
      case Backend::DirectX:
#if WITH_DIRECTX
        return std::make_unique<DeviceDirectX>();
#else
        return nullptr;
#endif
    }
  } catch (...) {
    // A failed construction may have left deferred errors from destructors
    // (e.g. null-handle release calls from partially-initialized backend
    // handles). Drop them so they don't poison the next unrelated call on
    // this thread.
    try {
      detail::drainErrors();
    } catch (...) {
    }
    return nullptr;
  }
  return nullptr;
}

std::unique_ptr<Device> createDevice(bool allowCPU) {
  const Backend order[] = {
#if defined(__APPLE__)
      Backend::Metal,
      Backend::OpenCL,
      Backend::Vulkan,
#elif defined(_WIN32)
      Backend::CUDA,
      Backend::DirectX,
      Backend::OpenCL,
      Backend::Vulkan,
#else
      Backend::CUDA,
      Backend::OpenCL,
      Backend::Vulkan,
#endif
  };

  for (auto backend : order) {
    if (auto device = createDevice(backend)) return device;
  }

  if (allowCPU) return createDevice(Backend::CPU);

  return nullptr;
}

std::vector<GpuInfo> enumerateDevices() {
  std::vector<GpuInfo> result;
  auto append = [&](std::vector<GpuInfo>&& devs) {
    result.insert(result.end(), std::make_move_iterator(devs.begin()),
                  std::make_move_iterator(devs.end()));
  };

#if WITH_METAL
  try {
    append(DeviceMetal::enumerateDevices());
  } catch (...) {
  }
#endif
#if WITH_CUDA
  try {
    append(DeviceCUDA::enumerateDevices());
  } catch (...) {
  }
#endif
#if WITH_OPENCL
  try {
    append(DeviceOpenCL::enumerateDevices());
  } catch (...) {
  }
#endif
#if WITH_VULKAN
  try {
    append(DeviceVulkan::enumerateDevices());
  } catch (...) {
  }
#endif
#if WITH_DIRECTX
  try {
    append(DeviceDirectX::enumerateDevices());
  } catch (...) {
  }
#endif

  return result;
}

std::optional<GpuInfo> preferredDevice(const std::vector<GpuInfo>& devices) {
  if (devices.empty()) return std::nullopt;

  // Filter out CPU devices.
  std::vector<const GpuInfo*> gpus;
  for (auto& d : devices) {
    if (d.implementation != "CPU") gpus.push_back(&d);
  }
  if (gpus.empty()) return std::nullopt;

  // Sort: discrete before integrated, then by VRAM descending.
  std::sort(gpus.begin(), gpus.end(), [](const GpuInfo* a, const GpuInfo* b) {
    if (a->unifiedMemory != b->unifiedMemory)
      return !a->unifiedMemory;  // discrete (non-unified) first
    return a->memory > b->memory;
  });

  return *gpus[0];
}

std::optional<GpuInfo> preferredDevice(std::optional<Backend> backend) {
  auto all = enumerateDevices();
  if (!backend) return preferredDevice(all);

  std::string name = backendName(*backend);
  std::vector<GpuInfo> filtered;
  for (auto& d : all) {
    if (d.implementation == name) filtered.push_back(std::move(d));
  }
  return preferredDevice(filtered);
}

}  // namespace ghost
