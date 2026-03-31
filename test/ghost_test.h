#ifndef GHOST_TEST_H
#define GHOST_TEST_H

#include <ghost/command_buffer.h>
#include <ghost/cpu/device.h>
#include <ghost/device.h>
#include <ghost/exception.h>
#include <ghost/gpu_info.h>
#include <gtest/gtest.h>

#if WITH_CUDA
#include <ghost/cuda/device.h>
#endif
#if WITH_DIRECTX
#include <ghost/directx/device.h>
#endif
#if WITH_METAL
#include <ghost/metal/device.h>
#endif
#if WITH_OPENCL
#include <ghost/opencl/device.h>
#endif
#if WITH_VULKAN
#include <ghost/vulkan/device.h>
#endif

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace ghost {
namespace test {

// ---------------------------------------------------------------------------
// Backend enumeration
// ---------------------------------------------------------------------------

enum class Backend { CPU, Metal, OpenCL, CUDA, Vulkan, DirectX };

inline std::string BackendName(Backend b) {
  switch (b) {
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

// Backends compiled into this build.
inline std::vector<Backend> availableBackends() {
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

// Backends that support loadLibraryFromText (GPU backends with runtime
// compilation).  CPU only supports loadLibraryFromFile (shared libraries).
// Vulkan/DirectX use binary shaders and are excluded for now.
inline std::vector<Backend> kernelBackends() {
  std::vector<Backend> backends;
#if WITH_METAL
  backends.push_back(Backend::Metal);
#endif
#if WITH_OPENCL
  backends.push_back(Backend::OpenCL);
#endif
#if WITH_CUDA
  backends.push_back(Backend::CUDA);
#endif
  return backends;
}

// ---------------------------------------------------------------------------
// Device factory
// ---------------------------------------------------------------------------

inline std::unique_ptr<Device> createDevice(Backend backend) {
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
  return nullptr;
}

// ---------------------------------------------------------------------------
// GpuInfo enumeration per backend
// ---------------------------------------------------------------------------

inline std::vector<GpuInfo> enumerateDevices(Backend backend) {
  switch (backend) {
    case Backend::CPU:
      return DeviceCPU::enumerateDevices();
    case Backend::Metal:
#if WITH_METAL
      return DeviceMetal::enumerateDevices();
#else
      return {};
#endif
    case Backend::OpenCL:
#if WITH_OPENCL
      return DeviceOpenCL::enumerateDevices();
#else
      return {};
#endif
    case Backend::CUDA:
#if WITH_CUDA
      return DeviceCUDA::enumerateDevices();
#else
      return {};
#endif
    case Backend::Vulkan:
#if WITH_VULKAN
      return DeviceVulkan::enumerateDevices();
#else
      return {};
#endif
    case Backend::DirectX:
#if WITH_DIRECTX
      return DeviceDirectX::enumerateDevices();
#else
      return {};
#endif
  }
  return {};
}

// ---------------------------------------------------------------------------
// Per-backend kernel sources
// ---------------------------------------------------------------------------

// mult_const_f: out[i] = A[i] * scale
inline const char* multConstKernelSource(Backend backend) {
  switch (backend) {
    case Backend::OpenCL:
#if WITH_OPENCL
      return R"(
__kernel void mult_const_f(__global float *out, __global const float *A, float scale) {
    int tid = get_global_id(0);
    out[tid] = A[tid] * scale;
})";
#else
      return nullptr;
#endif
    case Backend::Metal:
#if WITH_METAL
      return R"(
#include <metal_stdlib>
using namespace metal;
kernel void mult_const_f(device float* out [[buffer(0)]],
                         device const float* A [[buffer(1)]],
                         uint index [[thread_position_in_grid]],
                         constant float& scale [[buffer(2)]]) {
    out[index] = A[index] * scale;
})";
#else
      return nullptr;
#endif
    case Backend::CUDA:
#if WITH_CUDA
      return R"(
extern "C" __global__ void mult_const_f(float* out, const float* A, float scale) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    out[tid] = A[tid] * scale;
})";
#else
      return nullptr;
#endif
    default:
      return nullptr;
  }
}

// add_buffers: out[i] = A[i] + B[i]
inline const char* addBuffersKernelSource(Backend backend) {
  switch (backend) {
    case Backend::OpenCL:
#if WITH_OPENCL
      return R"(
__kernel void add_buffers(__global float *out,
                          __global const float *A,
                          __global const float *B) {
    int tid = get_global_id(0);
    out[tid] = A[tid] + B[tid];
})";
#else
      return nullptr;
#endif
    case Backend::Metal:
#if WITH_METAL
      return R"(
#include <metal_stdlib>
using namespace metal;
kernel void add_buffers(device float* out [[buffer(0)]],
                        device const float* A [[buffer(1)]],
                        device const float* B [[buffer(2)]],
                        uint index [[thread_position_in_grid]]) {
    out[index] = A[index] + B[index];
})";
#else
      return nullptr;
#endif
    case Backend::CUDA:
#if WITH_CUDA
      return R"(
extern "C" __global__ void add_buffers(float* out, const float* A, const float* B) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    out[tid] = A[tid] + B[tid];
})";
#else
      return nullptr;
#endif
    default:
      return nullptr;
  }
}

// mult_const_2d: out[y * W + x] = A[y * W + x] * scale
inline const char* multConst2DKernelSource(Backend backend) {
  switch (backend) {
    case Backend::OpenCL:
#if WITH_OPENCL
      return R"(
__kernel void mult_const_2d(__global float *out, __global const float *A, float scale, int W) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx = y * W + x;
    out[idx] = A[idx] * scale;
})";
#else
      return nullptr;
#endif
    case Backend::Metal:
#if WITH_METAL
      return R"(
#include <metal_stdlib>
using namespace metal;
struct MultConst2DParams { float scale; int W; };
kernel void mult_const_2d(device float* out [[buffer(0)]],
                          device const float* A [[buffer(1)]],
                          constant MultConst2DParams& params [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
    int idx = int(gid.y) * params.W + int(gid.x);
    out[idx] = A[idx] * params.scale;
})";
#else
      return nullptr;
#endif
    case Backend::CUDA:
#if WITH_CUDA
      return R"(
extern "C" __global__ void mult_const_2d(float* out, const float* A, float scale, int W) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = y * W + x;
    out[idx] = A[idx] * scale;
})";
#else
      return nullptr;
#endif
    default:
      return nullptr;
  }
}

// mult_const_3d: out[z * H * W + y * W + x] = A[...] * scale
inline const char* multConst3DKernelSource(Backend backend) {
  switch (backend) {
    case Backend::OpenCL:
#if WITH_OPENCL
      return R"(
__kernel void mult_const_3d(__global float *out, __global const float *A, float scale, int W, int H) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int idx = z * H * W + y * W + x;
    out[idx] = A[idx] * scale;
})";
#else
      return nullptr;
#endif
    case Backend::Metal:
#if WITH_METAL
      return R"(
#include <metal_stdlib>
using namespace metal;
struct MultConst3DParams { float scale; int W; int H; };
kernel void mult_const_3d(device float* out [[buffer(0)]],
                          device const float* A [[buffer(1)]],
                          constant MultConst3DParams& params [[buffer(2)]],
                          uint3 gid [[thread_position_in_grid]]) {
    int idx = int(gid.z) * params.H * params.W + int(gid.y) * params.W + int(gid.x);
    out[idx] = A[idx] * params.scale;
})";
#else
      return nullptr;
#endif
    case Backend::CUDA:
#if WITH_CUDA
      return R"(
extern "C" __global__ void mult_const_3d(float* out, const float* A, float scale, int W, int H) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    int idx = z * H * W + y * W + x;
    out[idx] = A[idx] * scale;
})";
#else
      return nullptr;
#endif
    default:
      return nullptr;
  }
}

// local_mem_sum: uses local memory to sum elements
inline const char* localMemKernelSource(Backend backend) {
  switch (backend) {
    case Backend::OpenCL:
#if WITH_OPENCL
      return R"(
__kernel void local_mem_sum(__global float *out, __global const float *A,
                            __local float *scratch) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    scratch[lid] = A[gid];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        float sum = 0.0f;
        int lsize = get_local_size(0);
        for (int i = 0; i < lsize; i++) sum += scratch[i];
        out[get_group_id(0)] = sum;
    }
})";
#else
      return nullptr;
#endif
    case Backend::Metal:
#if WITH_METAL
      return R"(
#include <metal_stdlib>
using namespace metal;
kernel void local_mem_sum(device float* out [[buffer(0)]],
                          device const float* A [[buffer(1)]],
                          threadgroup float* scratch [[threadgroup(0)]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint gid [[thread_position_in_grid]],
                          uint group_id [[threadgroup_position_in_grid]],
                          uint lsize [[threads_per_threadgroup]]) {
    scratch[lid] = A[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < lsize; i++) sum += scratch[i];
        out[group_id] = sum;
    }
})";
#else
      return nullptr;
#endif
    case Backend::CUDA:
#if WITH_CUDA
      return R"(
extern "C" __global__ void local_mem_sum(float* out, const float* A) {
    extern __shared__ float scratch[];
    int lid = threadIdx.x;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    scratch[lid] = A[gid];
    __syncthreads();
    if (lid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; i++) sum += scratch[i];
        out[blockIdx.x] = sum;
    }
})";
#else
      return nullptr;
#endif
    default:
      return nullptr;
  }
}

// Metal kernel with function constants for specialization testing.
// When USE_SCALE is true, out[i] = A[i] * scale; otherwise out[i] = A[i].
inline const char* specializedKernelSource() {
#if WITH_METAL
  return R"(
#include <metal_stdlib>
using namespace metal;
constant bool USE_SCALE [[function_constant(0)]];
kernel void specialized_fn(device float* out [[buffer(0)]],
                           device const float* A [[buffer(1)]],
                           constant float& scale [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
    if (USE_SCALE)
        out[index] = A[index] * scale;
    else
        out[index] = A[index];
})";
#else
  return nullptr;
#endif
}

// ---------------------------------------------------------------------------
// Test name generator for parameterized tests
// ---------------------------------------------------------------------------

struct BackendNameGenerator {
  std::string operator()(const testing::TestParamInfo<Backend>& info) const {
    return BackendName(info.param);
  }
};

// ---------------------------------------------------------------------------
// Base test fixture: creates a device, skips if hardware unavailable
// ---------------------------------------------------------------------------

class GhostTest : public testing::TestWithParam<Backend> {
 protected:
  void SetUp() override {
    try {
      device_ = createDevice(GetParam());
      if (!device_) {
        GTEST_SKIP() << BackendName(GetParam()) << " not compiled";
      }
    } catch (const std::exception& e) {
      GTEST_SKIP() << BackendName(GetParam()) << " unavailable: " << e.what();
    }
  }

  Device& device() { return *device_; }

  Stream stream() { return device_->defaultStream(); }

  Backend backend() const { return GetParam(); }

  std::unique_ptr<Device> device_;
};

// ---------------------------------------------------------------------------
// Kernel test fixture: also requires loadLibraryFromText support
// ---------------------------------------------------------------------------

class GhostKernelTest : public GhostTest {
 protected:
  void SetUp() override {
    GhostTest::SetUp();
    if (testing::Test::IsSkipped()) return;

    if (GetParam() == Backend::CPU) {
      GTEST_SKIP() << "CPU does not support loadLibraryFromText";
    }
  }

  const char* multConstSource() const {
    return multConstKernelSource(GetParam());
  }

  const char* addBuffersSource() const {
    return addBuffersKernelSource(GetParam());
  }

  const char* multConst2DSource() const {
    return multConst2DKernelSource(GetParam());
  }

  const char* multConst3DSource() const {
    return multConst3DKernelSource(GetParam());
  }

  const char* localMemSource() const {
    return localMemKernelSource(GetParam());
  }
};

}  // namespace test
}  // namespace ghost

// Convenience macros for test suite instantiation.
#define GHOST_INSTANTIATE_BACKEND_TESTS(suite)                                 \
  INSTANTIATE_TEST_SUITE_P(                                                    \
      AllBackends, suite, testing::ValuesIn(ghost::test::availableBackends()), \
      ghost::test::BackendNameGenerator())

#define GHOST_INSTANTIATE_KERNEL_TESTS(suite)                                \
  INSTANTIATE_TEST_SUITE_P(KernelBackends, suite,                            \
                           testing::ValuesIn(ghost::test::kernelBackends()), \
                           ghost::test::BackendNameGenerator())

#endif  // GHOST_TEST_H
