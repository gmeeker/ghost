#include "ghost_test.h"

using namespace ghost;
using namespace ghost::test;

// ---------------------------------------------------------------------------
// Binary loading tests — load pre-compiled binaries via loadLibraryFromData
// ---------------------------------------------------------------------------

class BinaryLoadTest : public GhostTest {};

// ---------------------------------------------------------------------------
// CUDA PTX loading
// ---------------------------------------------------------------------------

#ifdef GHOST_TEST_CUDA_PTX
#include GHOST_TEST_CUDA_PTX
#endif

TEST_P(BinaryLoadTest, CudaPtxLoad) {
  if (backend() != Backend::CUDA) GTEST_SKIP() << "CUDA-only test";
#ifndef GHOST_TEST_CUDA_PTX
  GTEST_SKIP() << "CUDA PTX not compiled at build time";
#else
  // PTX is text — pass with len=0 to signal null-terminated PTX string
  auto lib = device().loadLibraryFromData(ghost_test_cuda_ptx, 0);
  auto fn = lib.lookupFunction("mult_const_f");

  const size_t N = 32;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto inBuf = device().allocateBuffer(N * sizeof(float));
  auto outBuf = device().allocateBuffer(N * sizeof(float));
  inBuf.copy(stream(), input.data(), N * sizeof(float));

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(1);
  fn(la, stream())(outBuf, inBuf, 1.5f);
  outBuf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 1.5f) << "index " << i;
  }
#endif
}

// ---------------------------------------------------------------------------
// Metal metallib loading
// ---------------------------------------------------------------------------

#ifdef GHOST_TEST_METALLIB
#include GHOST_TEST_METALLIB
#endif

TEST_P(BinaryLoadTest, MetalMetallibLoad) {
  if (backend() != Backend::Metal) GTEST_SKIP() << "Metal-only test";
#ifndef GHOST_TEST_METALLIB
  GTEST_SKIP() << "Metal metallib not compiled at build time";
#else
  auto lib = device().loadLibraryFromData(ghost_test_metallib,
                                          ghost_test_metallib_size);
  auto fn = lib.lookupFunction("mult_const_f");

  const size_t N = 32;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto inBuf = device().allocateBuffer(N * sizeof(float));
  auto outBuf = device().allocateBuffer(N * sizeof(float));
  inBuf.copy(stream(), input.data(), N * sizeof(float));

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(1);
  fn(la, stream())(outBuf, inBuf, 1.5f);
  outBuf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 1.5f) << "index " << i;
  }
#endif
}

// ---------------------------------------------------------------------------
// Vulkan SPIR-V loading
// ---------------------------------------------------------------------------

#ifdef GHOST_TEST_VULKAN_SPIRV
#include GHOST_TEST_VULKAN_SPIRV
#endif

TEST_P(BinaryLoadTest, VulkanSpirvLoad) {
  if (backend() != Backend::Vulkan) GTEST_SKIP() << "Vulkan-only test";
#ifndef GHOST_TEST_VULKAN_SPIRV
  GTEST_SKIP() << "Vulkan SPIR-V not compiled at build time";
#else
  auto lib = device().loadLibraryFromData(ghost_test_vulkan_spirv,
                                          ghost_test_vulkan_spirv_size);
  auto fn = lib.lookupFunction("mult_const_f");
  EXPECT_NE(fn.impl().get(), nullptr);
#endif
}

// ---------------------------------------------------------------------------
// Vulkan SPIR-V dispatch (hand-written .spvasm path)
//
// Loads the spvasm-derived SPIR-V — which has a uniform buffer for the
// constant — and actually dispatches it. Exercises both the storage-buffer
// and uniform-buffer descriptor binding paths in FunctionVulkan.
// ---------------------------------------------------------------------------

TEST_P(BinaryLoadTest, VulkanSpirvDispatch) {
  if (backend() != Backend::Vulkan) GTEST_SKIP() << "Vulkan-only test";
#ifndef GHOST_TEST_VULKAN_SPIRV
  GTEST_SKIP() << "Vulkan SPIR-V not compiled at build time";
#else
  auto lib = device().loadLibraryFromData(ghost_test_vulkan_spirv,
                                          ghost_test_vulkan_spirv_size);
  auto fn = lib.lookupFunction("mult_const_f");

  const size_t N = 64;  // matches the kernel's [numthreads(64,1,1)]
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto inBuf = device().allocateBuffer(N * sizeof(float));
  auto outBuf = device().allocateBuffer(N * sizeof(float));
  inBuf.copy(stream(), input.data(), N * sizeof(float));

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(64);
  fn(la, stream())(outBuf, inBuf, 1.5f);
  outBuf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 1.5f) << "index " << i;
  }
#endif
}

// ---------------------------------------------------------------------------
// Vulkan HLSL → SPIR-V dispatch (dxc path)
//
// Compiles test_kernel.hlsl through dxc -spirv at build time and dispatches
// the result. Skips silently if dxc isn't available, since dxc is an
// optional toolchain dep (LunarG Vulkan SDK on Linux/macOS, Windows SDK or
// LunarG on Windows).
// ---------------------------------------------------------------------------

#ifdef GHOST_TEST_VULKAN_HLSL_SPIRV
#include GHOST_TEST_VULKAN_HLSL_SPIRV
#endif

TEST_P(BinaryLoadTest, VulkanHlslDispatch) {
  if (backend() != Backend::Vulkan) GTEST_SKIP() << "Vulkan-only test";
#ifndef GHOST_TEST_VULKAN_HLSL_SPIRV
  GTEST_SKIP() << "HLSL→SPIR-V via dxc not compiled at build time";
#else
  auto lib = device().loadLibraryFromData(ghost_test_vulkan_hlsl_spirv,
                                          ghost_test_vulkan_hlsl_spirv_size);
  auto fn = lib.lookupFunction("mult_const_f");

  const size_t N = 64;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto inBuf = device().allocateBuffer(N * sizeof(float));
  auto outBuf = device().allocateBuffer(N * sizeof(float));
  inBuf.copy(stream(), input.data(), N * sizeof(float));

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(64);
  fn(la, stream())(outBuf, inBuf, 1.5f);
  outBuf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 1.5f) << "index " << i;
  }
#endif
}

// ---------------------------------------------------------------------------
// DirectX DXIL loading
// ---------------------------------------------------------------------------

#ifdef GHOST_TEST_DIRECTX_DXIL
#include GHOST_TEST_DIRECTX_DXIL
#endif

TEST_P(BinaryLoadTest, DirectXDxilLoad) {
  if (backend() != Backend::DirectX) GTEST_SKIP() << "DirectX-only test";
#ifndef GHOST_TEST_DIRECTX_DXIL
  GTEST_SKIP() << "DirectX DXIL not compiled at build time";
#else
  auto lib = device().loadLibraryFromData(ghost_test_directx_dxil,
                                          ghost_test_directx_dxil_size);
  auto fn = lib.lookupFunction("mult_const_f");
  EXPECT_NE(fn.impl().get(), nullptr);
#endif
}

// ---------------------------------------------------------------------------
// DirectX DXIL dispatch
//
// Compiles test_kernel.hlsl to DXIL via dxc at build time and dispatches
// it through the reflected root signature. Verifies the UAV/SRV/CBV
// binding paths in FunctionDirectX with a real HLSL kernel.
// ---------------------------------------------------------------------------

TEST_P(BinaryLoadTest, DirectXDxilDispatch) {
  if (backend() != Backend::DirectX) GTEST_SKIP() << "DirectX-only test";
#ifndef GHOST_TEST_DIRECTX_DXIL
  GTEST_SKIP() << "DirectX DXIL not compiled at build time";
#else
  auto lib = device().loadLibraryFromData(ghost_test_directx_dxil,
                                          ghost_test_directx_dxil_size);
  auto fn = lib.lookupFunction("mult_const_f");

  const size_t N = 64;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto inBuf = device().allocateBuffer(N * sizeof(float));
  auto outBuf = device().allocateBuffer(N * sizeof(float));
  inBuf.copy(stream(), input.data(), N * sizeof(float));

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(64);
  fn(la, stream())(outBuf, inBuf, 1.5f);
  outBuf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 1.5f) << "index " << i;
  }
#endif
}

// ---------------------------------------------------------------------------
// OpenCL SPIR-V loading
// ---------------------------------------------------------------------------

#ifdef GHOST_TEST_OPENCL_SPIRV
#include GHOST_TEST_OPENCL_SPIRV
#endif

TEST_P(BinaryLoadTest, OpenCLSpirvLoad) {
  if (backend() != Backend::OpenCL) GTEST_SKIP() << "OpenCL-only test";
#ifndef GHOST_TEST_OPENCL_SPIRV
  GTEST_SKIP() << "OpenCL SPIR-V not compiled at build time";
#else
  // OpenCL SPIR-V loading requires clCreateProgramWithIL (CL 2.1+).
  // Skip gracefully if not available (e.g., macOS CL 1.2).
  try {
    auto lib = device().loadLibraryFromData(ghost_test_opencl_spirv,
                                            ghost_test_opencl_spirv_size);
    auto fn = lib.lookupFunction("mult_const_f");
    EXPECT_NE(fn.impl().get(), nullptr);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "OpenCL SPIR-V loading not supported: " << e.what();
  }
#endif
}

// ---------------------------------------------------------------------------
// Verify loadLibraryFromText throws on backends without runtime compilation
// ---------------------------------------------------------------------------

TEST_P(BinaryLoadTest, LoadFromTextUnsupported) {
#if WITH_CUDA && !WITH_CUDA_NVRTC
  if (backend() == Backend::CUDA) {
    EXPECT_THROW(device().loadLibraryFromText("invalid"),
                 ghost::unsupported_error);
    return;
  }
#endif
  if (backend() == Backend::Vulkan) {
    EXPECT_THROW(device().loadLibraryFromText("invalid"),
                 ghost::unsupported_error);
    return;
  }
  if (backend() == Backend::DirectX) {
    EXPECT_THROW(device().loadLibraryFromText("invalid"),
                 ghost::unsupported_error);
    return;
  }
  GTEST_SKIP() << "Backend supports loadLibraryFromText";
}

GHOST_INSTANTIATE_BACKEND_TESTS(BinaryLoadTest);
