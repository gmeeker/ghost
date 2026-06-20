#include <memory>
#include <vector>

#include "ghost/executable.h"
#include "ghost_test.h"

#if WITH_VULKAN && defined(GHOST_TEST_VULKAN_SPIRV)
#include GHOST_TEST_VULKAN_SPIRV
#endif

using namespace ghost;
using namespace ghost::test;

class ExecutableTest : public GhostKernelTest {};

// ---------------------------------------------------------------------------
// Basic compile / submit
// ---------------------------------------------------------------------------

TEST_P(ExecutableTest, CompileEmptyAndSubmit) {
  CommandBuffer cb(device());
  Executable exec = cb.compile();
  EXPECT_TRUE(static_cast<bool>(exec));
  EXPECT_NO_THROW(exec.submit(stream()));
  EXPECT_NO_THROW(stream().sync());
}

// A compiled copy executes the same as an immediate / cb-replayed copy.
TEST_P(ExecutableTest, CompileCopyParity) {
  const size_t N = 16;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto src = device().allocateBuffer(N * sizeof(float));
  auto dst = device().allocateBuffer(N * sizeof(float));
  src.copy(stream(), input.data(), N * sizeof(float));
  stream().sync();

  CommandBuffer cb(device());
  dst.copy(cb, src, N * sizeof(float));
  Executable exec = cb.compile();
  exec.submit(stream());

  dst.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i)) << "index " << i;
  }
}

// The snapshot is independent: compile(), then reset/reuse the source cb,
// must not disturb the Executable.
TEST_P(ExecutableTest, CompileSnapshotIsIndependent) {
  const size_t N = 16;
  std::vector<float> a(N, 1.0f), b(N, 2.0f), output(N, 0.0f);

  auto srcA = device().allocateBuffer(N * sizeof(float));
  auto srcB = device().allocateBuffer(N * sizeof(float));
  auto dst = device().allocateBuffer(N * sizeof(float));
  srcA.copy(stream(), a.data(), N * sizeof(float));
  srcB.copy(stream(), b.data(), N * sizeof(float));
  stream().sync();

  CommandBuffer cb(device());
  dst.copy(cb, srcA, N * sizeof(float));
  Executable exec = cb.compile();

  // Reset and record something different into the SAME cb. The compiled
  // Executable must still copy srcA.
  cb.reset();
  dst.copy(cb, srcB, N * sizeof(float));

  exec.submit(stream());
  dst.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();
  for (size_t i = 0; i < N; i++) EXPECT_FLOAT_EQ(output[i], 1.0f) << i;
}

// ---------------------------------------------------------------------------
// Dispatch parity
// ---------------------------------------------------------------------------

TEST_P(ExecutableTest, CompileDispatchParity) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  const size_t N = 256;
  const uint32_t localSize = 64;

  std::vector<float> input(N, 0.0f), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");

  auto inBuf = device().allocateBuffer(N * sizeof(float));
  auto outBuf = device().allocateBuffer(N * sizeof(float));
  inBuf.copy(stream(), input.data(), N * sizeof(float));
  stream().sync();

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(localSize);

  CommandBuffer cb(device());
  fn(la, cb)(outBuf, inBuf, 1.5f);
  Executable exec = cb.compile();
  exec.submit(stream());

  outBuf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 1.5f) << "index " << i;
  }
}

// Resubmitting the same Executable repeatedly is stable (the core use case:
// hold it and submit each frame).
TEST_P(ExecutableTest, ResubmitManyTimes) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  const size_t N = 128;
  std::vector<float> input(N, 0.0f), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");

  auto inBuf = device().allocateBuffer(N * sizeof(float));
  auto outBuf = device().allocateBuffer(N * sizeof(float));
  inBuf.copy(stream(), input.data(), N * sizeof(float));
  stream().sync();

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(1);

  CommandBuffer cb(device());
  fn(la, cb)(outBuf, inBuf, 2.0f);
  Executable exec = cb.compile();

  for (int frame = 0; frame < 5; frame++) {
    exec.submit(stream());
    stream().sync();
    outBuf.copyTo(stream(), output.data(), N * sizeof(float));
    stream().sync();
    for (size_t i = 0; i < N; i++) {
      EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 2.0f)
          << "frame " << frame << " index " << i;
    }
  }
}

// ---------------------------------------------------------------------------
// update(): rebind to new buffers across "frames"
// ---------------------------------------------------------------------------

TEST_P(ExecutableTest, UpdateRebindsBuffers) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  const size_t N = 128;
  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(1);

  // Compile once against the first frame's buffers.
  std::vector<float> input(N);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto in0 = device().allocateBuffer(N * sizeof(float));
  auto out0 = device().allocateBuffer(N * sizeof(float));
  in0.copy(stream(), input.data(), N * sizeof(float));
  stream().sync();

  CommandBuffer cb(device());
  fn(la, cb)(out0, in0, 2.0f);
  Executable exec = cb.compile();
  exec.submit(stream());
  stream().sync();

  std::vector<float> output(N, 0.0f);
  out0.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();
  for (size_t i = 0; i < N; i++)
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 2.0f) << "frame0 " << i;

  // Subsequent frames: same shape, new buffers + new scale, via update().
  for (int frame = 1; frame <= 3; frame++) {
    auto inN = device().allocateBuffer(N * sizeof(float));
    auto outN = device().allocateBuffer(N * sizeof(float));
    inN.copy(stream(), input.data(), N * sizeof(float));
    stream().sync();

    float scale = static_cast<float>(frame + 2);
    cb.reset();
    fn(la, cb)(outN, inN, scale);
    exec.update(cb);
    // CUDA: a dispatch-only graph with unchanged topology must take the
    // in-place node-param patch path (no re-capture / re-instantiate).
    if (GetParam() == Backend::CUDA) {
      EXPECT_TRUE(exec.lastUpdatePatched()) << "frame " << frame;
    }
    exec.submit(stream());
    stream().sync();

    std::fill(output.begin(), output.end(), 0.0f);
    outN.copyTo(stream(), output.data(), N * sizeof(float));
    stream().sync();
    for (size_t i = 0; i < N; i++) {
      EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * scale)
          << "frame " << frame << " index " << i;
    }
  }
}

// ---------------------------------------------------------------------------
// Sub-range / segmented capture
// ---------------------------------------------------------------------------

// A compute dispatch bracketed by host upload + readback (both NOT capturable).
// Marking just the compute span lets it compile natively while the transfers
// replay on the stream — instead of the whole sequence dropping to replay.
TEST_P(ExecutableTest, CompiledRegionBracketedByHostTransfers) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  const size_t N = 128;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");

  auto inBuf = device().allocateBuffer(N * sizeof(float));
  auto outBuf = device().allocateBuffer(N * sizeof(float));

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(1);

  CommandBuffer cb(device());
  inBuf.copy(cb, input.data(), N * sizeof(float));  // H2D upload (replayed)
  cb.beginCompiledRegion();
  fn(la, cb)(outBuf, inBuf, 2.0f);  // compute (compiled)
  cb.endCompiledRegion();
  outBuf.copyTo(cb, output.data(),
                N * sizeof(float));  // D2H readback (replayed)

  Executable exec = cb.compile();
  // On CUDA the marked region is a real graph → accelerated, even though the
  // whole sequence isn't capturable. (OpenCL/NVIDIA: region falls back to
  // replay — no command_buffer — but the result is still correct.)
  if (GetParam() == Backend::CUDA) EXPECT_TRUE(exec.accelerated());

  exec.submit(stream());
  stream().sync();
  for (size_t i = 0; i < N; i++)
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 2.0f) << "idx " << i;

  // Update (new scale), keeping the same shape/region; resubmit.
  cb.reset();
  inBuf.copy(cb, input.data(), N * sizeof(float));
  cb.beginCompiledRegion();
  fn(la, cb)(outBuf, inBuf, 5.0f);
  cb.endCompiledRegion();
  std::fill(output.begin(), output.end(), 0.0f);
  outBuf.copyTo(cb, output.data(), N * sizeof(float));
  exec.update(cb);
  exec.submit(stream());
  stream().sync();
  for (size_t i = 0; i < N; i++)
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 5.0f) << "update " << i;
}

// ---------------------------------------------------------------------------
// Capability / options
// ---------------------------------------------------------------------------

// accelerated() reflects whether a native graph backs the Executable. The
// fallback (everything pre-Phase-2, and any non-accelerated backend) reports
// false; a native backing reports true. Either way the result must be valid.
TEST_P(ExecutableTest, AcceleratedFlagIsConsistent) {
  CommandBuffer cb(device());
  auto buf = device().allocateBuffer(64);
  buf.fill(cb, 0, 64, static_cast<uint8_t>(0));
  Executable exec = cb.compile();
  bool native = exec.accelerated();

  // requireAccelerated must agree with accelerated(): if the backend can't
  // accelerate, it throws; if it can, it returns an accelerated Executable.
  if (native) {
    Executable exec2 = cb.compile(CompileOptions{/*requireAccelerated=*/true});
    EXPECT_TRUE(exec2.accelerated());
  } else {
    EXPECT_THROW(cb.compile(CompileOptions{/*requireAccelerated=*/true}),
                 ghost::unsupported_error);
  }
}

GHOST_INSTANTIATE_KERNEL_TESTS(ExecutableTest);

// ---------------------------------------------------------------------------
// Vulkan native Executable smoke test
//
// ExecutableTest is parameterized over kernelBackends() which excludes Vulkan
// (no runtime text-to-SPIRV path), so this standalone test exercises the
// native ExecutableVulkan path with kernel-free ops: record a copy/barrier/
// copy chain once at compile(), resubmit it twice without re-recording, then
// update() to new source buffers and resubmit.
// ---------------------------------------------------------------------------

#if WITH_VULKAN
TEST(ExecutableVulkanSmoke, RecordOnceResubmitAndUpdate) {
  std::unique_ptr<ghost::Device> devPtr;
  try {
    devPtr = ghost::createDevice(ghost::Backend::Vulkan);
  } catch (...) {
  }
  if (!devPtr) GTEST_SKIP() << "No Vulkan device available";
  ghost::Device& dev = *devPtr;
  ghost::Stream s = dev.defaultStream();

  const size_t N = 16;
  std::vector<float> inA(N), inB(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) {
    inA[i] = static_cast<float>(i);
    inB[i] = static_cast<float>(i) * 10.0f;
  }

  auto srcA = dev.allocateBuffer(N * sizeof(float));
  auto srcB = dev.allocateBuffer(N * sizeof(float));
  auto mid = dev.allocateBuffer(N * sizeof(float));
  auto dst = dev.allocateBuffer(N * sizeof(float));
  srcA.copy(s, inA.data(), N * sizeof(float));
  srcB.copy(s, inB.data(), N * sizeof(float));
  s.sync();

  // Record srcA -> mid -> dst (with a barrier) once, compile to an Executable.
  ghost::CommandBuffer cb(dev);
  mid.copy(cb, srcA, N * sizeof(float));
  cb.barrier();
  dst.copy(cb, mid, N * sizeof(float));
  ghost::Executable exec = cb.compile();
  EXPECT_TRUE(exec.accelerated());

  // Resubmit the same pre-recorded VkCommandBuffer twice.
  for (int iter = 0; iter < 2; iter++) {
    exec.submit(s);
    s.sync();
    std::fill(output.begin(), output.end(), -1.0f);
    dst.copyTo(s, output.data(), N * sizeof(float));
    s.sync();
    for (size_t i = 0; i < N; i++)
      EXPECT_FLOAT_EQ(output[i], static_cast<float>(i))
          << "iter " << iter << " idx " << i;
  }

  // Update to copy srcB instead (re-record), submit, verify.
  cb.reset();
  mid.copy(cb, srcB, N * sizeof(float));
  cb.barrier();
  dst.copy(cb, mid, N * sizeof(float));
  exec.update(cb);
  exec.submit(s);
  s.sync();
  std::fill(output.begin(), output.end(), -1.0f);
  dst.copyTo(s, output.data(), N * sizeof(float));
  s.sync();
  for (size_t i = 0; i < N; i++)
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 10.0f)
        << "update idx " << i;
}

// Dispatch through a Vulkan Executable: verifies the descriptor sets bound in
// the recorded VkCommandBuffer stay valid across resubmits (they are allocated
// once at record time and never freed), and that update() re-records cleanly.
TEST(ExecutableVulkanSmoke, DispatchRecordResubmitUpdate) {
#ifndef GHOST_TEST_VULKAN_SPIRV
  GTEST_SKIP() << "Vulkan SPIR-V not compiled at build time";
#else
  std::unique_ptr<ghost::Device> devPtr;
  try {
    devPtr = ghost::createDevice(ghost::Backend::Vulkan);
  } catch (...) {
  }
  if (!devPtr) GTEST_SKIP() << "No Vulkan device available";
  ghost::Device& dev = *devPtr;
  ghost::Stream s = dev.defaultStream();

  auto lib = dev.loadLibraryFromData(ghost_test_vulkan_spirv,
                                     ghost_test_vulkan_spirv_size);
  auto fn = lib.lookupFunction("mult_const_f");

  const size_t N = 64;  // matches the kernel's [numthreads(64,1,1)]
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto inBuf = dev.allocateBuffer(N * sizeof(float));
  auto outBuf = dev.allocateBuffer(N * sizeof(float));
  inBuf.copy(s, input.data(), N * sizeof(float));
  s.sync();

  ghost::LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(64);

  ghost::CommandBuffer cb(dev);
  fn(la, cb)(outBuf, inBuf, 1.5f);
  ghost::Executable exec = cb.compile();
  EXPECT_TRUE(exec.accelerated());

  // Resubmit the pre-recorded dispatch twice.
  for (int iter = 0; iter < 2; iter++) {
    exec.submit(s);
    s.sync();
    std::fill(output.begin(), output.end(), -1.0f);
    outBuf.copyTo(s, output.data(), N * sizeof(float));
    s.sync();
    for (size_t i = 0; i < N; i++)
      EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 1.5f)
          << "iter " << iter << " idx " << i;
  }

  // Update to a new scale (re-record), submit, verify.
  cb.reset();
  fn(la, cb)(outBuf, inBuf, 4.0f);
  exec.update(cb);
  exec.submit(s);
  s.sync();
  std::fill(output.begin(), output.end(), -1.0f);
  outBuf.copyTo(s, output.data(), N * sizeof(float));
  s.sync();
  for (size_t i = 0; i < N; i++)
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 4.0f)
        << "update idx " << i;
#endif
}
#endif  // WITH_VULKAN

// ---------------------------------------------------------------------------
// DirectX native Executable smoke test
//
// Written blind on Linux (mirrors CommandBufferDirectXSmoke). The Windows
// agent runs this to validate the ExecutableDirectX path: record a copy/
// barrier/copy chain once at compile(), re-execute the closed command list
// twice without re-recording, then update() to a new source and re-execute.
// ---------------------------------------------------------------------------

#if WITH_DIRECTX
TEST(ExecutableDirectXSmoke, RecordOnceResubmitAndUpdate) {
  std::unique_ptr<ghost::Device> devPtr;
  try {
    devPtr = ghost::createDevice(ghost::Backend::DirectX);
  } catch (...) {
  }
  if (!devPtr) GTEST_SKIP() << "No DirectX device available";
  ghost::Device& dev = *devPtr;
  ghost::Stream s = dev.defaultStream();

  const size_t N = 16;
  std::vector<float> inA(N), inB(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) {
    inA[i] = static_cast<float>(i);
    inB[i] = static_cast<float>(i) * 10.0f;
  }

  auto srcA = dev.allocateBuffer(N * sizeof(float));
  auto srcB = dev.allocateBuffer(N * sizeof(float));
  auto mid = dev.allocateBuffer(N * sizeof(float));
  auto dst = dev.allocateBuffer(N * sizeof(float));
  srcA.copy(s, inA.data(), N * sizeof(float));
  srcB.copy(s, inB.data(), N * sizeof(float));
  s.sync();

  ghost::CommandBuffer cb(dev);
  mid.copy(cb, srcA, N * sizeof(float));
  cb.barrier();
  dst.copy(cb, mid, N * sizeof(float));
  ghost::Executable exec = cb.compile();
  EXPECT_TRUE(exec.accelerated());

  for (int iter = 0; iter < 2; iter++) {
    exec.submit(s);
    s.sync();
    std::fill(output.begin(), output.end(), -1.0f);
    dst.copyTo(s, output.data(), N * sizeof(float));
    s.sync();
    for (size_t i = 0; i < N; i++)
      EXPECT_FLOAT_EQ(output[i], static_cast<float>(i))
          << "iter " << iter << " idx " << i;
  }

  cb.reset();
  mid.copy(cb, srcB, N * sizeof(float));
  cb.barrier();
  dst.copy(cb, mid, N * sizeof(float));
  exec.update(cb);
  exec.submit(s);
  s.sync();
  std::fill(output.begin(), output.end(), -1.0f);
  dst.copyTo(s, output.data(), N * sizeof(float));
  s.sync();
  for (size_t i = 0; i < N; i++)
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 10.0f)
        << "update idx " << i;
}
#endif  // WITH_DIRECTX
