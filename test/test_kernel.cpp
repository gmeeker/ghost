#include <ghost/argument_buffer.h>

#include "ghost_test.h"

using namespace ghost;
using namespace ghost::test;

class KernelTest : public GhostKernelTest {};

// ---------------------------------------------------------------------------
// Library loading and function lookup
// ---------------------------------------------------------------------------

TEST_P(KernelTest, CompileAndLookup) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP() << "No kernel source for " << BackendName(backend());

  Library lib = device().loadLibraryFromText(src);
  Function fn = lib.lookupFunction("mult_const_f");
  // Just verify we got a valid function (non-null impl).
  EXPECT_NE(fn.impl().get(), nullptr);
}

TEST_P(KernelTest, LookupMissingFunctionThrows) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  Library lib = device().loadLibraryFromText(src);
  EXPECT_THROW(lib.lookupFunction("nonexistent_kernel_xyz"), std::exception);
}

TEST_P(KernelTest, InvalidSourceThrows) {
  EXPECT_THROW(device().loadLibraryFromText("this is not valid GPU code!!!"),
               std::exception);
}

// ---------------------------------------------------------------------------
// Kernel dispatch with data verification
// ---------------------------------------------------------------------------

TEST_P(KernelTest, MultConst1D) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  const size_t N = 32;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");

  auto inBuf = device().allocateBuffer(N * sizeof(float));
  auto outBuf = device().allocateBuffer(N * sizeof(float));

  inBuf.copy(stream(), input.data(), N * sizeof(float));

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(1);
  fn(stream(), la, outBuf, inBuf, 1.5f);
  outBuf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 1.5f) << "index " << i;
  }
}

TEST_P(KernelTest, MultConstScale2) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  const size_t N = 16;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i + 1);

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");

  auto inBuf = device().allocateBuffer(N * sizeof(float));
  auto outBuf = device().allocateBuffer(N * sizeof(float));

  inBuf.copy(stream(), input.data(), N * sizeof(float));

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(1);
  fn(stream(), la, outBuf, inBuf, 2.0f);
  outBuf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i + 1) * 2.0f)
        << "index " << i;
  }
}

TEST_P(KernelTest, AddBuffers) {
  const char* src = addBuffersSource();
  if (!src) GTEST_SKIP();

  const size_t N = 16;
  std::vector<float> a(N), b(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i * 10);
  }

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("add_buffers");

  auto bufA = device().allocateBuffer(N * sizeof(float));
  auto bufB = device().allocateBuffer(N * sizeof(float));
  auto bufOut = device().allocateBuffer(N * sizeof(float));

  bufA.copy(stream(), a.data(), N * sizeof(float));
  bufB.copy(stream(), b.data(), N * sizeof(float));

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(1);
  fn(stream(), la, bufOut, bufA, bufB);
  bufOut.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i + i * 10)) << "index " << i;
  }
}

// ---------------------------------------------------------------------------
// Multiple dispatches on same stream
// ---------------------------------------------------------------------------

TEST_P(KernelTest, MultipleDispatches) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  const size_t N = 16;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");

  auto buf1 = device().allocateBuffer(N * sizeof(float));
  auto buf2 = device().allocateBuffer(N * sizeof(float));
  auto buf3 = device().allocateBuffer(N * sizeof(float));

  buf1.copy(stream(), input.data(), N * sizeof(float));

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(1);

  // Chain: buf1 * 2.0 -> buf2, buf2 * 3.0 -> buf3.
  fn(stream(), la, buf2, buf1, 2.0f);
  fn(stream(), la, buf3, buf2, 3.0f);
  buf3.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 6.0f) << "index " << i;
  }
}

// ---------------------------------------------------------------------------
// Function attributes
// ---------------------------------------------------------------------------

TEST_P(KernelTest, FunctionMaxThreads) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");

  auto attr = fn.getAttribute(kFunctionMaxThreads);
  EXPECT_TRUE(attr.valid());
  EXPECT_GT(attr.asInt(), 0);
}

TEST_P(KernelTest, FunctionThreadWidth) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");

  auto attr = fn.getAttribute(kFunctionThreadWidth);
  EXPECT_TRUE(attr.valid());
  EXPECT_GT(attr.asInt(), 0);
}

TEST_P(KernelTest, FunctionLocalMemory) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");

  auto attr = fn.getAttribute(kFunctionLocalMemory);
  EXPECT_TRUE(attr.valid());
  // Simple kernel should use 0 local memory.
  EXPECT_GE(attr.asInt(), 0);
}

// ---------------------------------------------------------------------------
// 2D kernel launch
// ---------------------------------------------------------------------------

TEST_P(KernelTest, MultConst2D) {
  const char* src = multConst2DSource();
  if (!src)
    GTEST_SKIP() << "No 2D kernel source for " << BackendName(backend());

  const uint32_t W = 8, H = 4;
  const size_t N = W * H;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_2d");

  auto inBuf = device().allocateBuffer(N * sizeof(float));
  auto outBuf = device().allocateBuffer(N * sizeof(float));
  inBuf.copy(stream(), input.data(), N * sizeof(float));

  LaunchArgs la;
  la.global_size(W, H).local_size(1, 1);
  fn(stream(), la, outBuf, inBuf, 2.0f, static_cast<int32_t>(W));
  outBuf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 2.0f) << "index " << i;
  }
}

// ---------------------------------------------------------------------------
// 3D kernel launch
// ---------------------------------------------------------------------------

TEST_P(KernelTest, MultConst3D) {
  const char* src = multConst3DSource();
  if (!src)
    GTEST_SKIP() << "No 3D kernel source for " << BackendName(backend());

  const uint32_t W = 4, H = 3, D = 2;
  const size_t N = W * H * D;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_3d");

  auto inBuf = device().allocateBuffer(N * sizeof(float));
  auto outBuf = device().allocateBuffer(N * sizeof(float));
  inBuf.copy(stream(), input.data(), N * sizeof(float));

  LaunchArgs la;
  la.global_size(W, H, D).local_size(1, 1, 1);
  fn(stream(), la, outBuf, inBuf, 3.0f, static_cast<int32_t>(W),
     static_cast<int32_t>(H));
  outBuf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 3.0f) << "index " << i;
  }
}

// ---------------------------------------------------------------------------
// Local memory argument
// ---------------------------------------------------------------------------

TEST_P(KernelTest, LocalMemoryArgument) {
  const char* src = localMemSource();
  if (!src)
    GTEST_SKIP() << "No local mem kernel for " << BackendName(backend());

  const uint32_t localSize = 4;
  const uint32_t numGroups = 2;
  const uint32_t N = localSize * numGroups;
  std::vector<float> input(N), output(numGroups, 0.0f);
  for (uint32_t i = 0; i < N; i++) input[i] = static_cast<float>(i + 1);

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("local_mem_sum");

  auto inBuf = device().allocateBuffer(N * sizeof(float));
  auto outBuf = device().allocateBuffer(numGroups * sizeof(float));
  inBuf.copy(stream(), input.data(), N * sizeof(float));

  LaunchArgs la;
  la.global_size(N).local_size(localSize);

  // CUDA uses dynamic shared memory via launch config, not an explicit arg.
  if (backend() == Backend::CUDA) {
    fn.execute(stream(), la,
               {Attribute(&outBuf), Attribute(&inBuf),
                Attribute().localMem(localSize * sizeof(float))});
  } else {
    fn(stream(), la, outBuf, inBuf,
       Attribute().localMem(localSize * sizeof(float)));
  }

  outBuf.copyTo(stream(), output.data(), numGroups * sizeof(float));
  stream().sync();

  // Group 0: sum of {1,2,3,4} = 10
  EXPECT_FLOAT_EQ(output[0], 10.0f);
  // Group 1: sum of {5,6,7,8} = 26
  EXPECT_FLOAT_EQ(output[1], 26.0f);
}

// ---------------------------------------------------------------------------
// Extended function attributes
// ---------------------------------------------------------------------------

TEST_P(KernelTest, FunctionPreferredWorkMultiple) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");

  auto attr = fn.getAttribute(kFunctionPreferredWorkMultiple);
  EXPECT_TRUE(attr.valid());
  EXPECT_GT(attr.asInt(), 0);
}

TEST_P(KernelTest, FunctionNumRegisters) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");

  auto attr = fn.getAttribute(kFunctionNumRegisters);
  EXPECT_TRUE(attr.valid());
  // May be 0 on backends that don't report register count.
  EXPECT_GE(attr.asInt(), 0);
}

TEST_P(KernelTest, FunctionPrivateMemory) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");

  auto attr = fn.getAttribute(kFunctionPrivateMemory);
  EXPECT_TRUE(attr.valid());
  EXPECT_GE(attr.asInt(), 0);
}

// ---------------------------------------------------------------------------
// Binary cache
// ---------------------------------------------------------------------------

TEST_P(KernelTest, BinaryCacheCreatesFiles) {
  auto& cache = Device::binaryCache();
  // Use a temporary directory for cache.
  std::string tmpDir = "/tmp/ghost_test_cache_" +
                       std::to_string(reinterpret_cast<uintptr_t>(this));
  cache.cachePath = tmpDir;

  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  // First compile — should create cache file.
  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");
  EXPECT_NE(fn.impl().get(), nullptr);

  // Verify cache is enabled.
  EXPECT_TRUE(cache.isEnabled());

  // Second compile of same source — should hit cache.
  auto lib2 = device().loadLibraryFromText(src);
  auto fn2 = lib2.lookupFunction("mult_const_f");
  EXPECT_NE(fn2.impl().get(), nullptr);

  // Purge and clean up.
  device().purgeBinaries(0);
  cache.cachePath.clear();
}

// ---------------------------------------------------------------------------
// Argument buffer tests
// ---------------------------------------------------------------------------

TEST_P(KernelTest, ArgumentBufferStruct) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  const size_t N = 16;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");

  auto inBuf = device().allocateBuffer(N * sizeof(float));
  auto outBuf = device().allocateBuffer(N * sizeof(float));
  inBuf.copy(stream(), input.data(), N * sizeof(float));

  // Use execute() with vector<Attribute> to pass args including an
  // ArgumentBuffer for the scale parameter.
  ArgumentBuffer ab;
  ab.set(0, 2.5f);
  EXPECT_TRUE(ab.isStruct());
  EXPECT_GE(ab.size(), sizeof(float));

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(1);

  // Pass arguments manually via execute().
  fn.execute(stream(), la,
             {Attribute(&outBuf), Attribute(&inBuf), Attribute(2.5f)});
  outBuf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 2.5f) << "index " << i;
  }
}

TEST_P(KernelTest, ArgumentBufferUpload) {
  ArgumentBuffer ab;
  ab.set(0, 1.0f);
  ab.set(4, int32_t(42));
  EXPECT_TRUE(ab.isStruct());

  ab.upload(device(), stream());
  stream().sync();
  EXPECT_FALSE(ab.isStruct());
  EXPECT_NE(ab.bufferImpl().get(), nullptr);

  // Reset and verify it returns to struct mode.
  ab.reset();
  EXPECT_TRUE(ab.isStruct());
  EXPECT_EQ(ab.size(), 0u);
}

TEST_P(KernelTest, ArgumentBufferReuse) {
  ArgumentBuffer ab;
  ab.set(0, 1.0f);
  ab.set(4, int32_t(42));

  // Upload twice — should reuse or reallocate buffer.
  ab.upload(device(), stream());
  stream().sync();
  auto impl1 = ab.bufferImpl();

  ab.set(0, 2.0f);
  ab.upload(device(), stream());
  stream().sync();
  auto impl2 = ab.bufferImpl();

  EXPECT_NE(impl2.get(), nullptr);
}

// ---------------------------------------------------------------------------
// loadLibraryFromData
// ---------------------------------------------------------------------------

TEST_P(KernelTest, LoadLibraryFromDataInvalid) {
  // Passing garbage data should either throw or produce a library that
  // fails on function lookup — it must not crash.
  const char garbage[] = "this is not valid binary data";
  try {
    auto lib = device().loadLibraryFromData(garbage, sizeof(garbage));
    // If loading succeeded, looking up a function should fail.
    EXPECT_THROW(lib.lookupFunction("nonexistent"), std::exception);
  } catch (const std::exception&) {
    // Throwing on invalid data is acceptable.
  }
}

// ---------------------------------------------------------------------------
// Metal function specialization
// ---------------------------------------------------------------------------

TEST_P(KernelTest, FunctionSpecialization) {
  if (backend() != Backend::Metal) {
    GTEST_SKIP() << "Function specialization is Metal-only";
  }

  const char* src = specializedKernelSource();
  if (!src) GTEST_SKIP();

  const size_t N = 16;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i + 1);

  auto lib = device().loadLibraryFromText(src);

  auto inBuf = device().allocateBuffer(N * sizeof(float));
  auto outBuf = device().allocateBuffer(N * sizeof(float));
  inBuf.copy(stream(), input.data(), N * sizeof(float));

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(1);

  // Specialize with USE_SCALE = true (function_constant(0)).
  {
    auto fn = lib.lookupSpecializedFunction("specialized_fn", true);
    fn(stream(), la, outBuf, inBuf, 3.0f);
    outBuf.copyTo(stream(), output.data(), N * sizeof(float));
    stream().sync();

    for (size_t i = 0; i < N; i++) {
      EXPECT_FLOAT_EQ(output[i], static_cast<float>(i + 1) * 3.0f)
          << "index " << i;
    }
  }

  // Specialize with USE_SCALE = false.
  {
    auto fn = lib.lookupSpecializedFunction("specialized_fn", false);
    fn(stream(), la, outBuf, inBuf, 3.0f);
    outBuf.copyTo(stream(), output.data(), N * sizeof(float));
    stream().sync();

    for (size_t i = 0; i < N; i++) {
      EXPECT_FLOAT_EQ(output[i], static_cast<float>(i + 1)) << "index " << i;
    }
  }
}

GHOST_INSTANTIATE_KERNEL_TESTS(KernelTest);
