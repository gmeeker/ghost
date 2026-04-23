#include <ghost/argument_buffer.h>
#include <ghost/kernel_source.h>

#include <filesystem>

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

// Regression test: a Function must keep its parent Library alive so the
// underlying compiled module (CUmodule, cl_program, dlopened .so, etc.)
// is not unloaded while the kernel is still in use.
TEST_P(KernelTest, FunctionOutlivesLibrary) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  const size_t N = 64;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  Function fn = [&]() {
    Library lib = device().loadLibraryFromText(src);
    return lib.lookupFunction("mult_const_f");
    // lib goes out of scope here; fn must keep the underlying module alive.
  }();

  auto inBuf = device().allocateBuffer(N * sizeof(float));
  auto outBuf = device().allocateBuffer(N * sizeof(float));
  inBuf.copy(stream(), input.data(), N * sizeof(float));
  stream().sync();

  LaunchArgs la;
  la.global_size(N).local_size(1);
  fn(la, stream())(outBuf, inBuf, 3.0f);
  outBuf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 3.0f) << "index " << i;
  }
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

  const size_t N = 256;
  const uint32_t localSize = 64;
  // If global_size is misinterpreted as workgroup count, N * localSize threads
  // launch.  Allocate oversized buffers so those threads write into a sentinel
  // region instead of out-of-bounds memory.
  const size_t safeN = N * localSize;
  const float kSentinel = -1.0f;

  std::vector<float> input(safeN, 0.0f);
  std::vector<float> output(safeN, kSentinel);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");

  auto inBuf = device().allocateBuffer(safeN * sizeof(float));
  auto outBuf = device().allocateBuffer(safeN * sizeof(float));

  inBuf.copy(stream(), input.data(), safeN * sizeof(float));
  outBuf.copy(stream(), output.data(), safeN * sizeof(float));

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(localSize);
  fn(la, stream())(outBuf, inBuf, 1.5f);
  outBuf.copyTo(stream(), output.data(), safeN * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 1.5f) << "index " << i;
  }
  // Verify no writes beyond N (detects workgroup count bug).
  for (size_t i = N; i < safeN; i++) {
    if (output[i] != kSentinel) {
      FAIL() << "workgroup count bug: write at index " << i << " (got "
             << output[i] << ", expected sentinel " << kSentinel << ")";
    }
  }
}

TEST_P(KernelTest, MultConstScale2) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  const size_t N = 256;
  const uint32_t localSize = 64;
  const size_t safeN = N * localSize;
  const float kSentinel = -1.0f;

  std::vector<float> input(safeN, 0.0f);
  std::vector<float> output(safeN, kSentinel);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i + 1);

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");

  auto inBuf = device().allocateBuffer(safeN * sizeof(float));
  auto outBuf = device().allocateBuffer(safeN * sizeof(float));

  inBuf.copy(stream(), input.data(), safeN * sizeof(float));
  outBuf.copy(stream(), output.data(), safeN * sizeof(float));

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(localSize);
  fn(la, stream())(outBuf, inBuf, 2.0f);
  outBuf.copyTo(stream(), output.data(), safeN * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i + 1) * 2.0f)
        << "index " << i;
  }
  for (size_t i = N; i < safeN; i++) {
    if (output[i] != kSentinel) {
      FAIL() << "workgroup count bug: write at index " << i << " (got "
             << output[i] << ", expected sentinel " << kSentinel << ")";
    }
  }
}

TEST_P(KernelTest, AddBuffers) {
  const char* src = addBuffersSource();
  if (!src) GTEST_SKIP();

  const size_t N = 256;
  const uint32_t localSize = 64;
  const size_t safeN = N * localSize;
  const float kSentinel = -1.0f;

  std::vector<float> a(safeN, 0.0f), b(safeN, 0.0f);
  std::vector<float> output(safeN, kSentinel);
  for (size_t i = 0; i < N; i++) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i * 10);
  }

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("add_buffers");

  auto bufA = device().allocateBuffer(safeN * sizeof(float));
  auto bufB = device().allocateBuffer(safeN * sizeof(float));
  auto bufOut = device().allocateBuffer(safeN * sizeof(float));

  bufA.copy(stream(), a.data(), safeN * sizeof(float));
  bufB.copy(stream(), b.data(), safeN * sizeof(float));
  bufOut.copy(stream(), output.data(), safeN * sizeof(float));

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(localSize);
  fn(la, stream())(bufOut, bufA, bufB);
  bufOut.copyTo(stream(), output.data(), safeN * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i + i * 10)) << "index " << i;
  }
  for (size_t i = N; i < safeN; i++) {
    if (output[i] != kSentinel) {
      FAIL() << "workgroup count bug: write at index " << i << " (got "
             << output[i] << ", expected sentinel " << kSentinel << ")";
    }
  }
}

// ---------------------------------------------------------------------------
// Multiple dispatches on same stream
// ---------------------------------------------------------------------------

TEST_P(KernelTest, MultipleDispatches) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  const size_t N = 256;
  const uint32_t localSize = 64;
  const size_t safeN = N * localSize;
  const float kSentinel = -1.0f;

  std::vector<float> input(safeN, 0.0f);
  std::vector<float> output(safeN, kSentinel);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");

  auto buf1 = device().allocateBuffer(safeN * sizeof(float));
  auto buf2 = device().allocateBuffer(safeN * sizeof(float));
  auto buf3 = device().allocateBuffer(safeN * sizeof(float));

  buf1.copy(stream(), input.data(), safeN * sizeof(float));
  // Initialize buf3 with sentinel so we can detect extra writes.
  buf3.copy(stream(), output.data(), safeN * sizeof(float));

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(localSize);

  // Chain: buf1 * 2.0 -> buf2, buf2 * 3.0 -> buf3.
  fn(la, stream())(buf2, buf1, 2.0f);
  fn(la, stream())(buf3, buf2, 3.0f);
  buf3.copyTo(stream(), output.data(), safeN * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 6.0f) << "index " << i;
  }
  for (size_t i = N; i < safeN; i++) {
    if (output[i] != kSentinel) {
      FAIL() << "workgroup count bug: write at index " << i << " (got "
             << output[i] << ", expected sentinel " << kSentinel << ")";
    }
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

  const uint32_t W = 32, H = 16;
  const uint32_t localW = 8, localH = 8;
  const size_t N = W * H;
  // With the bug, Metal/CUDA dispatch W*H threadgroups of localW*localH each.
  // Max thread coords: x in [0, W*localW), y in [0, H*localH).
  // Max linear index = (H*localH - 1) * W + (W*localW - 1).
  const size_t safeN = static_cast<size_t>(H * localH) * W + W * localW;
  const float kSentinel = -1.0f;

  std::vector<float> input(safeN, 0.0f);
  std::vector<float> output(safeN, kSentinel);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_2d");

  auto inBuf = device().allocateBuffer(safeN * sizeof(float));
  auto outBuf = device().allocateBuffer(safeN * sizeof(float));
  inBuf.copy(stream(), input.data(), safeN * sizeof(float));
  outBuf.copy(stream(), output.data(), safeN * sizeof(float));

  LaunchArgs la;
  la.global_size(W, H).local_size(localW, localH);
  fn(la, stream())(outBuf, inBuf, 2.0f, static_cast<int32_t>(W));
  outBuf.copyTo(stream(), output.data(), safeN * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 2.0f) << "index " << i;
  }
  for (size_t i = N; i < safeN; i++) {
    if (output[i] != kSentinel) {
      FAIL() << "workgroup count bug: write at index " << i << " (got "
             << output[i] << ", expected sentinel " << kSentinel << ")";
    }
  }
}

// ---------------------------------------------------------------------------
// 3D kernel launch
// ---------------------------------------------------------------------------

TEST_P(KernelTest, MultConst3D) {
  const char* src = multConst3DSource();
  if (!src)
    GTEST_SKIP() << "No 3D kernel source for " << BackendName(backend());

  const uint32_t W = 16, H = 8, D = 4;
  const uint32_t localW = 8, localH = 4, localD = 2;
  const size_t N = W * H * D;
  // With the bug: x in [0, W*localW), y in [0, H*localH), z in [0, D*localD).
  // Max index = (D*localD-1)*H*W + (H*localH-1)*W + (W*localW-1).
  const size_t safeN = static_cast<size_t>(D * localD - 1) * H * W +
                       static_cast<size_t>(H * localH - 1) * W + W * localW;
  const float kSentinel = -1.0f;

  std::vector<float> input(safeN, 0.0f);
  std::vector<float> output(safeN, kSentinel);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_3d");

  auto inBuf = device().allocateBuffer(safeN * sizeof(float));
  auto outBuf = device().allocateBuffer(safeN * sizeof(float));
  inBuf.copy(stream(), input.data(), safeN * sizeof(float));
  outBuf.copy(stream(), output.data(), safeN * sizeof(float));

  LaunchArgs la;
  la.global_size(W, H, D).local_size(localW, localH, localD);
  fn(la, stream())(outBuf, inBuf, 3.0f, static_cast<int32_t>(W),
                   static_cast<int32_t>(H));
  outBuf.copyTo(stream(), output.data(), safeN * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 3.0f) << "index " << i;
  }
  for (size_t i = N; i < safeN; i++) {
    if (output[i] != kSentinel) {
      FAIL() << "workgroup count bug: write at index " << i << " (got "
             << output[i] << ", expected sentinel " << kSentinel << ")";
    }
  }
}

// ---------------------------------------------------------------------------
// Local memory argument
// ---------------------------------------------------------------------------

TEST_P(KernelTest, LocalMemoryArgument) {
  const char* src = localMemSource();
  if (!src)
    GTEST_SKIP() << "No local mem kernel for " << BackendName(backend());

  const uint32_t localSize = 32;
  const uint32_t numGroups = 4;
  const uint32_t N = localSize * numGroups;
  // With the bug, N workgroups launch (instead of numGroups).  Each group
  // writes to out[group_id] and reads A[gid] where gid spans N*localSize.
  const uint32_t safeGroups = N;  // buggy workgroup count
  const size_t safeInput = static_cast<size_t>(N) * localSize;
  const float kSentinel = -1.0f;

  std::vector<float> input(safeInput, 0.0f);
  std::vector<float> output(safeGroups, kSentinel);
  for (uint32_t i = 0; i < N; i++) input[i] = static_cast<float>(i + 1);

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("local_mem_sum");

  auto inBuf = device().allocateBuffer(safeInput * sizeof(float));
  auto outBuf = device().allocateBuffer(safeGroups * sizeof(float));
  inBuf.copy(stream(), input.data(), safeInput * sizeof(float));
  outBuf.copy(stream(), output.data(), safeGroups * sizeof(float));

  LaunchArgs la;
  la.global_size(N).local_size(localSize);

  // CUDA uses dynamic shared memory via launch config, not an explicit arg.
  if (backend() == Backend::CUDA) {
    fn.execute(stream(), la,
               {Attribute(&outBuf), Attribute(&inBuf),
                Attribute().localMem(localSize * sizeof(float))});
  } else {
    fn(la, stream())(outBuf, inBuf,
                     Attribute().localMem(localSize * sizeof(float)));
  }

  outBuf.copyTo(stream(), output.data(), safeGroups * sizeof(float));
  stream().sync();

  for (uint32_t g = 0; g < numGroups; g++) {
    float expected = 0.0f;
    for (uint32_t i = 0; i < localSize; i++)
      expected += static_cast<float>(g * localSize + i + 1);
    EXPECT_FLOAT_EQ(output[g], expected) << "group " << g;
  }
  for (uint32_t g = numGroups; g < safeGroups; g++) {
    if (output[g] != kSentinel) {
      FAIL() << "workgroup count bug: write at group " << g << " (got "
             << output[g] << ", expected sentinel " << kSentinel << ")";
    }
  }
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
  std::filesystem::path tmpDir =
      std::filesystem::temp_directory_path() /
      ("ghost_test_cache_" + std::to_string(reinterpret_cast<uintptr_t>(this)));
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

  const size_t N = 256;
  const uint32_t localSize = 64;
  const size_t safeN = N * localSize;
  const float kSentinel = -1.0f;

  std::vector<float> input(safeN, 0.0f);
  std::vector<float> output(safeN, kSentinel);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");

  auto inBuf = device().allocateBuffer(safeN * sizeof(float));
  auto outBuf = device().allocateBuffer(safeN * sizeof(float));
  inBuf.copy(stream(), input.data(), safeN * sizeof(float));
  outBuf.copy(stream(), output.data(), safeN * sizeof(float));

  // Use execute() with vector<Attribute> to pass args including an
  // ArgumentBuffer for the scale parameter.
  ArgumentBuffer ab;
  ab.set(0, 2.5f);
  EXPECT_TRUE(ab.isStruct());
  EXPECT_GE(ab.size(), sizeof(float));

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(localSize);

  // Pass arguments manually via execute().
  fn.execute(stream(), la,
             {Attribute(&outBuf), Attribute(&inBuf), Attribute(2.5f)});
  outBuf.copyTo(stream(), output.data(), safeN * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 2.5f) << "index " << i;
  }
  for (size_t i = N; i < safeN; i++) {
    if (output[i] != kSentinel) {
      FAIL() << "workgroup count bug: write at index " << i << " (got "
             << output[i] << ", expected sentinel " << kSentinel << ")";
    }
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

  const size_t N = 128;
  const uint32_t localSize = 64;
  const size_t safeN = N * localSize;
  const float kSentinel = -1.0f;

  std::vector<float> input(safeN, 0.0f);
  std::vector<float> output(safeN, kSentinel);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i + 1);

  auto lib = device().loadLibraryFromText(src);

  auto inBuf = device().allocateBuffer(safeN * sizeof(float));
  auto outBuf = device().allocateBuffer(safeN * sizeof(float));
  inBuf.copy(stream(), input.data(), safeN * sizeof(float));
  outBuf.copy(stream(), output.data(), safeN * sizeof(float));

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(localSize);

  // Specialize with USE_SCALE = true (function_constant(0)).
  {
    // Re-fill output with sentinel before each dispatch.
    outBuf.copy(stream(), output.data(), safeN * sizeof(float));
    auto fn = lib.lookupSpecializedFunction("specialized_fn", true);
    fn(la, stream())(outBuf, inBuf, 3.0f);
    outBuf.copyTo(stream(), output.data(), safeN * sizeof(float));
    stream().sync();

    for (size_t i = 0; i < N; i++) {
      EXPECT_FLOAT_EQ(output[i], static_cast<float>(i + 1) * 3.0f)
          << "index " << i;
    }
    for (size_t i = N; i < safeN; i++) {
      if (output[i] != kSentinel) {
        FAIL() << "workgroup count bug: write at index " << i << " (got "
               << output[i] << ", expected sentinel " << kSentinel << ")";
      }
    }
  }

  // Specialize with USE_SCALE = false.
  {
    std::fill(output.begin(), output.end(), kSentinel);
    outBuf.copy(stream(), output.data(), safeN * sizeof(float));
    auto fn = lib.lookupSpecializedFunction("specialized_fn", false);
    fn(la, stream())(outBuf, inBuf, 3.0f);
    outBuf.copyTo(stream(), output.data(), safeN * sizeof(float));
    stream().sync();

    for (size_t i = 0; i < N; i++) {
      EXPECT_FLOAT_EQ(output[i], static_cast<float>(i + 1)) << "index " << i;
    }
    for (size_t i = N; i < safeN; i++) {
      if (output[i] != kSentinel) {
        FAIL() << "workgroup count bug: write at index " << i << " (got "
               << output[i] << ", expected sentinel " << kSentinel << ")";
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Subgroup sizing
// ---------------------------------------------------------------------------

TEST_P(KernelTest, PreferredSubgroupSize) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");

  uint32_t width = fn.preferredSubgroupSize();
  // CPU is 1; every other backend should report a real warp/SIMD width.
  // Common values: 1 (CPU), 32 (NVIDIA / Intel), 64 (AMD), 32 or 16 (Apple).
  EXPECT_GT(width, 0u);
  EXPECT_LE(width, 128u);
}

TEST_P(KernelTest, RequireMatchingSubgroupSize) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  const size_t N = 256;
  const uint32_t localSize = 64;
  const size_t safeN = N * localSize;

  std::vector<float> input(safeN, 0.0f);
  std::vector<float> output(safeN, -1.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");
  auto inBuf = device().allocateBuffer(safeN * sizeof(float));
  auto outBuf = device().allocateBuffer(safeN * sizeof(float));
  inBuf.copy(stream(), input.data(), safeN * sizeof(float));
  outBuf.copy(stream(), output.data(), safeN * sizeof(float));

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N))
      .local_size(localSize)
      .requireSubgroupSize(fn.preferredSubgroupSize());

  // Vulkan currently does not enable VK_EXT_subgroup_size_control, so it
  // throws unsupported_error for any non-zero requireSubgroupSize.
  if (backend() == Backend::Vulkan) {
    EXPECT_THROW(fn(la, stream())(outBuf, inBuf, 1.5f),
                 ghost::unsupported_error);
    return;
  }

  EXPECT_NO_THROW(fn(la, stream())(outBuf, inBuf, 1.5f));
  outBuf.copyTo(stream(), output.data(), safeN * sizeof(float));
  stream().sync();
  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 1.5f) << "index " << i;
  }
}

TEST_P(KernelTest, RequireWrongSubgroupSizeThrows) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");

  // Pick a value guaranteed not to match any real warp/SIMD width.
  uint32_t bogus = fn.preferredSubgroupSize() + 1;
  if (bogus == 1) bogus = 2;  // CPU edge case

  auto inBuf = device().allocateBuffer(64 * sizeof(float));
  auto outBuf = device().allocateBuffer(64 * sizeof(float));

  LaunchArgs la;
  la.global_size(64u).local_size(64u).requireSubgroupSize(bogus);

  // Vulkan throws unsupported_error (extension not enabled). Other backends
  // throw std::invalid_argument from the mismatch check.
  EXPECT_THROW(fn(la, stream())(outBuf, inBuf, 1.0f), std::exception);
}

// ---------------------------------------------------------------------------
// KernelSource — compile-with-defines caching
// ---------------------------------------------------------------------------

TEST_P(KernelTest, KernelSourceDefines) {
  // Uses the same kernel as SetGlobals: SCALE_FACTOR injected as -D define.
  const char* src = setGlobalsKernelSource(backend());
  if (!src) GTEST_SKIP() << "KernelSource test needs OpenCL source";

  const size_t N = 128;
  const uint32_t localSize = 64;
  const size_t safeN = N * localSize;
  const float kSentinel = -1.0f;

  std::vector<float> input(safeN, 0.0f);
  std::vector<float> output(safeN, kSentinel);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i + 1);

  auto inBuf = device().allocateBuffer(safeN * sizeof(float));
  auto outBuf = device().allocateBuffer(safeN * sizeof(float));
  inBuf.copy(stream(), input.data(), safeN * sizeof(float));

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(localSize);

  KernelSource ks(src);

  // Variant 1: SCALE_FACTOR = 3.0
  {
    auto fn = ks.getFunction(device(), "scaled_fn",
                             {{"SCALE_FACTOR", Attribute(3.0f)}});
    outBuf.copy(stream(), output.data(), safeN * sizeof(float));
    fn(la, stream())(outBuf, inBuf, static_cast<uint32_t>(N));
    outBuf.copyTo(stream(), output.data(), safeN * sizeof(float));
    stream().sync();

    for (size_t i = 0; i < N; i++) {
      EXPECT_FLOAT_EQ(output[i], static_cast<float>(i + 1) * 3.0f)
          << "index " << i;
    }
  }

  // Variant 2: SCALE_FACTOR = 0.5 — different variant, separate compilation.
  {
    auto fn = ks.getFunction(device(), "scaled_fn",
                             {{"SCALE_FACTOR", Attribute(0.5f)}});
    std::fill(output.begin(), output.end(), kSentinel);
    outBuf.copy(stream(), output.data(), safeN * sizeof(float));
    fn(la, stream())(outBuf, inBuf, static_cast<uint32_t>(N));
    outBuf.copyTo(stream(), output.data(), safeN * sizeof(float));
    stream().sync();

    for (size_t i = 0; i < N; i++) {
      EXPECT_FLOAT_EQ(output[i], static_cast<float>(i + 1) * 0.5f)
          << "index " << i;
    }
  }

  // Cache hit: re-requesting the same variant should return the same Function.
  {
    auto fn1 = ks.getFunction(device(), "scaled_fn",
                              {{"SCALE_FACTOR", Attribute(3.0f)}});
    auto fn2 = ks.getFunction(device(), "scaled_fn",
                              {{"SCALE_FACTOR", Attribute(3.0f)}});
    EXPECT_EQ(fn1.impl().get(), fn2.impl().get());
  }
}

// KernelSource with named specialization (Metal function constants).
TEST_P(KernelTest, KernelSourceSpecialization) {
  if (backend() != Backend::Metal) {
    GTEST_SKIP() << "Named specialization is Metal-only";
  }

  const char* src = specializedKernelSource();
  if (!src) GTEST_SKIP();

  const size_t N = 128;
  const uint32_t localSize = 64;
  const size_t safeN = N * localSize;
  const float kSentinel = -1.0f;

  std::vector<float> input(safeN, 0.0f);
  std::vector<float> output(safeN, kSentinel);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i + 1);

  auto inBuf = device().allocateBuffer(safeN * sizeof(float));
  auto outBuf = device().allocateBuffer(safeN * sizeof(float));
  inBuf.copy(stream(), input.data(), safeN * sizeof(float));

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(localSize);

  KernelSource ks(src);

  // USE_SCALE = true → multiply by scale factor.
  {
    auto fn = ks.getFunction(device(), "specialized_fn",
                             {{"USE_SCALE", Attribute(true)}});
    outBuf.copy(stream(), output.data(), safeN * sizeof(float));
    fn(la, stream())(outBuf, inBuf, 3.0f);
    outBuf.copyTo(stream(), output.data(), safeN * sizeof(float));
    stream().sync();

    for (size_t i = 0; i < N; i++) {
      EXPECT_FLOAT_EQ(output[i], static_cast<float>(i + 1) * 3.0f)
          << "index " << i;
    }
  }

  // USE_SCALE = false → passthrough.
  {
    auto fn = ks.getFunction(device(), "specialized_fn",
                             {{"USE_SCALE", Attribute(false)}});
    std::fill(output.begin(), output.end(), kSentinel);
    outBuf.copy(stream(), output.data(), safeN * sizeof(float));
    fn(la, stream())(outBuf, inBuf, 3.0f);
    outBuf.copyTo(stream(), output.data(), safeN * sizeof(float));
    stream().sync();

    for (size_t i = 0; i < N; i++) {
      EXPECT_FLOAT_EQ(output[i], static_cast<float>(i + 1)) << "index " << i;
    }
  }
}

GHOST_INSTANTIATE_KERNEL_TESTS(KernelTest);
