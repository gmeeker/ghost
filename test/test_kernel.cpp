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

GHOST_INSTANTIATE_KERNEL_TESTS(KernelTest);
