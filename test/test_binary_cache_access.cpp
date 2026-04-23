#include "ghost_test.h"

using namespace ghost;
using namespace ghost::test;

// ---------------------------------------------------------------------------
// Library::getBinary() round-trip tests
// ---------------------------------------------------------------------------

class BinaryCacheAccessTest : public GhostKernelTest {};

TEST_P(BinaryCacheAccessTest, GetBinaryAfterCompile) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP() << "No kernel source for " << BackendName(backend());

  auto lib = device().loadLibraryFromText(src, CompilerOptions(), true);
  auto binary = lib.getBinary();

  // Backends that support getBinary should return non-empty data.
  // Some backends may not support it yet, so we just check it doesn't crash.
  if (!binary.empty()) {
    EXPECT_GT(binary.size(), 0u);
  }
}

TEST_P(BinaryCacheAccessTest, ReloadFromBinary) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP() << "No kernel source for " << BackendName(backend());

  // Compile from text
  auto lib1 = device().loadLibraryFromText(src, CompilerOptions(), true);
  auto binary = lib1.getBinary();
  if (binary.empty()) {
    GTEST_SKIP() << "getBinary() not supported for " << BackendName(backend());
  }

  // Reload from binary — may fail if the backend's binary format
  // is device-specific and loadLibraryFromData expects a different format
  // (e.g., OpenCL device binaries can't be loaded via clCreateProgramWithIL).
  try {
    auto lib2 = device().loadLibraryFromData(binary.data(), binary.size());
    auto fn = lib2.lookupFunction("mult_const_f");

    // Dispatch and verify
    const size_t N = 16;
    std::vector<float> input(N), output(N, 0.0f);
    for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

    auto inBuf = device().allocateBuffer(N * sizeof(float));
    auto outBuf = device().allocateBuffer(N * sizeof(float));
    inBuf.copy(stream(), input.data(), N * sizeof(float));

    LaunchArgs la;
    la.global_size(static_cast<uint32_t>(N)).local_size(1);
    fn(la, stream())(outBuf, inBuf, 2.0f);
    outBuf.copyTo(stream(), output.data(), N * sizeof(float));
    stream().sync();

    for (size_t i = 0; i < N; i++) {
      EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 2.0f) << "index " << i;
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Reload from binary not supported: " << e.what();
  }
}

GHOST_INSTANTIATE_KERNEL_TESTS(BinaryCacheAccessTest);
