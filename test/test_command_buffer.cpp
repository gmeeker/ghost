#include "ghost_test.h"

using namespace ghost;
using namespace ghost::test;

class CommandBufferTest : public GhostKernelTest {};

// ---------------------------------------------------------------------------
// Basic command buffer operations
// ---------------------------------------------------------------------------

TEST_P(CommandBufferTest, CreateAndReset) {
  CommandBuffer cb(device());
  EXPECT_NO_THROW(cb.reset());
}

TEST_P(CommandBufferTest, EmptySubmit) {
  CommandBuffer cb(device());
  EXPECT_NO_THROW(cb.submit(stream()));
  EXPECT_NO_THROW(stream().sync());
}

// ---------------------------------------------------------------------------
// Buffer copy via command buffer
// ---------------------------------------------------------------------------

TEST_P(CommandBufferTest, CopyBuffer) {
  const size_t N = 16;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto src = device().allocateBuffer(N * sizeof(float));
  auto dst = device().allocateBuffer(N * sizeof(float));
  src.copy(stream(), input.data(), N * sizeof(float));
  stream().sync();

  CommandBuffer cb(device());
  dst.copy(cb, src, N * sizeof(float));
  cb.submit(stream());

  dst.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i)) << "index " << i;
  }
}

TEST_P(CommandBufferTest, CopyBufferWithOffsets) {
  const size_t N = 16;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto src = device().allocateBuffer(N * sizeof(float));
  auto dst = device().allocateBuffer(N * sizeof(float));

  src.copy(stream(), input.data(), N * sizeof(float));
  dst.copy(stream(), output.data(), N * sizeof(float));  // zero dst
  stream().sync();

  CommandBuffer cb(device());
  // Copy src[4..8) -> dst[8..12).
  dst.copy(cb, src, 4 * sizeof(float), 8 * sizeof(float), 4 * sizeof(float));
  cb.submit(stream());

  dst.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 8; i < 12; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i - 4)) << "index " << i;
  }
}

// ---------------------------------------------------------------------------
// Fill buffer via command buffer
// ---------------------------------------------------------------------------

TEST_P(CommandBufferTest, FillBuffer) {
  const size_t N = 64;
  auto buf = device().allocateBuffer(N);

  CommandBuffer cb(device());
  buf.fill(cb, 0, N, static_cast<uint8_t>(0xCC));
  cb.submit(stream());

  std::vector<uint8_t> output(N, 0);
  buf.copyTo(stream(), output.data(), N);
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_EQ(output[i], 0xCC) << "index " << i;
  }
}

// ---------------------------------------------------------------------------
// Dispatch via command buffer
// ---------------------------------------------------------------------------

TEST_P(CommandBufferTest, DispatchKernel) {
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
  stream().sync();

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(localSize);

  CommandBuffer cb(device());
  fn(la, cb)(outBuf, inBuf, 1.5f);
  cb.submit(stream());

  outBuf.copyTo(stream(), output.data(), safeN * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 1.5f) << "index " << i;
  }
  for (size_t i = N; i < safeN; i++) {
    if (output[i] != kSentinel) {
      FAIL() << "workgroup count bug: write at index " << i << " (got "
             << output[i] << ", expected sentinel " << kSentinel << ")";
    }
  }
}

// ---------------------------------------------------------------------------
// Barrier
// ---------------------------------------------------------------------------

TEST_P(CommandBufferTest, BarrierOrdering) {
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
  stream().sync();

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(localSize);

  CommandBuffer cb(device());
  fn(la, cb)(buf2, buf1, 2.0f);
  cb.barrier();
  fn(la, cb)(buf3, buf2, 3.0f);
  cb.submit(stream());

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
// Reset and reuse
// ---------------------------------------------------------------------------

TEST_P(CommandBufferTest, ResetAndReuse) {
  const size_t N = 16;
  std::vector<float> data1(N, 1.0f), data2(N, 2.0f), output(N, 0.0f);

  auto buf = device().allocateBuffer(N * sizeof(float));

  CommandBuffer cb(device());

  // First use: fill with data1.
  auto src1 = device().allocateBuffer(N * sizeof(float));
  src1.copy(stream(), data1.data(), N * sizeof(float));
  stream().sync();
  buf.copy(cb, src1, N * sizeof(float));
  cb.submit(stream());
  buf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();
  for (size_t i = 0; i < N; i++) EXPECT_FLOAT_EQ(output[i], 1.0f);

  // Reset and second use: fill with data2.
  cb.reset();
  auto src2 = device().allocateBuffer(N * sizeof(float));
  src2.copy(stream(), data2.data(), N * sizeof(float));
  stream().sync();
  buf.copy(cb, src2, N * sizeof(float));
  cb.submit(stream());
  buf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();
  for (size_t i = 0; i < N; i++) EXPECT_FLOAT_EQ(output[i], 2.0f);
}

// ---------------------------------------------------------------------------
// Command buffer with event
// ---------------------------------------------------------------------------

TEST_P(CommandBufferTest, SubmitWithEvent) {
  const size_t N = 16;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto src = device().allocateBuffer(N * sizeof(float));
  auto dst = device().allocateBuffer(N * sizeof(float));
  src.copy(stream(), input.data(), N * sizeof(float));
  stream().sync();

  CommandBuffer cb(device());
  dst.copy(cb, src, N * sizeof(float));

  Event event(nullptr);
  try {
    event = cb.recordEvent();
  } catch (const ghost::unsupported_error&) {
    GTEST_SKIP() << "Command buffer events not supported";
  }

  cb.submit(stream());

  try {
    event.wait();
    EXPECT_TRUE(event.isComplete());
  } catch (const ghost::unsupported_error&) {
    // Fallback — just sync the stream.
    stream().sync();
  }

  dst.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i)) << "index " << i;
  }
}

// ---------------------------------------------------------------------------
// Command buffer with many operations
// ---------------------------------------------------------------------------

TEST_P(CommandBufferTest, ManyOperations) {
  const size_t N = 64;
  auto buf = device().allocateBuffer(N);

  CommandBuffer cb(device());
  // Record 100 fill operations.
  for (int i = 0; i < 100; i++) {
    buf.fill(cb, 0, N, static_cast<uint8_t>(i));
  }
  cb.submit(stream());

  std::vector<uint8_t> output(N, 0);
  buf.copyTo(stream(), output.data(), N);
  stream().sync();

  // Last fill wins: value should be 99.
  for (size_t i = 0; i < N; i++) {
    EXPECT_EQ(output[i], 99) << "index " << i;
  }
}

// ---------------------------------------------------------------------------
// Indirect dispatch via command buffer
// ---------------------------------------------------------------------------

TEST_P(CommandBufferTest, DispatchIndirect) {
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
  stream().sync();

  // Write workgroup counts (N, 1, 1) into an indirect buffer.
  // With local_size=1, workgroup count equals total thread count.
  uint32_t counts[3] = {static_cast<uint32_t>(N), 1, 1};
  auto indirectBuf = device().allocateBuffer(sizeof(counts));
  indirectBuf.copy(stream(), counts, sizeof(counts));
  stream().sync();

  CommandBuffer cb(device());
  cb.dispatchIndirect(fn, indirectBuf, 0, outBuf, inBuf, 1.5f);
  cb.submit(stream());

  outBuf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 1.5f) << "index " << i;
  }
}

TEST_P(CommandBufferTest, DispatchIndirectWithOffset) {
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
  stream().sync();

  // Place a dummy entry before the real counts at offset 16.
  uint32_t data[6] = {0, 0, 0, 0, static_cast<uint32_t>(N), 1};
  // counts at offset 12: {N, 1, 1} — but we need 4-byte alignment.
  // Use offset 12 for the 3 uint32_t values.
  uint32_t fullData[6] = {99, 99, 99, static_cast<uint32_t>(N), 1, 1};
  auto indirectBuf = device().allocateBuffer(sizeof(fullData));
  indirectBuf.copy(stream(), fullData, sizeof(fullData));
  stream().sync();

  CommandBuffer cb(device());
  cb.dispatchIndirect(fn, indirectBuf, 3 * sizeof(uint32_t), outBuf, inBuf,
                      2.0f);
  cb.submit(stream());

  outBuf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 2.0f) << "index " << i;
  }
}

// ---------------------------------------------------------------------------
// Lifetime: Buffer wrappers may go out of scope between record() and submit()
// ---------------------------------------------------------------------------

// Regression test for the Attribute lifetime bug: when a Buffer wrapper passed
// to dispatch() is destroyed before submit(), the recorded dispatch must
// still execute correctly because the Attribute carries a strong reference
// to the underlying buffer impl.
TEST_P(CommandBufferTest, BufferWrappersOutOfScope) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  const size_t N = 32;
  std::vector<float> input(N);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");

  // Allocate the output buffer in the outer scope so we can read it back.
  auto outBuf = device().allocateBuffer(N * sizeof(float));

  CommandBuffer cb(device());

  // Record dispatches whose input Buffer wrappers go out of scope before
  // submit. The Attribute stored in the recorded command must keep the
  // underlying impl alive.
  for (int iter = 0; iter < 4; iter++) {
    auto inBuf = device().allocateBuffer(N * sizeof(float));
    inBuf.copy(stream(), input.data(), N * sizeof(float));
    stream().sync();

    LaunchArgs la;
    la.global_size(N).local_size(1);
    fn(la, cb)(outBuf, inBuf, static_cast<float>(iter + 1));
    // inBuf wrapper destroyed at end of this iteration.
  }

  cb.submit(stream());
  stream().sync();

  // The last recorded dispatch (iter=3, scale=4) wins.
  std::vector<float> output(N, 0.0f);
  outBuf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();
  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 4.0f) << "index " << i;
  }
}

// Regression test for the second bug Inferency hit: storing Buffer wrappers
// in a vector that reallocates mid-batch must not invalidate the recorded
// dispatches.
TEST_P(CommandBufferTest, BufferWrappersInReallocatingVector) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  const size_t N = 16;
  std::vector<float> input(N);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");
  auto outBuf = device().allocateBuffer(N * sizeof(float));

  CommandBuffer cb(device());
  std::vector<Buffer> wrappers;  // intentionally not reserved
  wrappers.reserve(1);           // force reallocation as we push more

  for (int iter = 0; iter < 8; iter++) {
    Buffer inBuf = device().allocateBuffer(N * sizeof(float));
    inBuf.copy(stream(), input.data(), N * sizeof(float));
    stream().sync();
    wrappers.push_back(inBuf);  // push_back may reallocate

    LaunchArgs la;
    la.global_size(N).local_size(1);
    fn(la, cb)(outBuf, wrappers.back(), 2.0f);
  }

  cb.submit(stream());
  stream().sync();

  std::vector<float> output(N, 0.0f);
  outBuf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();
  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 2.0f) << "index " << i;
  }
}

GHOST_INSTANTIATE_KERNEL_TESTS(CommandBufferTest);
