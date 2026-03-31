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
  cb.copyBuffer(dst, src, N * sizeof(float));
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
  cb.copyBuffer(dst, 8 * sizeof(float), src, 4 * sizeof(float),
                4 * sizeof(float));
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
  cb.fillBuffer(buf, 0, N, 0xCC);
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

  const size_t N = 32;
  std::vector<float> input(N), output(N, 0.0f);
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
  cb.dispatch(fn, la, outBuf, inBuf, 1.5f);
  cb.submit(stream());

  outBuf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 1.5f) << "index " << i;
  }
}

// ---------------------------------------------------------------------------
// Barrier
// ---------------------------------------------------------------------------

TEST_P(CommandBufferTest, BarrierOrdering) {
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
  stream().sync();

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(1);

  CommandBuffer cb(device());
  cb.dispatch(fn, la, buf2, buf1, 2.0f);
  cb.barrier();
  cb.dispatch(fn, la, buf3, buf2, 3.0f);
  cb.submit(stream());

  buf3.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 6.0f) << "index " << i;
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
  cb.copyBuffer(buf, src1, N * sizeof(float));
  cb.submit(stream());
  buf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();
  for (size_t i = 0; i < N; i++) EXPECT_FLOAT_EQ(output[i], 1.0f);

  // Reset and second use: fill with data2.
  cb.reset();
  auto src2 = device().allocateBuffer(N * sizeof(float));
  src2.copy(stream(), data2.data(), N * sizeof(float));
  stream().sync();
  cb.copyBuffer(buf, src2, N * sizeof(float));
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
  cb.copyBuffer(dst, src, N * sizeof(float));

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
    cb.fillBuffer(buf, 0, N, static_cast<uint8_t>(i));
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

GHOST_INSTANTIATE_KERNEL_TESTS(CommandBufferTest);
