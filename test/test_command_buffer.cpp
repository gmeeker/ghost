#include <atomic>
#include <memory>

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
    // Each iteration writes to the same outBuf; insert a barrier so the
    // "last wins" check below isn't racing with the prior iteration.
    // (CommandBuffer defaults to concurrent dispatches.)
    cb.barrier();
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

// Regression test: storing Buffer wrappers
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

// ---------------------------------------------------------------------------
// Concurrency policy: chained dispatches with default vs concurrent(false)
// ---------------------------------------------------------------------------

// Chained dependent dispatches without an explicit barrier. With the default
// CommandBufferOptions (concurrent=true), backends do NOT insert a
// per-dispatch barrier and the second dispatch may race with the first's
// writes. With concurrent(false) the backend restores the auto-barrier and
// the chain runs correctly without explicit cb.barrier() calls.
TEST_P(CommandBufferTest, AutoBarrierOptInChainedDispatches) {
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

  auto buf1 = device().allocateBuffer(safeN * sizeof(float));
  auto buf2 = device().allocateBuffer(safeN * sizeof(float));
  auto buf3 = device().allocateBuffer(safeN * sizeof(float));
  buf1.copy(stream(), input.data(), safeN * sizeof(float));
  stream().sync();

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(localSize);

  // Opt into auto-barriers so chained dispatches without cb.barrier() work.
  CommandBuffer cb(device(), CommandBufferOptions{/*concurrent=*/false});
  fn(la, cb)(buf2, buf1, 2.0f);
  fn(la, cb)(buf3, buf2, 3.0f);  // depends on buf2 written above
  cb.submit(stream());

  buf3.copyTo(stream(), output.data(), safeN * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 6.0f) << "index " << i;
  }
}

// Symmetric test of StreamOptions::concurrent: verify the option plumbs
// through without breaking simple independent dispatches. (Correctness of
// the concurrent semantics for dependent dispatches is the caller's
// responsibility; this just exercises the option path.)
TEST_P(CommandBufferTest, StreamConcurrentOptionIndependentDispatch) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  const size_t N = 256;
  const uint32_t localSize = 64;

  std::vector<float> input(N, 0.0f);
  std::vector<float> output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");

  auto inBuf = device().allocateBuffer(N * sizeof(float));
  auto outBuf = device().allocateBuffer(N * sizeof(float));

  // Stream with concurrent=true skips the per-dispatch auto-barrier. Use
  // independent buffers (single dispatch) so the test result is
  // deterministic regardless.
  Stream s = device().createStream(StreamOptions{/*profiling=*/false,
                                                 /*forceEventChain=*/false,
                                                 /*concurrent=*/true});
  inBuf.copy(s, input.data(), N * sizeof(float));

  LaunchArgs la;
  la.global_size(static_cast<uint32_t>(N)).local_size(localSize);
  fn(la, s)(outBuf, inBuf, 1.5f);

  outBuf.copyTo(s, output.data(), N * sizeof(float));
  s.sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 1.5f) << "index " << i;
  }
}

// ---------------------------------------------------------------------------
// Fire-and-forget buffer release: caller drops the Buffer wrapper between
// dispatch and stream sync. On CUDA, the BufferCUDA destructor must defer
// cuMemFree until the stream has advanced past the pending kernel; on
// OpenCL, the runtime's implicit cl_mem retention covers this. Without the
// fix, CUDA crashes with CUDA_ERROR_ILLEGAL_ADDRESS at the next driver call.
// ---------------------------------------------------------------------------

TEST_P(CommandBufferTest, DropBufferBeforeStreamSync) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  const size_t N = 256;
  const uint32_t localSize = 64;

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");

  std::vector<float> output(N, 0.0f);
  auto outBuf = device().allocateBuffer(N * sizeof(float));

  {
    std::vector<float> input(N);
    for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);
    auto inBuf = device().allocateBuffer(N * sizeof(float));
    inBuf.copy(stream(), input.data(), N * sizeof(float));

    LaunchArgs la;
    la.global_size(static_cast<uint32_t>(N)).local_size(localSize);
    fn(la, stream())(outBuf, inBuf, 2.0f);
    // inBuf drops here, before stream().sync(). The kernel above is still
    // pending on the stream; the underlying device allocation must survive
    // until the GPU finishes reading it.
  }

  outBuf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 2.0f) << "index " << i;
  }
}

TEST_P(CommandBufferTest, DropCommandBufferBeforeStreamSync) {
  const char* src = multConstSource();
  if (!src) GTEST_SKIP();

  const size_t N = 256;
  const uint32_t localSize = 64;

  auto lib = device().loadLibraryFromText(src);
  auto fn = lib.lookupFunction("mult_const_f");

  std::vector<float> output(N, 0.0f);
  auto outBuf = device().allocateBuffer(N * sizeof(float));

  {
    std::vector<float> input(N);
    for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);
    auto inBuf = device().allocateBuffer(N * sizeof(float));
    inBuf.copy(stream(), input.data(), N * sizeof(float));
    stream().sync();

    CommandBuffer cb(device());
    LaunchArgs la;
    la.global_size(static_cast<uint32_t>(N)).local_size(localSize);
    fn(la, cb)(outBuf, inBuf, 3.0f);
    cb.submit(stream());
    // Both inBuf and cb drop here — the recorded shared_ptr<impl::Buffer>
    // refs in cb's commands are released BEFORE stream().sync() drains the
    // GPU. The underlying allocation must outlive these drops.
  }

  outBuf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i) * 3.0f) << "index " << i;
  }
}

// FR #20 regression: cb-recorded host->buffer copy must capture the source
// bytes at record time, not the pointer. A stack-local source whose frame is
// gone by submit() must still produce the correct dst contents.
TEST_P(CommandBufferTest, CopyHostStackLocalThroughCommandBuffer) {
  const size_t N = 16;
  auto buf = device().allocateBuffer(N * sizeof(uint32_t));

  CommandBuffer cb(device());

  // Record from a stack frame that immediately returns. Without the fix the
  // dst will see whatever happens to occupy that stack address at submit().
  auto recordFromTransientFrame = [&]() {
    uint32_t tmp[N];
    for (size_t i = 0; i < N; i++)
      tmp[i] = 0xC0FFEE00u + static_cast<uint32_t>(i);
    buf.copy(cb, tmp, N * sizeof(uint32_t));
  };
  recordFromTransientFrame();

  // Stomp the prior stack frame so any pointer-capture bug surfaces.
  {
    volatile uint32_t scratch[N];
    for (size_t i = 0; i < N; i++) scratch[i] = 0xDEADBEEFu;
    (void)scratch;
  }

  cb.submit(stream());
  stream().sync();

  std::vector<uint32_t> output(N, 0u);
  buf.copyTo(stream(), output.data(), N * sizeof(uint32_t));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_EQ(output[i], 0xC0FFEE00u + static_cast<uint32_t>(i))
        << "index " << i;
  }
}

// FR #19: D2H readback recorded on a CommandBuffer used to throw on Metal
// for Private storage. The cb path now defers the host memcpy via a
// completion handler so stream.sync() drains the readback into dst.
TEST_P(CommandBufferTest, ReadbackThroughCommandBuffer) {
  const size_t N = 16;
  std::vector<float> input(N);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i) * 0.5f;

  auto buf = device().allocateBuffer(N * sizeof(float));
  buf.copy(stream(), input.data(), N * sizeof(float));
  stream().sync();

  std::vector<float> output(N, -1.0f);
  CommandBuffer cb(device());
  buf.copyTo(cb, output.data(), N * sizeof(float));
  cb.submit(stream());
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], input[i]) << "index " << i;
  }
}

// onCompletion: handler registered before submit fires after the cb's
// recorded work has completed on the GPU. stream.sync() must observe the
// handler having run.
TEST_P(CommandBufferTest, OnCompletionFiresAfterSubmit) {
  const size_t N = 16;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto src = device().allocateBuffer(N * sizeof(float));
  auto dst = device().allocateBuffer(N * sizeof(float));
  src.copy(stream(), input.data(), N * sizeof(float));
  stream().sync();

  // Wrap the flag so the handler's stack-local captures stay safe through
  // the async completion (Metal/Vulkan/DirectX can fire on a worker thread).
  auto fired = std::make_shared<std::atomic<int>>(0);

  CommandBuffer cb(device());
  dst.copy(cb, src, N * sizeof(float));
  cb.onCompletion([fired]() { fired->fetch_add(1); });
  cb.onCompletion([fired]() { fired->fetch_add(10); });
  cb.submit(stream());
  stream().sync();

  EXPECT_EQ(fired->load(), 11) << "both handlers must have run";

  dst.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();
  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], input[i]) << "index " << i;
  }
}

GHOST_INSTANTIATE_KERNEL_TESTS(CommandBufferTest);

// ---------------------------------------------------------------------------
// Vulkan native CommandBuffer smoke test
//
// CommandBufferTest is parameterized over kernelBackends() which excludes
// Vulkan (no runtime text-to-SPIRV path). This standalone test exercises
// the native CommandBufferVulkan submit / fence / barrier path with
// kernel-free ops (copy + fill) so the native cb is at least minimally
// validated on Vulkan.
// ---------------------------------------------------------------------------

#if WITH_VULKAN
TEST(CommandBufferVulkanSmoke, CopyAndFillAndBarrier) {
  std::unique_ptr<ghost::Device> devPtr;
  try {
    devPtr = ghost::createDevice(ghost::Backend::Vulkan);
  } catch (...) {
  }
  if (!devPtr) GTEST_SKIP() << "No Vulkan device available";
  ghost::Device& dev = *devPtr;
  ghost::Stream s = dev.defaultStream();

  const size_t N = 16;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto src = dev.allocateBuffer(N * sizeof(float));
  auto mid = dev.allocateBuffer(N * sizeof(float));
  auto dst = dev.allocateBuffer(N * sizeof(float));
  src.copy(s, input.data(), N * sizeof(float));
  s.sync();

  ghost::CommandBuffer cb(dev);
  mid.copy(cb, src, N * sizeof(float));
  cb.barrier();
  dst.copy(cb, mid, N * sizeof(float));
  cb.submit(s);
  s.sync();

  dst.copyTo(s, output.data(), N * sizeof(float));
  s.sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i)) << "index " << i;
  }

  // Reset and reuse the same cb.
  cb.reset();
  dst.copy(cb, src, N * sizeof(float));
  cb.submit(s);
  s.sync();
  std::fill(output.begin(), output.end(), 0.0f);
  dst.copyTo(s, output.data(), N * sizeof(float));
  s.sync();
  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i)) << "reuse index " << i;
  }
}

// Submit-then-sync ordering: cb.submit() followed immediately by
// stream.sync() with no intervening stream ops. Stream.sync must wait
// for the cb's submission. We use a host-visible mapped buffer so the
// host can observe the GPU's writes directly — going through another
// stream op for the readback would mask the bug because queue FIFO
// would drag the cb's work along anyway.
TEST(CommandBufferVulkanSmoke, SubmitThenSyncWaitsForCommandBuffer) {
  std::unique_ptr<ghost::Device> devPtr;
  try {
    devPtr = ghost::createDevice(ghost::Backend::Vulkan);
  } catch (...) {
  }
  if (!devPtr) GTEST_SKIP() << "No Vulkan device available";
  ghost::Device& dev = *devPtr;
  ghost::Stream s = dev.defaultStream();

  // 1 MiB — large enough that the GPU fill is unlikely to complete
  // synchronously, so the bug-or-not state is observable.
  const size_t N = 256 * 1024;
  auto mapped = dev.allocateMappedBuffer(
      N * sizeof(uint32_t), ghost::BufferOptions{ghost::Access::ReadWrite,
                                                 ghost::AllocHint::Staging});

  // Pre-fill the mapped buffer with a sentinel via the host. Use map
  // with sync=true to be sure all prior GPU work has drained first.
  auto* hostPtr =
      static_cast<uint32_t*>(mapped.map(s, ghost::Access::ReadWrite));
  ASSERT_NE(hostPtr, nullptr);
  const uint32_t sentinel = 0xDEADBEEFu;
  for (size_t i = 0; i < N; i++) hostPtr[i] = sentinel;
  mapped.unmap(s);
  s.sync();

  // Record a fill into a CommandBuffer and submit+sync. After sync the
  // host MUST see the pattern, not the sentinel.
  ghost::CommandBuffer cb(dev);
  const uint8_t pattern = 0x5A;  // vkCmdFillBuffer fills 32-bit lanes,
                                 // every byte = 0x5A → 0x5A5A5A5A.
  mapped.fill(cb, 0, N * sizeof(uint32_t), pattern);
  cb.submit(s);
  s.sync();

  hostPtr = static_cast<uint32_t*>(mapped.map(s, ghost::Access::ReadOnly));
  ASSERT_NE(hostPtr, nullptr);
  const uint32_t expected = 0x5A5A5A5Au;
  // Sample a few positions across the buffer — if sync returned early,
  // at least the tail of the buffer is likely to still hold the sentinel.
  EXPECT_EQ(hostPtr[0], expected) << "head still " << std::hex << hostPtr[0];
  EXPECT_EQ(hostPtr[N / 2], expected)
      << "mid still " << std::hex << hostPtr[N / 2];
  EXPECT_EQ(hostPtr[N - 1], expected)
      << "tail still " << std::hex << hostPtr[N - 1];
  mapped.unmap(s);
}
#endif  // WITH_VULKAN

// ---------------------------------------------------------------------------
// DirectX native CommandBuffer smoke test
//
// Written blind on Linux. The Windows agent runs this to validate the
// CommandBufferDirectX path end-to-end. Mirrors the Vulkan smoke tests:
// copy / barrier / reset, plus the submit-then-sync ordering case. On
// DirectX the ordering case should pass without further work because
// StreamDirectX::executeOnQueue already bumps the stream's monotonic
// fence (see src/directx/directx_device.cpp).
// ---------------------------------------------------------------------------

#if WITH_DIRECTX
TEST(CommandBufferDirectXSmoke, CopyAndFillAndBarrier) {
  std::unique_ptr<ghost::Device> devPtr;
  try {
    devPtr = ghost::createDevice(ghost::Backend::DirectX);
  } catch (...) {
  }
  if (!devPtr) GTEST_SKIP() << "No DirectX device available";
  ghost::Device& dev = *devPtr;
  ghost::Stream s = dev.defaultStream();

  const size_t N = 16;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto src = dev.allocateBuffer(N * sizeof(float));
  auto mid = dev.allocateBuffer(N * sizeof(float));
  auto dst = dev.allocateBuffer(N * sizeof(float));
  src.copy(s, input.data(), N * sizeof(float));
  s.sync();

  ghost::CommandBuffer cb(dev);
  mid.copy(cb, src, N * sizeof(float));
  cb.barrier();
  dst.copy(cb, mid, N * sizeof(float));
  cb.submit(s);
  s.sync();

  dst.copyTo(s, output.data(), N * sizeof(float));
  s.sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i)) << "index " << i;
  }

  // Reset and reuse the same cb.
  cb.reset();
  dst.copy(cb, src, N * sizeof(float));
  cb.submit(s);
  s.sync();
  std::fill(output.begin(), output.end(), 0.0f);
  dst.copyTo(s, output.data(), N * sizeof(float));
  s.sync();
  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i)) << "reuse index " << i;
  }
}

// Submit-then-sync ordering on DirectX. The cb does a copy from a
// host-uploaded src to a readback dst, then submit + sync. After sync
// the host MUST see the uploaded data in dst, because dst was zero
// before the cb's copy. If StreamDirectX::executeOnQueue ever stops
// bumping the stream's fence, this test will start failing.
//
// Uses MappedBuffer with Access::ReadWrite (UPLOAD heap on D3D12) for
// the src so the host can write directly; Access::WriteOnly (READBACK
// heap) for the dst so the host can read what the cb wrote.
TEST(CommandBufferDirectXSmoke, SubmitThenSyncWaitsForCommandBuffer) {
  std::unique_ptr<ghost::Device> devPtr;
  try {
    devPtr = ghost::createDevice(ghost::Backend::DirectX);
  } catch (...) {
  }
  if (!devPtr) GTEST_SKIP() << "No DirectX device available";
  ghost::Device& dev = *devPtr;
  ghost::Stream s = dev.defaultStream();

  // 1 MiB — large enough that the GPU copy is unlikely to be free.
  const size_t N = 256 * 1024;
  auto src = dev.allocateMappedBuffer(
      N * sizeof(uint32_t), ghost::BufferOptions{ghost::Access::ReadWrite,
                                                 ghost::AllocHint::Staging});
  auto dst = dev.allocateMappedBuffer(
      N * sizeof(uint32_t), ghost::BufferOptions{ghost::Access::WriteOnly,
                                                 ghost::AllocHint::Staging});

  // Host-write a recognizable pattern into the UPLOAD-heap src.
  auto* srcPtr = static_cast<uint32_t*>(src.map(s, ghost::Access::ReadWrite));
  ASSERT_NE(srcPtr, nullptr);
  const uint32_t pattern = 0xCAFEBABEu;
  for (size_t i = 0; i < N; i++) srcPtr[i] = pattern;
  src.unmap(s);

  // Read dst's current state (should be zero or uninitialized) to
  // contrast against the post-cb state.
  auto* dstPtr = static_cast<uint32_t*>(dst.map(s, ghost::Access::ReadOnly));
  ASSERT_NE(dstPtr, nullptr);
  // Don't assert on pre-state — just snapshot the head value as a
  // sanity baseline; D3D12 READBACK heap initial contents aren't
  // formally specified, but with a fresh allocation it's almost
  // certainly zero on every implementation we care about.
  uint32_t preHead = dstPtr[0];
  dst.unmap(s);

  // Record a copy through a CommandBuffer and submit+sync.
  ghost::CommandBuffer cb(dev);
  dst.copy(cb, src, N * sizeof(uint32_t));
  cb.submit(s);
  s.sync();

  // After sync, the host MUST see the pattern in dst.
  dstPtr = static_cast<uint32_t*>(dst.map(s, ghost::Access::ReadOnly));
  ASSERT_NE(dstPtr, nullptr);
  EXPECT_EQ(dstPtr[0], pattern)
      << "head: pre=" << std::hex << preHead << " post=" << dstPtr[0];
  EXPECT_EQ(dstPtr[N / 2], pattern) << "mid: " << std::hex << dstPtr[N / 2];
  EXPECT_EQ(dstPtr[N - 1], pattern) << "tail: " << std::hex << dstPtr[N - 1];
  dst.unmap(s);
}
#endif  // WITH_DIRECTX

// ---------------------------------------------------------------------------
// Metal native CommandBuffer smoke test
//
// Mirrors the Vulkan/DirectX smoke tests. Validates the
// CommandBufferMetal path: native cb allocated per-submit on the stream's
// queue, recorded variants replayed into it, barrier between dispatches
// realised as encoder boundary, and stream.sync() waits for the submitted
// cb via the StreamMetal::attachCommitted hook.
// ---------------------------------------------------------------------------

#if WITH_METAL
TEST(CommandBufferMetalSmoke, CopyAndFillAndBarrier) {
  std::unique_ptr<ghost::Device> devPtr;
  try {
    devPtr = ghost::createDevice(ghost::Backend::Metal);
  } catch (...) {
  }
  if (!devPtr) GTEST_SKIP() << "No Metal device available";
  ghost::Device& dev = *devPtr;
  ghost::Stream s = dev.defaultStream();

  const size_t N = 16;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto src = dev.allocateBuffer(N * sizeof(float));
  auto mid = dev.allocateBuffer(N * sizeof(float));
  auto dst = dev.allocateBuffer(N * sizeof(float));
  src.copy(s, input.data(), N * sizeof(float));
  s.sync();

  ghost::CommandBuffer cb(dev);
  mid.copy(cb, src, N * sizeof(float));
  cb.barrier();
  dst.copy(cb, mid, N * sizeof(float));
  cb.submit(s);
  s.sync();

  dst.copyTo(s, output.data(), N * sizeof(float));
  s.sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i)) << "index " << i;
  }

  // Reset and reuse the same cb.
  cb.reset();
  dst.copy(cb, src, N * sizeof(float));
  cb.submit(s);
  s.sync();
  std::fill(output.begin(), output.end(), 0.0f);
  dst.copyTo(s, output.data(), N * sizeof(float));
  s.sync();
  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i)) << "reuse index " << i;
  }
}

// Submit-then-sync ordering on Metal. The cb fills a host-visible mapped
// buffer with a pattern; submit + sync must observe the pattern (not the
// pre-filled sentinel) from the host afterwards. Without
// StreamMetal::attachCommitted wiring CommandBufferMetal's cb into the
// stream's _lastCommitted, sync() would short-circuit.
TEST(CommandBufferMetalSmoke, SubmitThenSyncWaitsForCommandBuffer) {
  std::unique_ptr<ghost::Device> devPtr;
  try {
    devPtr = ghost::createDevice(ghost::Backend::Metal);
  } catch (...) {
  }
  if (!devPtr) GTEST_SKIP() << "No Metal device available";
  ghost::Device& dev = *devPtr;
  ghost::Stream s = dev.defaultStream();

  // 1 MiB — large enough that the GPU fill is unlikely to finish
  // synchronously with submit.
  const size_t N = 256 * 1024;
  auto mapped = dev.allocateMappedBuffer(
      N * sizeof(uint32_t), ghost::BufferOptions{ghost::Access::ReadWrite,
                                                 ghost::AllocHint::Staging});

  auto* hostPtr =
      static_cast<uint32_t*>(mapped.map(s, ghost::Access::ReadWrite));
  ASSERT_NE(hostPtr, nullptr);
  const uint32_t sentinel = 0xDEADBEEFu;
  for (size_t i = 0; i < N; i++) hostPtr[i] = sentinel;
  mapped.unmap(s);
  s.sync();

  ghost::CommandBuffer cb(dev);
  const uint8_t pattern = 0x5A;  // every byte = 0x5A → 0x5A5A5A5A
  mapped.fill(cb, 0, N * sizeof(uint32_t), pattern);
  cb.submit(s);
  s.sync();

  hostPtr = static_cast<uint32_t*>(mapped.map(s, ghost::Access::ReadOnly));
  ASSERT_NE(hostPtr, nullptr);
  const uint32_t expected = 0x5A5A5A5Au;
  EXPECT_EQ(hostPtr[0], expected) << "head still " << std::hex << hostPtr[0];
  EXPECT_EQ(hostPtr[N / 2], expected)
      << "mid still " << std::hex << hostPtr[N / 2];
  EXPECT_EQ(hostPtr[N - 1], expected)
      << "tail still " << std::hex << hostPtr[N - 1];
  mapped.unmap(s);
}
#endif  // WITH_METAL
