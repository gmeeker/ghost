#include "ghost_test.h"

using namespace ghost;
using namespace ghost::test;

class BufferTest : public GhostTest {};

// ---------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------

TEST_P(BufferTest, AllocateAndSize) {
  auto buf = device().allocateBuffer(1024);
  EXPECT_EQ(buf.size(), 1024u);
}

TEST_P(BufferTest, AllocateMultiple) {
  auto a = device().allocateBuffer(512);
  auto b = device().allocateBuffer(256);
  EXPECT_EQ(a.size(), 512u);
  EXPECT_EQ(b.size(), 256u);
}

// ---------------------------------------------------------------------------
// Host <-> Device copy round-trip
// ---------------------------------------------------------------------------

TEST_P(BufferTest, HostDeviceRoundTrip) {
  const size_t N = 32;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto buf = device().allocateBuffer(N * sizeof(float));
  buf.copy(stream(), input.data(), N * sizeof(float));
  buf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i)) << "index " << i;
  }
}

// ---------------------------------------------------------------------------
// Device-to-device copy
// ---------------------------------------------------------------------------

TEST_P(BufferTest, DeviceToDeviceCopy) {
  const size_t N = 16;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i * 2);

  auto src = device().allocateBuffer(N * sizeof(float));
  auto dst = device().allocateBuffer(N * sizeof(float));

  src.copy(stream(), input.data(), N * sizeof(float));
  dst.copy(stream(), src, N * sizeof(float));
  dst.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i * 2)) << "index " << i;
  }
}

// ---------------------------------------------------------------------------
// Copy with offsets
// ---------------------------------------------------------------------------

TEST_P(BufferTest, CopyWithOffsets) {
  const size_t N = 16;
  std::vector<float> input(N), output(N, -1.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto src = device().allocateBuffer(N * sizeof(float));
  auto dst = device().allocateBuffer(N * sizeof(float));

  // Fill dst with zeros first.
  std::vector<float> zeros(N, 0.0f);
  dst.copy(stream(), zeros.data(), N * sizeof(float));

  // Copy src elements [4..8) into dst at offset 8 (elements [8..12)).
  src.copy(stream(), input.data(), N * sizeof(float));
  dst.copy(stream(), src, 4 * sizeof(float), 8 * sizeof(float),
           4 * sizeof(float));
  dst.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  // dst[0..8) should be 0, dst[8..12) should be {4,5,6,7}, dst[12..16) = 0.
  for (size_t i = 0; i < 8; i++) {
    EXPECT_FLOAT_EQ(output[i], 0.0f) << "index " << i;
  }
  for (size_t i = 8; i < 12; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i - 4)) << "index " << i;
  }
  for (size_t i = 12; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], 0.0f) << "index " << i;
  }
}

TEST_P(BufferTest, HostCopyWithOffset) {
  const size_t N = 8;
  std::vector<float> data = {10.0f, 20.0f};
  std::vector<float> output(N, 0.0f);

  auto buf = device().allocateBuffer(N * sizeof(float));
  // Zero the buffer.
  buf.copy(stream(), output.data(), N * sizeof(float));
  // Write {10, 20} at offset 2 (element index 2).
  buf.copy(stream(), data.data(), 2 * sizeof(float), 2 * sizeof(float));
  buf.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  EXPECT_FLOAT_EQ(output[0], 0.0f);
  EXPECT_FLOAT_EQ(output[1], 0.0f);
  EXPECT_FLOAT_EQ(output[2], 10.0f);
  EXPECT_FLOAT_EQ(output[3], 20.0f);
  EXPECT_FLOAT_EQ(output[4], 0.0f);
}

TEST_P(BufferTest, CopyToWithOffset) {
  const size_t N = 8;
  std::vector<float> input(N);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto buf = device().allocateBuffer(N * sizeof(float));
  buf.copy(stream(), input.data(), N * sizeof(float));

  // Read elements [4..6) from the buffer.
  std::vector<float> output(2, -1.0f);
  buf.copyTo(stream(), output.data(), 4 * sizeof(float), 2 * sizeof(float));
  stream().sync();

  EXPECT_FLOAT_EQ(output[0], 4.0f);
  EXPECT_FLOAT_EQ(output[1], 5.0f);
}

// ---------------------------------------------------------------------------
// Fill
// ---------------------------------------------------------------------------

TEST_P(BufferTest, FillByte) {
  const size_t N = 64;
  auto buf = device().allocateBuffer(N);
  buf.fill(stream(), 0, N, uint8_t(0xAB));

  std::vector<uint8_t> output(N, 0);
  buf.copyTo(stream(), output.data(), N);
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_EQ(output[i], 0xAB) << "index " << i;
  }
}

TEST_P(BufferTest, FillPartial) {
  const size_t N = 64;
  auto buf = device().allocateBuffer(N);

  // Zero the whole buffer, then fill only [16..48) with 0xFF.
  buf.fill(stream(), 0, N, uint8_t(0));
  buf.fill(stream(), 16, 32, uint8_t(0xFF));

  std::vector<uint8_t> output(N, 0x42);
  buf.copyTo(stream(), output.data(), N);
  stream().sync();

  for (size_t i = 0; i < 16; i++) {
    EXPECT_EQ(output[i], 0) << "index " << i;
  }
  for (size_t i = 16; i < 48; i++) {
    EXPECT_EQ(output[i], 0xFF) << "index " << i;
  }
  for (size_t i = 48; i < N; i++) {
    EXPECT_EQ(output[i], 0) << "index " << i;
  }
}

TEST_P(BufferTest, FillPattern) {
  const size_t N = 32;
  auto buf = device().allocateBuffer(N * sizeof(uint32_t));

  uint32_t pattern = 0xDEADBEEF;
  try {
    buf.fill(stream(), 0, N * sizeof(uint32_t), &pattern, sizeof(pattern));
  } catch (const ghost::unsupported_error&) {
    GTEST_SKIP() << "Pattern fill not supported on " << BackendName(backend());
  }

  std::vector<uint32_t> output(N, 0);
  buf.copyTo(stream(), output.data(), N * sizeof(uint32_t));
  stream().sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_EQ(output[i], 0xDEADBEEF) << "index " << i;
  }
}

// ---------------------------------------------------------------------------
// Sub-buffers
// ---------------------------------------------------------------------------

TEST_P(BufferTest, SubBufferCreation) {
  auto parent = device().allocateBuffer(1024);
  Buffer sub(nullptr);
  try {
    sub = parent.createSubBuffer(256, 512);
  } catch (const ghost::unsupported_error&) {
    GTEST_SKIP() << "Sub-buffers not supported on " << BackendName(backend());
  }
  EXPECT_EQ(sub.size(), 512u);
}

TEST_P(BufferTest, SubBufferAliasing) {
  const size_t N = 16;
  std::vector<float> input(N);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto parent = device().allocateBuffer(N * sizeof(float));
  parent.copy(stream(), input.data(), N * sizeof(float));

  Buffer sub(nullptr);
  try {
    // Sub-buffer covering elements [4..8).
    sub = parent.createSubBuffer(4 * sizeof(float), 4 * sizeof(float));
  } catch (const ghost::unsupported_error&) {
    GTEST_SKIP() << "Sub-buffers not supported on " << BackendName(backend());
  }

  // Read from sub-buffer.
  std::vector<float> output(4, -1.0f);
  sub.copyTo(stream(), output.data(), 4 * sizeof(float));
  stream().sync();

  EXPECT_FLOAT_EQ(output[0], 4.0f);
  EXPECT_FLOAT_EQ(output[1], 5.0f);
  EXPECT_FLOAT_EQ(output[2], 6.0f);
  EXPECT_FLOAT_EQ(output[3], 7.0f);
}

TEST_P(BufferTest, SubBufferWriteVisibleInParent) {
  const size_t N = 8;
  std::vector<float> zeros(N, 0.0f);

  auto parent = device().allocateBuffer(N * sizeof(float));
  parent.copy(stream(), zeros.data(), N * sizeof(float));

  Buffer sub(nullptr);
  try {
    sub = parent.createSubBuffer(2 * sizeof(float), 2 * sizeof(float));
  } catch (const ghost::unsupported_error&) {
    GTEST_SKIP() << "Sub-buffers not supported";
  }

  // Write to sub-buffer.
  std::vector<float> data = {99.0f, 100.0f};
  sub.copy(stream(), data.data(), 2 * sizeof(float));

  // Read from parent.
  std::vector<float> output(N, -1.0f);
  parent.copyTo(stream(), output.data(), N * sizeof(float));
  stream().sync();

  EXPECT_FLOAT_EQ(output[0], 0.0f);
  EXPECT_FLOAT_EQ(output[1], 0.0f);
  EXPECT_FLOAT_EQ(output[2], 99.0f);
  EXPECT_FLOAT_EQ(output[3], 100.0f);
  EXPECT_FLOAT_EQ(output[4], 0.0f);
}

TEST_P(BufferTest, SubSubBuffer) {
  // Create a sub-buffer of a sub-buffer and verify the offset composes
  // correctly. This catches double-counting bugs in baseOffset() that
  // single-level sub-buffer tests miss.
  const size_t N = 16;
  std::vector<float> input(N);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto parent = device().allocateBuffer(N * sizeof(float));
  parent.copy(stream(), input.data(), N * sizeof(float));

  Buffer child(nullptr);
  Buffer grandchild(nullptr);
  try {
    // child covers elements [4..12) -> 8 floats
    child = parent.createSubBuffer(4 * sizeof(float), 8 * sizeof(float));
    // grandchild covers child[2..6) -> parent[6..10) -> 4 floats
    grandchild = child.createSubBuffer(2 * sizeof(float), 4 * sizeof(float));
  } catch (const ghost::unsupported_error&) {
    GTEST_SKIP() << "Sub-buffers not supported on " << BackendName(backend());
  }

  // Read through grandchild — should see parent[6..10).
  std::vector<float> output(4, -1.0f);
  grandchild.copyTo(stream(), output.data(), 4 * sizeof(float));
  stream().sync();

  EXPECT_FLOAT_EQ(output[0], 6.0f);
  EXPECT_FLOAT_EQ(output[1], 7.0f);
  EXPECT_FLOAT_EQ(output[2], 8.0f);
  EXPECT_FLOAT_EQ(output[3], 9.0f);

  // Write through grandchild and verify visibility in parent.
  std::vector<float> data = {90.0f, 91.0f, 92.0f, 93.0f};
  grandchild.copy(stream(), data.data(), 4 * sizeof(float));

  std::vector<float> parentOut(N, -1.0f);
  parent.copyTo(stream(), parentOut.data(), N * sizeof(float));
  stream().sync();

  EXPECT_FLOAT_EQ(parentOut[5], 5.0f);
  EXPECT_FLOAT_EQ(parentOut[6], 90.0f);
  EXPECT_FLOAT_EQ(parentOut[7], 91.0f);
  EXPECT_FLOAT_EQ(parentOut[8], 92.0f);
  EXPECT_FLOAT_EQ(parentOut[9], 93.0f);
  EXPECT_FLOAT_EQ(parentOut[10], 10.0f);
}

// ---------------------------------------------------------------------------
// Mapped buffers
// ---------------------------------------------------------------------------

TEST_P(BufferTest, MappedBufferRoundTrip) {
  auto supports = device().getAttribute(kDeviceSupportsMappedBuffer);
  if (!supports.asBool()) {
    GTEST_SKIP() << BackendName(backend())
                 << " does not support mapped buffers";
  }

  const size_t N = 16;
  auto mbuf = device().allocateMappedBuffer(N * sizeof(float));

  // Map for writing.
  float* ptr = nullptr;
  try {
    ptr = static_cast<float*>(mbuf.map(stream(), Access::WriteOnly));
  } catch (const ghost::unsupported_error&) {
    GTEST_SKIP() << "Mapped buffers not supported";
  }
  ASSERT_NE(ptr, nullptr);

  for (size_t i = 0; i < N; i++) ptr[i] = static_cast<float>(i * 3);
  mbuf.unmap(stream());

  // Map for reading and verify.
  ptr = static_cast<float*>(mbuf.map(stream(), Access::ReadOnly));
  ASSERT_NE(ptr, nullptr);
  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(ptr[i], static_cast<float>(i * 3)) << "index " << i;
  }
  mbuf.unmap(stream());
}

// ---------------------------------------------------------------------------
// Host memory allocation
// ---------------------------------------------------------------------------

TEST_P(BufferTest, HostMemory) {
  void* ptr = device().allocateHostMemory(4096);
  ASSERT_NE(ptr, nullptr);
  // Should be writable without crashing.
  std::memset(ptr, 0, 4096);
  device().freeHostMemory(ptr);
}

// ---------------------------------------------------------------------------
// Zero-length buffer
// ---------------------------------------------------------------------------

TEST_P(BufferTest, ZeroLengthBuffer) {
  // Zero-length allocation should either succeed or throw cleanly.
  try {
    auto buf = device().allocateBuffer(0);
    EXPECT_EQ(buf.size(), 0u);
  } catch (const std::exception&) {
    // Throwing is acceptable behavior.
  }
}

// ---------------------------------------------------------------------------
// Mapped buffer with sync=false
// ---------------------------------------------------------------------------

TEST_P(BufferTest, MappedBufferNoSync) {
  auto supports = device().getAttribute(kDeviceSupportsMappedBuffer);
  if (!supports.asBool()) {
    GTEST_SKIP() << BackendName(backend())
                 << " does not support mapped buffers";
  }

  const size_t N = 16;
  auto mbuf = device().allocateMappedBuffer(N * sizeof(float));

  float* ptr = nullptr;
  try {
    ptr = static_cast<float*>(mbuf.map(stream(), Access::WriteOnly, false));
  } catch (const ghost::unsupported_error&) {
    GTEST_SKIP() << "Mapped buffers not supported";
  }
  ASSERT_NE(ptr, nullptr);

  for (size_t i = 0; i < N; i++) ptr[i] = static_cast<float>(i * 5);
  mbuf.unmap(stream());
  stream().sync();

  // Read back and verify.
  ptr = static_cast<float*>(mbuf.map(stream(), Access::ReadOnly));
  ASSERT_NE(ptr, nullptr);
  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(ptr[i], static_cast<float>(i * 5)) << "index " << i;
  }
  mbuf.unmap(stream());
}

// ---------------------------------------------------------------------------
// Copy with mismatched sizes
// ---------------------------------------------------------------------------

TEST_P(BufferTest, CopyPartialBuffer) {
  // Copy fewer bytes than the buffer size — should work.
  const size_t N = 16;
  std::vector<float> input(N), output(N, -1.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto buf = device().allocateBuffer(N * sizeof(float));
  // Only copy first 8 floats.
  buf.copy(stream(), input.data(), 8 * sizeof(float));
  buf.copyTo(stream(), output.data(), 8 * sizeof(float));
  stream().sync();

  for (size_t i = 0; i < 8; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i)) << "index " << i;
  }
}

// ---------------------------------------------------------------------------
// Maximum-size buffer allocation
// ---------------------------------------------------------------------------

TEST_P(BufferTest, MaxSizeBufferAllocation) {
  auto attr = device().getAttribute(kDeviceMaxBufferSize);
  if (!attr.valid() || attr.asUInt64() == 0) {
    GTEST_SKIP() << "kDeviceMaxBufferSize not available";
  }

  uint64_t maxSize = attr.asUInt64();
  // Try allocating near the limit (90% of max to avoid OOM).
  uint64_t allocSize = maxSize * 9 / 10;
  // Cap at 256MB to avoid actually exhausting memory in tests.
  if (allocSize > 256 * 1024 * 1024) {
    allocSize = 256 * 1024 * 1024;
  }

  try {
    auto buf = device().allocateBuffer(static_cast<size_t>(allocSize));
    EXPECT_GE(buf.size(), static_cast<size_t>(allocSize));
  } catch (const std::exception&) {
    // OOM is acceptable in a test environment.
  }
}

// ---------------------------------------------------------------------------
// Many small buffer allocations (resource leak test)
// ---------------------------------------------------------------------------

TEST_P(BufferTest, ManySmallAllocations) {
  const size_t count = 1000;
  const size_t bufSize = 64;

  std::vector<Buffer> buffers;
  buffers.reserve(count);

  for (size_t i = 0; i < count; i++) {
    try {
      buffers.push_back(device().allocateBuffer(bufSize));
    } catch (const std::exception&) {
      // Resource exhaustion is acceptable, but we should get a fair number.
      EXPECT_GT(i, 100u) << "Exhausted resources too quickly";
      break;
    }
  }

  // Verify first and last buffers are still usable.
  if (!buffers.empty()) {
    std::vector<uint8_t> data(bufSize, 0xAA);
    buffers.front().copy(stream(), data.data(), bufSize);
    buffers.back().copy(stream(), data.data(), bufSize);
    stream().sync();
  }
}

// ---------------------------------------------------------------------------
// Device destruction while buffers still alive (RAII cleanup)
// ---------------------------------------------------------------------------

TEST_P(BufferTest, DeviceDestructionWithLiveBuffers) {
  auto dev = createDevice(backend());
  if (!dev) GTEST_SKIP();

  Buffer buf(nullptr);
  {
    buf = dev->allocateBuffer(1024);
    EXPECT_EQ(buf.size(), 1024u);
  }
  // Destroy device while buffer is still alive.
  dev.reset();
  // Buffer destructor should not crash.
}

// ---------------------------------------------------------------------------
// Copy with overlapping regions in same buffer
// ---------------------------------------------------------------------------

TEST_P(BufferTest, CopyOverlappingRegions) {
  const size_t N = 16;
  std::vector<float> input(N);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto buf = device().allocateBuffer(N * sizeof(float));
  buf.copy(stream(), input.data(), N * sizeof(float));
  stream().sync();

  // Copy buf[0..8) to buf[4..12) — overlapping region [4..8).
  try {
    buf.copy(stream(), buf, 0, 4 * sizeof(float), 8 * sizeof(float));
    stream().sync();

    std::vector<float> output(N, -1.0f);
    buf.copyTo(stream(), output.data(), N * sizeof(float));
    stream().sync();

    // buf[4..12) should contain original buf[0..8) = {0,1,2,3,4,5,6,7}.
    for (size_t i = 4; i < 12; i++) {
      EXPECT_FLOAT_EQ(output[i], static_cast<float>(i - 4)) << "index " << i;
    }
    // buf[0..4) should be unchanged: {0,1,2,3}.
    for (size_t i = 0; i < 4; i++) {
      EXPECT_FLOAT_EQ(output[i], static_cast<float>(i)) << "index " << i;
    }
    // buf[12..16) should be unchanged: {12,13,14,15}.
    for (size_t i = 12; i < N; i++) {
      EXPECT_FLOAT_EQ(output[i], static_cast<float>(i)) << "index " << i;
    }
  } catch (const std::exception&) {
    // Some backends may not support self-copy with overlapping regions.
  }
}

// ---------------------------------------------------------------------------
// Sub-buffer at misaligned offset on OpenCL
// ---------------------------------------------------------------------------

TEST_P(BufferTest, SubBufferMisalignedOffset) {
  if (backend() != Backend::OpenCL) {
    GTEST_SKIP() << "Misaligned sub-buffer test is OpenCL-specific";
  }

  auto parent = device().allocateBuffer(1024);

  // Use a 1-byte offset, which is guaranteed to be misaligned on OpenCL
  // (OpenCL requires sub-buffer offsets aligned to
  // CL_DEVICE_MEM_BASE_ADDR_ALIGN).
  try {
    auto sub = parent.createSubBuffer(1, 64);
    // If it succeeds, the backend auto-aligned.
    EXPECT_GT(sub.size(), 0u);
  } catch (const std::exception&) {
    // Expected: OpenCL should throw on misaligned offset.
  }
}

// ---------------------------------------------------------------------------
// Overlapping sub-buffer writes from concurrent streams
// ---------------------------------------------------------------------------

TEST_P(BufferTest, ConcurrentSubBufferWrites) {
  auto parent = device().allocateBuffer(1024);
  Buffer sub1(nullptr), sub2(nullptr);
  try {
    // Two non-overlapping sub-buffers.
    sub1 = parent.createSubBuffer(0, 512);
    sub2 = parent.createSubBuffer(512, 512);
  } catch (const ghost::unsupported_error&) {
    GTEST_SKIP() << "Sub-buffers not supported";
  }

  auto s1 = device().createStream();
  auto s2 = device().createStream();

  // Write different patterns to each sub-buffer on different streams.
  std::vector<uint8_t> pattern1(512, 0xAA);
  std::vector<uint8_t> pattern2(512, 0xBB);

  sub1.copy(s1, pattern1.data(), 512);
  sub2.copy(s2, pattern2.data(), 512);

  s1.sync();
  s2.sync();

  // Read back the full parent buffer and verify both regions.
  std::vector<uint8_t> output(1024, 0);
  parent.copyTo(stream(), output.data(), 1024);
  stream().sync();

  for (size_t i = 0; i < 512; i++) {
    EXPECT_EQ(output[i], 0xAA) << "index " << i;
  }
  for (size_t i = 512; i < 1024; i++) {
    EXPECT_EQ(output[i], 0xBB) << "index " << i;
  }
}

GHOST_INSTANTIATE_BACKEND_TESTS(BufferTest);
