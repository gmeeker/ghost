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
    ptr = static_cast<float*>(mbuf.map(stream(), Access_WriteOnly));
  } catch (const ghost::unsupported_error&) {
    GTEST_SKIP() << "Mapped buffers not supported";
  }
  ASSERT_NE(ptr, nullptr);

  for (size_t i = 0; i < N; i++) ptr[i] = static_cast<float>(i * 3);
  mbuf.unmap(stream());

  // Map for reading and verify.
  ptr = static_cast<float*>(mbuf.map(stream(), Access_ReadOnly));
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

GHOST_INSTANTIATE_BACKEND_TESTS(BufferTest);
