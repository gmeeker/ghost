#include "ghost_test.h"

using namespace ghost;
using namespace ghost::test;

class StreamTest : public GhostTest {};

// ---------------------------------------------------------------------------
// Basic stream operations
// ---------------------------------------------------------------------------

TEST_P(StreamTest, DefaultStreamSync) { EXPECT_NO_THROW(stream().sync()); }

TEST_P(StreamTest, CreateAndSync) {
  auto s = device().createStream();
  EXPECT_NO_THROW(s.sync());
}

TEST_P(StreamTest, MultipleStreams) {
  auto s1 = device().createStream();
  auto s2 = device().createStream();

  // Both should be independently syncable.
  EXPECT_NO_THROW(s1.sync());
  EXPECT_NO_THROW(s2.sync());
}

TEST_P(StreamTest, IndependentStreamWork) {
  // Two streams copying data independently.
  const size_t N = 16;
  std::vector<float> data1(N, 1.0f), data2(N, 2.0f);
  std::vector<float> out1(N, 0.0f), out2(N, 0.0f);

  auto s1 = device().createStream();
  auto s2 = device().createStream();

  auto buf1 = device().allocateBuffer(N * sizeof(float));
  auto buf2 = device().allocateBuffer(N * sizeof(float));

  buf1.copy(s1, data1.data(), N * sizeof(float));
  buf2.copy(s2, data2.data(), N * sizeof(float));

  buf1.copyTo(s1, out1.data(), N * sizeof(float));
  buf2.copyTo(s2, out2.data(), N * sizeof(float));

  s1.sync();
  s2.sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(out1[i], 1.0f);
    EXPECT_FLOAT_EQ(out2[i], 2.0f);
  }
}

GHOST_INSTANTIATE_BACKEND_TESTS(StreamTest);

// ---------------------------------------------------------------------------
// Event tests (separate fixture since events may not be supported)
// ---------------------------------------------------------------------------

class EventTest : public GhostTest {};

TEST_P(EventTest, RecordAndWait) {
  Event event(nullptr);
  try {
    event = stream().record();
  } catch (const ghost::unsupported_error&) {
    GTEST_SKIP() << "Events not supported on " << BackendName(backend());
  }

  EXPECT_NO_THROW(event.wait());
}

TEST_P(EventTest, IsCompleteAfterSync) {
  // Enqueue some work so the event has something to wait on.
  const size_t N = 256;
  std::vector<float> data(N, 1.0f);
  auto buf = device().allocateBuffer(N * sizeof(float));
  buf.copy(stream(), data.data(), N * sizeof(float));

  Event event(nullptr);
  try {
    event = stream().record();
  } catch (const ghost::unsupported_error&) {
    GTEST_SKIP() << "Events not supported";
  }

  event.wait();
  EXPECT_TRUE(event.isComplete());
}

TEST_P(EventTest, ElapsedIsNonNegative) {
  const size_t N = 256;
  std::vector<float> data(N, 1.0f);
  auto buf = device().allocateBuffer(N * sizeof(float));

  Event start(nullptr), end(nullptr);
  try {
    start = stream().record();
  } catch (const ghost::unsupported_error&) {
    GTEST_SKIP() << "Events not supported";
  }

  buf.copy(stream(), data.data(), N * sizeof(float));
  end = stream().record();
  end.wait();

  double elapsed = Event::elapsed(start, end);
  EXPECT_GE(elapsed, 0.0);
}

TEST_P(EventTest, CrossStreamWait) {
  const size_t N = 16;
  std::vector<float> input(N), output(N, 0.0f);
  for (size_t i = 0; i < N; i++) input[i] = static_cast<float>(i);

  auto s1 = device().createStream();
  auto s2 = device().createStream();
  auto buf = device().allocateBuffer(N * sizeof(float));

  // Copy on s1, record event.
  buf.copy(s1, input.data(), N * sizeof(float));

  Event event(nullptr);
  try {
    event = s1.record();
  } catch (const ghost::unsupported_error&) {
    GTEST_SKIP() << "Events not supported";
  }

  // s2 waits for s1's event, then reads.
  try {
    s2.waitForEvent(event);
  } catch (const ghost::unsupported_error&) {
    GTEST_SKIP() << "Cross-stream wait not supported";
  }
  buf.copyTo(s2, output.data(), N * sizeof(float));
  s2.sync();

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i)) << "index " << i;
  }
}

TEST_P(EventTest, RapidRecordWaitCycles) {
  const size_t N = 256;
  auto buf = device().allocateBuffer(N * sizeof(float));
  std::vector<float> data(N, 1.0f);

  for (int cycle = 0; cycle < 50; cycle++) {
    buf.copy(stream(), data.data(), N * sizeof(float));
    Event event(nullptr);
    try {
      event = stream().record();
    } catch (const ghost::unsupported_error&) {
      GTEST_SKIP() << "Events not supported";
    }
    event.wait();
    EXPECT_TRUE(event.isComplete());
  }
}

TEST_P(EventTest, EventTimestamp) {
  const size_t N = 256;
  auto buf = device().allocateBuffer(N * sizeof(float));
  std::vector<float> data(N, 1.0f);
  buf.copy(stream(), data.data(), N * sizeof(float));

  Event event(nullptr);
  try {
    event = stream().record();
  } catch (const ghost::unsupported_error&) {
    GTEST_SKIP() << "Events not supported";
  }
  event.wait();

  double ts = event.timestamp();
  // Timestamp may be 0 on backends that don't support absolute timestamps.
  EXPECT_GE(ts, 0.0);
}

GHOST_INSTANTIATE_BACKEND_TESTS(EventTest);
