// Copyright (c) 2025 Digital Anarchy, Inc. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause

#include <ghost/cpu/device.h>
#include <ghost/thread_pool.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <set>
#include <thread>
#include <vector>

#include "ghost_test.h"

using namespace ghost;

// ---------------------------------------------------------------------------
// Default pool
// ---------------------------------------------------------------------------

TEST(ThreadPoolTest, CreateDefault) {
  auto pool = ThreadPool::createDefault();
  ASSERT_NE(pool, nullptr);
  EXPECT_GT(pool->workerCount(), 0u);
}

TEST(ThreadPoolTest, CreateDefaultWithExplicitWorkers) {
  auto pool = ThreadPool::createDefault(4);
  ASSERT_NE(pool, nullptr);
  EXPECT_EQ(pool->workerCount(), 4u);
}

TEST(ThreadPoolTest, ParallelRunsAllIndices) {
  auto pool = ThreadPool::createDefault();
  const size_t N = 1024;
  std::vector<int> hits(N, 0);
  pool->parallel(N, [&](size_t i, size_t count) {
    EXPECT_EQ(count, N);
    hits[i] = 1;
  });
  for (size_t i = 0; i < N; i++) EXPECT_EQ(hits[i], 1) << "missing index " << i;
}

TEST(ThreadPoolTest, ParallelIsBlocking) {
  // After parallel() returns, all indices must have completed — no work
  // outstanding for a later sync to wait on.
  auto pool = ThreadPool::createDefault();
  std::atomic<size_t> done{0};
  pool->parallel(256, [&](size_t, size_t) { done.fetch_add(1); });
  EXPECT_EQ(done.load(), 256u);
}

TEST(ThreadPoolTest, ParallelZeroIsNoOp) {
  auto pool = ThreadPool::createDefault();
  std::atomic<size_t> hit{0};
  pool->parallel(0, [&](size_t, size_t) { hit.fetch_add(1); });
  EXPECT_EQ(hit.load(), 0u);
}

TEST(ThreadPoolTest, ParallelOneRunsInline) {
  auto pool = ThreadPool::createDefault();
  std::thread::id callerThread = std::this_thread::get_id();
  std::thread::id workerThread{};
  pool->parallel(
      1, [&](size_t, size_t) { workerThread = std::this_thread::get_id(); });
  EXPECT_EQ(workerThread, callerThread);
}

TEST(ThreadPoolTest, NestedParallelDoesNotDeadlock) {
  auto pool = ThreadPool::createDefault();
  const size_t Outer = 8;
  const size_t Inner = 16;
  std::atomic<size_t> total{0};
  pool->parallel(Outer, [&](size_t, size_t) {
    pool->parallel(Inner, [&](size_t, size_t) { total.fetch_add(1); });
  });
  EXPECT_EQ(total.load(), Outer * Inner);
}

// Regression test for the dispatch-state publication race that produced
// missed and double-dispatched shards under back-to-back varied-count
// dispatches. The original failure mode: parent's stack-allocated dispatch
// state was overwritten by the next parallel() call at the same address
// while a worker from the previous dispatch was still reading
// state->numShards / state->total, causing two workers to compute the
// same shard range and one shard to go unprocessed. A single dispatch
// like ParallelRunsAllIndices does not expose this — the bug requires
// many back-to-back calls with varied counts so the stack is repeatedly
// reused. Atomic hits detect the duplicate writes that non-atomic counters
// would silently merge.
//
// Pre-fix this fired within the first ~10K dispatches reproducibly. The
// 16K dispatches here keep CI under 2s while still catching regressions.
TEST(ThreadPoolTest, NoMissedOrDoubleDispatchUnderVariedCounts) {
  auto pool = ThreadPool::createDefault();
  const std::vector<size_t> counts = {2, 4, 8, 12, 16, 24, 32, 64};
  for (size_t iter = 0; iter < 2000; ++iter) {
    for (size_t count : counts) {
      std::vector<std::atomic<int>> hits(count);
      for (auto& h : hits) h.store(0);
      pool->parallel(count,
                     [&hits](size_t i, size_t) { hits[i].fetch_add(1); });
      for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(hits[i].load(), 1)
            << "iter=" << iter << " count=" << count << " idx=" << i;
      }
    }
  }
}

// Helper: a small varied-count smoke test parameterised by spinDuration.
// Same shape as the regression test above but smaller (the regression net
// is the test above; these just verify the spin-duration knob plumbing
// produces a working pool).
namespace {
void smokeTestPool(ThreadPool& pool) {
  const std::vector<size_t> counts = {2, 4, 8, 16, 24};
  for (size_t iter = 0; iter < 200; ++iter) {
    for (size_t count : counts) {
      std::vector<std::atomic<int>> hits(count);
      for (auto& h : hits) h.store(0);
      pool.parallel(count, [&hits](size_t i, size_t) { hits[i].fetch_add(1); });
      for (size_t i = 0; i < count; ++i) {
        ASSERT_EQ(hits[i].load(), 1)
            << "iter=" << iter << " count=" << count << " idx=" << i;
      }
    }
  }
}
}  // namespace

// Passive mode: workers park immediately on idle. Verifies the
// notify-when-parked path (parent's lock-gated isParked check, last-worker
// _doneCv notify) is correct and not subject to lost wakeups, since
// workers are always in cv.wait between dispatches.
TEST(ThreadPoolTest, CreateDefault_Passive_NoSpin) {
  auto pool = ThreadPool::createDefault(0, std::chrono::microseconds(0));
  ASSERT_NE(pool, nullptr);
  smokeTestPool(*pool);
}

// Active mode: workers effectively never park. Verifies the
// skip-notify-when-spinning path (parent's lock-gated isParked check sees
// false, no notify_one issued) is correct — workers must observe the new
// assignedEpoch through their spin loop, not via cv signal.
TEST(ThreadPoolTest, CreateDefault_Active_LongSpin) {
  auto pool = ThreadPool::createDefault(0, std::chrono::hours(24));
  ASSERT_NE(pool, nullptr);
  smokeTestPool(*pool);
}

// Destruction while workers are parked must not hang. With spinDuration=0
// the workers have already entered cv.wait by the time the test reaches
// the closing brace, so the destructor's _shouldStop + epoch bump must
// successfully wake all 24 workers under their per-slot mutex+cv. This
// catches a regression where the destructor only signals the global stop
// flag without waking the cvs.
TEST(ThreadPoolTest, DestructorWakesParkedWorkers) {
  // Wrap in a thread with a watchdog so a hang manifests as a test
  // failure rather than a hung process.
  std::atomic<bool> destroyed{false};
  std::thread t([&] {
    {
      auto pool = ThreadPool::createDefault(0, std::chrono::microseconds(0));
      // Issue one dispatch so workers have transitioned through park
      // (their first-iteration spin window is 0, so they cv.wait
      // immediately on construction; this dispatch wakes them once).
      pool->parallel(4, [](size_t, size_t) {});
      // Workers loop back, see no new epoch, and park again.
      // pool falls out of scope here — destructor must wake them.
    }
    destroyed.store(true, std::memory_order_release);
  });

  // 5-second watchdog. Realistic destructor finishes in <1 ms; if it
  // hangs we'd otherwise block the entire test binary indefinitely.
  for (int i = 0; i < 5000; ++i) {
    if (destroyed.load(std::memory_order_acquire)) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_TRUE(destroyed.load()) << "pool destructor hung — parked workers "
                                   "not woken";
  t.join();
}

// ---------------------------------------------------------------------------
// Device::threadPool() and DeviceCPU injection
// ---------------------------------------------------------------------------

TEST(ThreadPoolTest, CpuDeviceExposesPool) {
  DeviceCPU dev;
  auto pool = dev.threadPool();
  ASSERT_NE(pool, nullptr);
  EXPECT_GT(pool->workerCount(), 0u);
}

namespace {
class CountingPool : public ThreadPool {
 public:
  std::atomic<size_t> calls{0};

  void parallel(size_t count, std::function<void(size_t, size_t)> fn) override {
    calls.fetch_add(1);
    for (size_t i = 0; i < count; i++) fn(i, count);
  }

  size_t workerCount() const override { return 1; }
};
}  // namespace

TEST(ThreadPoolTest, CpuDeviceCustomPoolViaConstructor) {
  auto custom = std::make_shared<CountingPool>();
  DeviceCPU dev(custom);
  EXPECT_EQ(dev.threadPool().get(), custom.get());
}

TEST(ThreadPoolTest, CpuDeviceCustomPoolReceivesDispatch) {
  auto custom = std::make_shared<CountingPool>();
  DeviceCPU dev(custom);

  // Run a CPU kernel through an inline library and verify the custom pool
  // saw the dispatch.
  static int calls = 0;
  auto kernel = +[](size_t i, size_t n, const std::vector<Attribute>& args) {
    (void)i;
    (void)n;
    (void)args;
    calls++;
  };
  calls = 0;
  Library lib = dev.loadLibraryFromFunctions({{"k", kernel}});
  Function fn = lib.lookupFunction("k");
  LaunchArgs la;
  la.global_size(64u).local_size(1u);
  fn(la, dev.defaultStream())();
  dev.defaultStream().sync();

  EXPECT_GE(custom->calls.load(), 1u);
  EXPECT_GT(calls, 0);
}

TEST(ThreadPoolTest, SetThreadPoolReplacesPool) {
  DeviceCPU dev;
  auto custom = std::make_shared<CountingPool>();
  dev.setThreadPool(custom);
  EXPECT_EQ(dev.threadPool().get(), custom.get());
}

TEST(ThreadPoolTest, SetThreadPoolNullRevertsToDefault) {
  auto custom = std::make_shared<CountingPool>();
  DeviceCPU dev(custom);
  dev.setThreadPool(nullptr);
  ASSERT_NE(dev.threadPool(), nullptr);
  EXPECT_NE(dev.threadPool().get(), static_cast<ThreadPool*>(custom.get()));
}

// GPU backends should not expose a thread pool; CPU should.
class DeviceThreadPoolTest : public ghost::test::GhostTest {};

TEST_P(DeviceThreadPoolTest, ThreadPoolPresenceMatchesBackend) {
  if (backend() == Backend::CPU) {
    EXPECT_NE(device().threadPool(), nullptr);
  } else {
    EXPECT_EQ(device().threadPool(), nullptr);
  }
}

GHOST_INSTANTIATE_BACKEND_TESTS(DeviceThreadPoolTest);
