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
