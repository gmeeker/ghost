// Copyright (c) 2025 Digital Anarchy, Inc. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include <ghost/cpu/device.h>
#include <ghost/cpu/impl_device.h>
#include <ghost/thread_pool.h>

#if WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif __sgi__
#include <sys/sysmp.h>
#elif __linux__
#include <sys/sysinfo.h>
#elif __APPLE_CC__
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>

#ifdef __APPLE_CC__
#include <dispatch/dispatch.h>
#endif

namespace ghost {
namespace implementation {

size_t DeviceCPU::getNumberOfCores() {
#if WIN32
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  return (size_t)sysinfo.dwNumberOfProcessors;
#elif __sgi__
  return (size_t)sysmp(MP_NAPROCS);
#elif __linux__
  return (size_t)get_nprocs();
#elif __APPLE_CC__
  int cpus = 0;
  size_t length = sizeof(cpus);
  int error = sysctlbyname("hw.activecpu", &cpus, &length, NULL, 0);
  if (error < 0) cpus = 1;
  return (size_t)cpus;
#elif defined(_SC_NPROCESSORS_ONLN)
  return (size_t)sysconf(_SC_NPROCESSORS_ONLN);
#else
  auto count = std::thread::hardware_concurrency();
  return count > 0 ? count : 1;
#endif
}

namespace {

#ifdef __APPLE_CC__

/// @brief Default ThreadPool implementation backed by libdispatch on Apple
/// platforms.
///
/// Uses @c dispatch_apply for blocking parallel-for, which is the optimal
/// path on macOS / iOS / etc. — it parks workers on the global concurrent
/// queue and wakes them as needed without us having to manage worker
/// lifetimes.
class ThreadPoolDefault : public ghost::ThreadPool {
 public:
  explicit ThreadPoolDefault(size_t workers) : _workers(workers) {
    if (_workers == 0) _workers = DeviceCPU::getNumberOfCores();
  }

  void parallel(size_t count, std::function<void(size_t, size_t)> fn) override {
    if (count == 0) return;
    if (count == 1 || _nestingDepth > 0) {
      // Nested or single-item: run inline on the calling thread.
      for (size_t i = 0; i < count; i++) fn(i, count);
      return;
    }
    _nestingDepth++;
    dispatch_queue_t queue =
        dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
    // dispatch_apply is blocking and parallelizes across the global queue.
    dispatch_apply(count, queue, ^(size_t i) {
      fn(i, count);
    });
    _nestingDepth--;
  }

  size_t workerCount() const override { return _workers; }

 private:
  size_t _workers;
  // Tracking nesting depth on the calling thread is sufficient because
  // dispatch_apply blocks the caller until all work completes.
  static thread_local size_t _nestingDepth;
};

thread_local size_t ThreadPoolDefault::_nestingDepth = 0;

#else  // !__APPLE_CC__

/// @brief Default ThreadPool implementation backed by @c std::thread workers.
///
/// Workers consume from a shared work queue. @ref parallel pushes @p count
/// items, then blocks on a completion latch until all of them have run. Nested
/// calls (those made from inside a worker callback) run inline on the calling
/// thread to avoid deadlock.
class ThreadPoolDefault : public ghost::ThreadPool {
 public:
  explicit ThreadPoolDefault(size_t workers) {
    if (workers == 0) workers = DeviceCPU::getNumberOfCores();
    if (workers == 0) workers = 1;
    _shouldStop = false;
    for (size_t i = 0; i < workers; i++) {
      _threads.emplace_back(&ThreadPoolDefault::worker, this);
    }
  }

  ~ThreadPoolDefault() override {
    {
      std::lock_guard<std::mutex> lk(_mutex);
      _shouldStop = true;
    }
    _cv.notify_all();
    for (auto& t : _threads) {
      if (t.joinable()) t.join();
    }
  }

  void parallel(size_t count, std::function<void(size_t, size_t)> fn) override {
    if (count == 0) return;
    if (count == 1 || _nestingDepth > 0) {
      for (size_t i = 0; i < count; i++) fn(i, count);
      return;
    }
    _nestingDepth++;
    auto state = std::make_shared<DispatchState>();
    state->fn = std::move(fn);
    state->total = count;
    state->remaining = count;

    {
      std::lock_guard<std::mutex> lk(_mutex);
      for (size_t i = 0; i < count; i++) {
        _work.push({state, i});
      }
    }
    _cv.notify_all();

    // Wait until all items for this dispatch have completed.
    std::unique_lock<std::mutex> lk(state->doneMutex);
    state->doneCv.wait(lk, [&] { return state->remaining == 0; });
    _nestingDepth--;
  }

  size_t workerCount() const override { return _threads.size(); }

 private:
  struct DispatchState {
    std::function<void(size_t, size_t)> fn;
    size_t total = 0;
    std::atomic<size_t> remaining{0};
    std::mutex doneMutex;
    std::condition_variable doneCv;
  };

  struct WorkItem {
    std::shared_ptr<DispatchState> state;
    size_t index;
  };

  void worker() {
    for (;;) {
      WorkItem item;
      {
        std::unique_lock<std::mutex> lk(_mutex);
        _cv.wait(lk, [this] { return _shouldStop || !_work.empty(); });
        if (_shouldStop && _work.empty()) return;
        item = std::move(_work.front());
        _work.pop();
      }
      item.state->fn(item.index, item.state->total);
      // Signal completion
      if (item.state->remaining.fetch_sub(1) == 1) {
        std::lock_guard<std::mutex> lk(item.state->doneMutex);
        item.state->doneCv.notify_one();
      }
    }
  }

  std::vector<std::thread> _threads;
  std::queue<WorkItem> _work;
  std::mutex _mutex;
  std::condition_variable _cv;
  bool _shouldStop;

  static thread_local size_t _nestingDepth;
};

thread_local size_t ThreadPoolDefault::_nestingDepth = 0;

#endif

}  // namespace
}  // namespace implementation

std::shared_ptr<ThreadPool> ThreadPool::createDefault(size_t workers) {
  return std::make_shared<implementation::ThreadPoolDefault>(workers);
}

}  // namespace ghost
