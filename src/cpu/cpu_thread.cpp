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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
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

/// @brief Default ThreadPool implementation backed by @c std::thread workers
/// using a spin-then-park barrier.
///
/// Each @ref parallel call publishes a static slice across @c min(count, W)
/// workers and bumps a global epoch counter. Workers spin on the epoch for
/// @c kSpinDuration after their previous task; if no new dispatch arrives in
/// that window, they park on a condvar. The master spins on the remaining
/// counter for the same window before parking on a per-dispatch condvar.
///
/// This avoids the futex round-trip per dispatch that plagued the original
/// queue+notify_all design (~20 µs floor on Linux), making back-to-back
/// fine-grained dispatches — the dominant ML-kernel pattern — competitive
/// with @c libgomp's @c parallel @c for barrier (~1–2 µs/dispatch).
///
/// Nested @ref parallel calls (from inside @c fn, on either the master or a
/// worker thread) run inline on the calling thread.
class ThreadPoolDefault : public ghost::ThreadPool {
 public:
  explicit ThreadPoolDefault(size_t workers) {
    if (workers == 0) workers = DeviceCPU::getNumberOfCores();
    if (workers == 0) workers = 1;
    _shouldStop.store(false, std::memory_order_relaxed);
    _epoch.store(0, std::memory_order_relaxed);
    _state.store(nullptr, std::memory_order_relaxed);
    for (size_t i = 0; i < workers; i++) {
      _threads.emplace_back(&ThreadPoolDefault::worker, this, i);
    }
  }

  ~ThreadPoolDefault() override {
    {
      std::lock_guard<std::mutex> lk(_mutex);
      _shouldStop.store(true, std::memory_order_relaxed);
      // Release pairs with the worker's acquire load of _epoch and publishes
      // _shouldStop along with the epoch bump.
      _epoch.fetch_add(1, std::memory_order_release);
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
    // Serialize concurrent top-level dispatches from different threads.
    std::lock_guard<std::mutex> dispatchLk(_dispatchMutex);

    const size_t W = _threads.size();
    const size_t numShards = std::min<size_t>(count, W);

    DispatchState state;
    state.fn = &fn;
    state.total = count;
    state.numShards = numShards;
    state.remaining.store(numShards, std::memory_order_relaxed);

    // Publish state, then bump epoch with release. Workers see _state and
    // *state through the release-acquire pair on _epoch.
    _state.store(&state, std::memory_order_relaxed);
    _epoch.fetch_add(1, std::memory_order_release);
    // Wake any parked workers. Workers still in their spin window observe
    // the new epoch directly without paying the cv round-trip.
    _cv.notify_all();

    // Wait for all shards to finish: spin briefly, then park.
    if (!spinWaitDone(state.remaining)) {
      std::unique_lock<std::mutex> lk(state.doneMutex);
      state.doneCv.wait(lk, [&] {
        return state.remaining.load(std::memory_order_acquire) == 0;
      });
    }

    _state.store(nullptr, std::memory_order_relaxed);
    _nestingDepth--;
  }

  size_t workerCount() const override { return _threads.size(); }

 private:
  struct DispatchState {
    std::function<void(size_t, size_t)>* fn = nullptr;
    size_t total = 0;
    size_t numShards = 0;
    std::atomic<size_t> remaining{0};
    std::mutex doneMutex;
    std::condition_variable doneCv;
  };

  // Spin window before parking. Sized to cover back-to-back fine-grained
  // dispatches (for example, dozens of dispatches at <50 µs each).
  // Calibrated by wall clock so it survives differences in `pause` latency
  // across CPU vendors (Skylake+ pause is ~140 cycles, older x86 ~10).
  static constexpr auto kSpinDuration = std::chrono::microseconds(200);

  // Returns true if `remaining` reached 0 within the spin window.
  static bool spinWaitDone(std::atomic<size_t>& remaining) {
    if (remaining.load(std::memory_order_acquire) == 0) return true;
    const auto deadline = std::chrono::steady_clock::now() + kSpinDuration;
    int i = 0;
    for (;;) {
      cpuPause();
      if (remaining.load(std::memory_order_acquire) == 0) return true;
      if ((++i & 31) == 0 && std::chrono::steady_clock::now() >= deadline) {
        return remaining.load(std::memory_order_acquire) == 0;
      }
    }
  }

  void worker(size_t workerId) {
    uint64_t lastEpoch = 0;
    for (;;) {
      uint64_t epoch = _epoch.load(std::memory_order_acquire);
      if (epoch == lastEpoch) {
        const auto deadline = std::chrono::steady_clock::now() + kSpinDuration;
        for (int i = 0;; ++i) {
          cpuPause();
          epoch = _epoch.load(std::memory_order_acquire);
          if (epoch != lastEpoch) break;
          if ((i & 31) == 0 && std::chrono::steady_clock::now() >= deadline) {
            std::unique_lock<std::mutex> lk(_mutex);
            _cv.wait(lk, [&] {
              return _shouldStop.load(std::memory_order_relaxed) ||
                     _epoch.load(std::memory_order_acquire) != lastEpoch;
            });
            epoch = _epoch.load(std::memory_order_acquire);
            break;
          }
        }
      }
      lastEpoch = epoch;
      // _shouldStop is published by the destructor via the release on _epoch
      // we just acquired, so a relaxed load here is safe.
      if (_shouldStop.load(std::memory_order_relaxed)) return;

      DispatchState* state = _state.load(std::memory_order_relaxed);
      if (!state) continue;
      if (workerId >= state->numShards) continue;

      const size_t total = state->total;
      const size_t shards = state->numShards;
      const size_t base = (total / shards) * workerId +
                          std::min<size_t>(workerId, total % shards);
      const size_t size =
          (total / shards) + (workerId < total % shards ? 1u : 0u);

      // Mark this thread as in a parallel region so any nested parallel()
      // call from inside fn runs inline (avoids re-entering the pool).
      _nestingDepth++;
      for (size_t i = 0; i < size; ++i) {
        (*state->fn)(base + i, total);
      }
      _nestingDepth--;

      if (state->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lk(state->doneMutex);
        state->doneCv.notify_one();
      }
    }
  }

  static void cpuPause() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
    asm volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
  }

  std::vector<std::thread> _threads;
  std::atomic<uint64_t> _epoch;
  std::atomic<DispatchState*> _state;
  std::mutex _dispatchMutex;
  std::mutex _mutex;
  std::condition_variable _cv;
  std::atomic<bool> _shouldStop;

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
