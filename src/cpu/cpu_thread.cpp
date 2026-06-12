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

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#ifdef WITH_GCD
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

#ifdef WITH_GCD

/// @brief Default ThreadPool implementation backed by libdispatch on Apple
/// platforms.
///
/// Uses @c dispatch_apply for blocking parallel-for, which is the optimal
/// path on macOS / iOS / etc. — it parks workers on the global concurrent
/// queue and wakes them as needed without us having to manage worker
/// lifetimes.
class ThreadPoolDefault : public ghost::ThreadPool {
 public:
  // The spin-duration argument is ignored — libdispatch picks its own
  // wait policy and is already low-overhead on Apple targets. We accept
  // it only to share a constructor signature with the non-Apple impl.
  ThreadPoolDefault(size_t workers, std::chrono::microseconds /*spinDuration*/)
      : _workers(workers) {
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

#if WIN32
/// One entry per physical core: the processor group and the affinity mask of
/// all its logical processors (both SMT siblings on an SMT core).
struct PhysicalCore {
  WORD group;
  KAFFINITY mask;
};

/// Enumerates physical cores via
/// GetLogicalProcessorInformationEx(RelationProcessorCore). Returns an empty
/// vector on failure — callers must treat that as "no placement hints".
std::vector<PhysicalCore> enumeratePhysicalCores() {
  std::vector<PhysicalCore> cores;
  DWORD bytes = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
  if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || bytes == 0) return cores;
  std::vector<uint8_t> buffer(bytes);
  auto* info =
      reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data());
  if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &bytes)) {
    return cores;
  }
  for (DWORD offset = 0; offset < bytes;) {
    auto* entry = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
        buffer.data() + offset);
    if (entry->Relationship == RelationProcessorCore &&
        entry->Processor.GroupCount >= 1) {
      cores.push_back({entry->Processor.GroupMask[0].Group,
                       entry->Processor.GroupMask[0].Mask});
    }
    offset += entry->Size;
  }
  return cores;
}

/// Sets a helper thread's ideal processor to the first logical processor of
/// the given physical core. An ideal processor is a soft scheduler hint —
/// no privilege required, the affinity masks and CPU sets still win,
/// and the scheduler may run the thread elsewhere under contention. The hint
/// is enough to stop the Windows scheduler's default behavior of stacking
/// fork-join workers onto SMT siblings of already-busy cores (measured on a
/// Zen 3 5900: one-per-core placement is ~15% e2e on a conv-heavy workload
/// dispatched at workers == physical core count).
void setIdealProcessor(std::thread& t, const PhysicalCore& core) {
  KAFFINITY mask = core.mask;
  BYTE number = 0;
  while (mask > 1 && (mask & 1) == 0) {
    mask >>= 1;
    ++number;
  }
  PROCESSOR_NUMBER pn = {};
  pn.Group = core.group;
  pn.Number = number;
  SetThreadIdealProcessorEx(static_cast<HANDLE>(t.native_handle()), &pn,
                            nullptr);
}
#endif  // WIN32

/// @brief Default ThreadPool implementation — OpenMP-style fork-join.
///
/// Modeled on libgomp's `parallel for` and TBB's `task_arena`:
///
/// - **Calling thread participates.** Like libgomp's master thread and
///   TBB's `task_arena` calling thread, the thread that calls
///   @ref parallel does its own share of the work alongside helper
///   threads. With `workers=N`, the pool has N-1 helper threads plus
///   the caller — N total participants. This means `workers=N` matches
///   N cores cleanly: no separate "dispatcher" core wasted, no need
///   for the host to leave a spare core in their affinity mask.
///
/// - **Static slicing** by default (`schedule(static)` analog). Each
///   participant (caller + helpers) gets a fixed contiguous range of
///   indices — no work-stealing, no `fetch_add` cache contention on
///   the hot path. Balanced ML kernels (per-row, per-tile, per-channel)
///   hit this cell.
///
/// - **Long spin window** (per-pool, ~10 ms by default, configurable via
///   the `spinDuration` ctor parameter or the `GHOST_THREAD_SPINCOUNT_US`
///   env var) before workers park. Comparable to libgomp's
///   `GOMP_SPINCOUNT` — back-to-back dispatches in an active workload
///   arrive within the window so workers never enter the kernel for
///   fork/join. Truly idle pools still park, so the process can sleep
///   when not under load. Pass `0` for `OMP_WAIT_POLICY=PASSIVE`-style
///   immediate parking; pass a very large value for `ACTIVE`-style
///   never-park behavior.
///
/// - **Per-worker spin slot** (`assignedEpoch` + `mu` + `cv`). Parent
///   updates only participating workers' slots — non-participating
///   workers stay parked or spinning idle and never touch `Job`. This
///   avoids the thundering-herd cost of a single shared condvar.
///
/// - **isParked-flag-gated notify**. Parent skips `cv.notify_one` for
///   workers that are spinning (the common case in active workloads),
///   reducing per-dispatch overhead from N syscalls to N short
///   uncontended lock acquisitions. Workers in spin observe the new
///   `assignedEpoch` directly via cache coherence; workers that have
///   parked still get woken correctly.
///
/// - **Heap-allocated Job + epoch tag** (the #43 correctness fix).
///   Parent owns Job via `std::unique_ptr`; the per-job `epoch` field
///   lets workers detect torn `(job, epoch)` reads across parent's
///   publish without locking.
///
/// - **Cooperative spin yield**. After a brief tight-spin warmup
///   (~1024 iters / ~10 µs), workers periodically call
///   `std::this_thread::yield()` while spinning. This is the libgomp
///   pattern: tight cpuPause inside the warmup window covers
///   back-to-back dispatch (so a worker doesn't lose its core on a
///   <10 µs gap), but past warmup the yield lets the OS schedule
///   co-pinned threads — most notably the main thread, when the host
///   pins `workers=N` to `N` cores. Without the yield, CFS waits for
///   a worker's 3 ms time-slice to expire before scheduling main, so
///   parallel() dispatch latency floors at ~3 ms in that affinity.
///
/// Nested `parallel` calls (from inside `fn`, on either parent or worker)
/// run inline on the calling thread.
class ThreadPoolDefault : public ghost::ThreadPool {
 public:
  explicit ThreadPoolDefault(size_t workers,
                             std::chrono::microseconds spinDuration) {
    if (workers == 0) workers = DeviceCPU::getNumberOfCores();
    if (workers == 0) workers = 1;
    // `workers` is the team size (caller + helpers). Helpers count is
    // workers - 1. With workers=1, there are no helpers — every
    // parallel() call runs entirely on the caller.
    const size_t helpers = workers - 1;
    _spinDuration = resolveSpinDuration(spinDuration);
    _shouldStop.store(false, std::memory_order_relaxed);
    _perWorker = std::vector<PerWorker>(helpers);
    for (size_t i = 0; i < helpers; i++) {
      _threads.emplace_back(&ThreadPoolDefault::worker, this, i);
    }
#if WIN32
    // Spread helpers one-per-physical-core via ideal-processor hints.
    // Unlike Linux's CFS, the Windows scheduler will happily co-schedule
    // helper threads onto SMT siblings of busy cores; for a fork-join team
    // sized to the physical core count that serializes pairs of slices and
    // shows up as negative scaling past ~half the team size. Helper i is
    // hinted to core (i + 1) % numCores, leaving core 0 as the natural home
    // of the calling thread (participant 0). Soft hints only: the
    // process affinity / CPU sets always take precedence.
    const auto cores = enumeratePhysicalCores();
    if (cores.size() > 1) {
      for (size_t i = 0; i < _threads.size(); i++) {
        setIdealProcessor(_threads[i], cores[(i + 1) % cores.size()]);
      }
    }
#endif
  }

  ~ThreadPoolDefault() override {
    _shouldStop.store(true, std::memory_order_relaxed);
    // Wake every worker — each has its own slot.
    for (auto& w : _perWorker) {
      {
        std::lock_guard<std::mutex> lk(w.mu);
        w.assignedEpoch.fetch_add(1, std::memory_order_release);
      }
      w.cv.notify_one();
    }
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

    // Total participants = caller + at most _threads.size() helpers,
    // capped at count (no point firing helpers we have no shards for).
    const size_t totalParticipants =
        std::min<size_t>(count, _threads.size() + 1);
    // Helpers we need to wake. Caller is participant 0 and runs its
    // own slice inline — it doesn't ack `done`.
    const size_t helpersToFire = totalParticipants - 1;

    auto job = std::make_unique<Job>();
    job->fn = std::move(fn);
    job->count = count;
    job->totalParticipants = totalParticipants;

    const uint64_t newEpoch =
        _globalEpoch.fetch_add(1, std::memory_order_relaxed) + 1;
    job->epoch = newEpoch;

    // Publish Job ptr with release. Helpers acquire-load _job and verify
    // job->epoch matches their assigned epoch (catches torn reads across
    // the two-step publish below).
    _job.store(job.get(), std::memory_order_release);

    // Bump assignedEpoch for the helpers we want firing. Helpers in
    // their spin window observe this directly via cache coherence — no
    // notify needed for them.
    for (size_t i = 0; i < helpersToFire; ++i) {
      _perWorker[i].assignedEpoch.store(newEpoch, std::memory_order_release);
    }

    // For each helper: take its per-slot mutex briefly. If the helper
    // is parked (isParked == true), notify it. If the helper is
    // spinning (isParked == false), skip notify — it will see the new
    // assignedEpoch from its spin loop within nanoseconds.
    //
    // Taking the mutex (rather than just atomic-loading isParked) is
    // what makes this race-free: a helper about to park sets
    // isParked=true *under* mu and only then evaluates its cv.wait
    // predicate. By holding mu here, parent serializes against that
    // transition — either the helper has already entered cv.wait
    // (parent sees isParked=true and notifies) or the helper hasn't
    // taken mu yet and will see the new assignedEpoch on its predicate
    // check (which happens under mu, sequenced after parent's release).
    for (size_t i = 0; i < helpersToFire; ++i) {
      auto& w = _perWorker[i];
      std::lock_guard<std::mutex> lk(w.mu);
      if (w.isParked.load(std::memory_order_relaxed)) {
        w.cv.notify_one();
      }
    }

    // Caller is participant 0 — run its slice inline alongside the
    // helpers' slices.
    {
      const size_t total = count;
      const size_t shards = totalParticipants;
      const size_t base = 0;  // participant 0
      const size_t size = (total / shards) + (0 < total % shards ? 1u : 0u);
      auto& callerFn = job->fn;
      for (size_t i = 0; i < size; ++i) {
        callerFn(base + i, total);
      }
    }

    // Wait for helpers to ack `done`. With helpersToFire == 0 (e.g.,
    // workers == 1), there's nothing to wait for.
    if (helpersToFire > 0) {
      if (!spinWaitDone(job->done, helpersToFire)) {
        std::unique_lock<std::mutex> lk(_doneMutex);
        _doneCv.wait(lk, [&] {
          return job->done.load(std::memory_order_acquire) >= helpersToFire;
        });
      }
    }

    _job.store(nullptr, std::memory_order_release);
    _nestingDepth--;
    // job (unique_ptr) destructor runs here, freeing Job's memory.
  }

  // Team size = caller + helpers. Matches libgomp's
  // omp_get_max_threads() and TBB's task_arena's effective concurrency.
  size_t workerCount() const override { return _threads.size() + 1; }

 private:
  struct Job {
    // Owned by-value so the std::function's lifetime is tied to the Job's
    // heap allocation, not parent's stack frame. Parent's stack frame
    // gets reused across calls, so a stack-pointer would silently invoke
    // the next dispatch's lambda from a slow worker.
    std::function<void(size_t, size_t)> fn;
    size_t count = 0;
    // Total participants = caller (participant 0) + helpers
    // (participants 1..totalParticipants-1). Used by helpers to compute
    // their static slice and by the caller to size its wait predicate
    // (helpersToFire = totalParticipants - 1).
    size_t totalParticipants = 0;
    // Generation tag — equals the assigned epoch at publish time. Lets
    // workers detect torn (job, epoch) pairs across parent's two-step
    // publish without locking. Original source of bug #43.
    uint64_t epoch = 0;
    std::atomic<size_t> done{0};
  };

  // Per-worker slot, cache-line aligned to avoid false sharing.
  struct alignas(64) PerWorker {
    std::atomic<uint64_t> assignedEpoch{0};
    // True iff the worker is currently in cv.wait (or about to enter
    // it). Set under `mu` right before the predicate check; cleared
    // under `mu` after wait returns. Parent reads under `mu` to decide
    // whether notify_one is needed.
    std::atomic<bool> isParked{false};
    std::mutex mu;
    std::condition_variable cv;
  };

  // Resolves the per-pool spin duration. If the caller passed a
  // negative duration (the public API's "default" sentinel), check
  // `GHOST_THREAD_SPINCOUNT_US` and fall back to ~10 ms.
  static std::chrono::microseconds resolveSpinDuration(
      std::chrono::microseconds explicitValue) {
    if (explicitValue >= std::chrono::microseconds(0)) {
      return explicitValue;
    }
    const char* env = std::getenv("GHOST_THREAD_SPINCOUNT_US");
    if (env && *env) {
      char* end = nullptr;
      long long v = std::strtoll(env, &end, 10);
      if (end != env && v >= 0) {
        return std::chrono::microseconds(v);
      }
    }
    // Default ~10 ms — covers any realistic inter-dispatch gap in an
    // active ML pipeline while still letting the process sleep on
    // sustained idle. Comparable to libgomp's GOMP_SPINCOUNT default
    // (~30 ms) but smaller to be a touch friendlier to general-purpose
    // hosts that didn't ask for OMP-active behavior.
    return std::chrono::microseconds(10000);
  }

  // Spin window before parking; set per pool at construction time.
  std::chrono::microseconds _spinDuration;

  // Returns true if `done` reached `target` within the spin window.
  bool spinWaitDone(std::atomic<size_t>& done, size_t target) const {
    if (done.load(std::memory_order_acquire) >= target) return true;
    const auto deadline = std::chrono::steady_clock::now() + _spinDuration;
    int i = 0;
    for (;;) {
      cpuPause();
      if (done.load(std::memory_order_acquire) >= target) return true;
      if ((++i & 31) == 0 && std::chrono::steady_clock::now() >= deadline) {
        return done.load(std::memory_order_acquire) >= target;
      }
    }
  }

  void worker(size_t workerId) {
    auto& slot = _perWorker[workerId];
    uint64_t lastEpoch = 0;
    for (;;) {
      uint64_t epoch = slot.assignedEpoch.load(std::memory_order_acquire);
      if (epoch == lastEpoch) {
        // Spin window — long enough to bridge active dispatch streams
        // without parking.
        //
        // Two phases:
        //  1. Tight phase (first kSpinTightIters iterations, ~10 µs):
        //     pure cpuPause. Covers the back-to-back dispatch case where
        //     the next parallel() call is microseconds away — yielding
        //     here would push that dispatch off the worker's core onto
        //     the kernel's runqueue and pessimise the common active
        //     workload.
        //  2. Yielding phase (after warmup, until deadline): cpuPause +
        //     periodic std::this_thread::yield(). Required when workers
        //     are pinned to the same core set as the main thread (the
        //     usual `taskset -c 0..N-1` recipe with N workers): without
        //     yielding, 100%-spinning workers prevent CFS from
        //     scheduling the co-pinned main thread until its 3 ms
        //     time-slice expires, blocking parallel() dispatch on a
        //     full CFS tick. libgomp solves this the same way.
        constexpr int kSpinTightIters = 1024;
        constexpr int kYieldEveryIters = 512;
        const auto deadline = std::chrono::steady_clock::now() + _spinDuration;
        for (int i = 0;; ++i) {
          cpuPause();
          epoch = slot.assignedEpoch.load(std::memory_order_acquire);
          if (epoch != lastEpoch) break;
          if (i > kSpinTightIters && (i & (kYieldEveryIters - 1)) == 0) {
            // Cooperative yield. When no thread is waiting on the
            // worker's core, sched_yield returns immediately
            // (~100–200 ns); when main is waiting, this is what lets
            // CFS schedule it without a 3 ms time-slice eviction.
            std::this_thread::yield();
          }
          if ((i & 31) == 0 && std::chrono::steady_clock::now() >= deadline) {
            // Park. isParked = true is set under mu so parent's notify
            // gate (also under mu) sees the up-to-date value.
            std::unique_lock<std::mutex> lk(slot.mu);
            slot.isParked.store(true, std::memory_order_relaxed);
            slot.cv.wait(lk, [&] {
              return _shouldStop.load(std::memory_order_relaxed) ||
                     slot.assignedEpoch.load(std::memory_order_acquire) !=
                         lastEpoch;
            });
            slot.isParked.store(false, std::memory_order_relaxed);
            epoch = slot.assignedEpoch.load(std::memory_order_acquire);
            break;
          }
        }
      }
      if (_shouldStop.load(std::memory_order_relaxed)) return;

      Job* job = _job.load(std::memory_order_acquire);
      if (!job || job->epoch != epoch) {
        cpuPause();
        continue;
      }
      lastEpoch = epoch;

      // Helper `workerId` is participant `workerId + 1` (caller is
      // participant 0). If we fall outside the participating range
      // (small count, fewer than _threads.size() helpers fired), skip.
      const size_t partIdx = workerId + 1;
      if (partIdx >= job->totalParticipants) continue;

      // Static slice — fixed contiguous range for this participant,
      // computed from immutable Job fields. Matches OpenMP's
      // schedule(static). No fetch_add(next) cache-line bouncing.
      const size_t total = job->count;
      const size_t shards = job->totalParticipants;
      const size_t base = (total / shards) * partIdx +
                          std::min<size_t>(partIdx, total % shards);
      const size_t size =
          (total / shards) + (partIdx < total % shards ? 1u : 0u);

      _nestingDepth++;
      for (size_t i = 0; i < size; ++i) {
        job->fn(base + i, total);
      }
      _nestingDepth--;

      // Snapshot helpersToFire BEFORE the fetch_add. Reading it after
      // is UAF: caller's wait predicate (done >= helpersToFire)
      // synchronizes-with our fetch_add, so once we increment, caller
      // may immediately observe completion and destroy Job.
      const size_t helpersToFire = job->totalParticipants - 1;
      if (job->done.fetch_add(1, std::memory_order_acq_rel) + 1 ==
          helpersToFire) {
        std::lock_guard<std::mutex> lk(_doneMutex);
        _doneCv.notify_all();
      }
    }
  }

  static void cpuPause() {
#if defined(_MSC_VER) && (defined(_M_ARM64) || defined(_M_ARM64EC))
    // Checked before _M_X64: ARM64EC defines both _M_ARM64EC and _M_X64,
    // and should use the native yield rather than emulated _mm_pause.
    __yield();
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    // MSVC defines _M_X64/_M_IX86, not __x86_64__ — before this branch
    // existed, MSVC builds fell through to std::this_thread::yield(), i.e.
    // a SwitchToThread syscall on EVERY spin iteration. That turns the
    // "tight spin" phases (worker wait and parent spinWaitDone) into a
    // syscall storm and invites the scheduler to migrate spinning workers.
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
    asm volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
  }

  std::vector<std::thread> _threads;
  std::vector<PerWorker> _perWorker;
  std::atomic<uint64_t> _globalEpoch{0};
  // Non-owning pointer to the current dispatch's Job. Parent owns the Job
  // (heap-allocated unique_ptr local to parallel()). Workers acquire-load
  // _job and verify job->epoch matches their assigned epoch.
  std::atomic<Job*> _job{nullptr};
  std::mutex _dispatchMutex;
  // Pool-owned so that a worker mid-notify cannot touch a destroyed object
  // after the parent returns from parallel().
  std::mutex _doneMutex;
  std::condition_variable _doneCv;
  std::atomic<bool> _shouldStop;

  static thread_local size_t _nestingDepth;
};

thread_local size_t ThreadPoolDefault::_nestingDepth = 0;

#endif

}  // namespace
}  // namespace implementation

std::shared_ptr<ThreadPool> ThreadPool::createDefault(
    size_t workers, std::chrono::microseconds spinDuration) {
  return std::make_shared<implementation::ThreadPoolDefault>(workers,
                                                             spinDuration);
}

}  // namespace ghost
