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

#ifndef GHOST_THREAD_POOL_H
#define GHOST_THREAD_POOL_H

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>

namespace ghost {

/// @brief A worker pool used by the CPU backend for parallel kernel dispatch
/// and by user code that needs host-side parallel-for primitives.
///
/// Implementations split @c [0, count) across workers and call @p fn for each
/// index. The call is synchronous: @ref parallel returns once every index has
/// completed. Nested @ref parallel calls run inline on the calling thread.
///
/// A default implementation backed by @c std::thread workers (or
/// @c dispatch_apply on macOS) is available via @ref createDefault. Hosts that
/// want Ghost to share their existing executor (e.g. a video editor's job
/// system, TBB, etc.) can subclass @c ThreadPool and inject the instance via
/// @c DeviceCPU::setThreadPool or the @c DeviceCPU constructor.
class ThreadPool {
 public:
  virtual ~ThreadPool() = default;

  /// @brief Run @p fn for each index in @c [0, count) and block until every
  /// invocation has completed.
  ///
  /// @param count Number of work items.
  /// @param fn Callable invoked as @c fn(i, count) for each @c i in
  ///   @c [0, count). Must be safe to call from worker threads concurrently.
  virtual void parallel(size_t count,
                        std::function<void(size_t i, size_t count)> fn) = 0;

  /// @brief The number of worker threads available to this pool.
  ///
  /// A return value of 1 means @ref parallel runs inline on the calling
  /// thread.
  virtual size_t workerCount() const = 0;

  /// @brief Construct Ghost's default thread pool — OpenMP-style fork-join
  /// with static slicing, long-spin then condvar park.
  ///
  /// **Caller participates**: like libgomp's master thread and TBB's
  /// `task_arena` calling thread, the thread that calls @ref parallel
  /// runs one of the static slices alongside the helper threads. With
  /// @c workers=N the pool maintains N-1 helper threads and the caller
  /// fills the Nth slot at dispatch time, so @c workers=N is the right
  /// recipe for an N-core affinity — no separate dispatcher core is
  /// needed. This matches @c omp_set_num_threads(N) /
  /// @c omp_get_max_threads() in libgomp.
  ///
  /// Per-dispatch overhead is ~5–10 µs at @c count==workers in the active
  /// case (helpers stay in spin between dispatches). Truly idle pools
  /// park on a per-worker condvar so the process can sleep. Static
  /// slicing matches OpenMP's @c schedule(static) — each participant
  /// gets a fixed contiguous index range, no work-stealing.
  ///
  /// @param workers Team size, including the calling thread. Pass 0 to
  ///   use @c std::thread::hardware_concurrency(). The pool allocates
  ///   @c workers-1 helper threads.
  /// @param spinDuration How long workers spin before parking on a
  ///   condvar, and how long the parent spins waiting for completion
  ///   before parking. Passing a duration of `-1` (the default) uses
  ///   the @c GHOST_THREAD_SPINCOUNT_US environment variable, falling
  ///   back to ~10 ms if unset. Pass @c 0 for "passive" (park
  ///   immediately) — minimum CPU usage when idle, ~50–100 µs/dispatch
  ///   wake-up cost. Pass a large value (e.g., @c hours(24)) for
  ///   "active" (effectively never park) — matches libgomp's
  ///   @c OMP_WAIT_POLICY=ACTIVE behavior, lowest dispatch latency,
  ///   workers consume full cores while pool is alive.
  ///
  /// @return A shared pointer to a newly constructed default pool.
  static std::shared_ptr<ThreadPool> createDefault(
      size_t workers = 0,
      std::chrono::microseconds spinDuration = std::chrono::microseconds(-1));
};

}  // namespace ghost

#endif
