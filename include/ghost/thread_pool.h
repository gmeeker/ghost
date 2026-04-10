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

  /// @brief Construct Ghost's default thread pool.
  ///
  /// @param workers Number of worker threads. Pass 0 to use
  ///   @c std::thread::hardware_concurrency().
  /// @return A shared pointer to a newly constructed default pool.
  static std::shared_ptr<ThreadPool> createDefault(size_t workers = 0);
};

}  // namespace ghost

#endif
