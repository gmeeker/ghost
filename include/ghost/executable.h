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

#ifndef GHOST_EXECUTABLE_H
#define GHOST_EXECUTABLE_H

#include <ghost/command_buffer.h>
#include <ghost/implementation/executable.h>

#include <memory>

namespace ghost {

/// @brief A compiled, reusable command sequence — the instantiated
/// counterpart of a @ref CommandBuffer.
///
/// A @ref CommandBuffer is the cheap, one-shot recording. Calling
/// @ref CommandBuffer::compile turns it into an Executable: a heavier object
/// whose whole purpose is reuse. On backends with native support (e.g. CUDA
/// graphs) it owns an instantiated graph that relaunches cheaply; elsewhere it
/// falls back to command replay (see @ref accelerated). Retention is explicit
/// — the caller holds the Executable (typically on a persistent per-instance
/// scope) and submits it each frame.
///
/// @code
/// ghost::CommandBuffer cb(device);
/// fn(launch, cb)(input, output);
/// ghost::Executable exec = cb.compile();   // instantiate once
/// // ...store exec across frames...
/// cb.reset();
/// fn(launch, cb)(newInput, newOutput);
/// exec.update(cb);                          // rebind buffers
/// exec.submit(stream);                      // relaunch (cheap on native
/// backings)
/// @endcode
///
/// ## Fast path: capture-once / submit-many
///
/// The cheapest pattern is to @c compile once and @c submit repeatedly with
/// **identical** baked resources — no @c update between submits. Then each
/// submit is a single native graph launch / command-buffer enqueue with zero
/// per-op encode cost. This is the supported "pure replay" fast path; favor it
/// (stable arena-backed intermediates, fixed shapes) when you can.
///
/// ## update() cost varies by backend
///
/// - **CUDA**: a dispatch-only graph with unchanged topology is patched in
///   place (per-kernel-node params) with no re-capture — the per-frame
///   sweet spot. Other sequences re-capture and either topology-update or
///   re-instantiate.
/// - **OpenCL**: with @c cl_khr_command_buffer_mutable_dispatch, a
///   dispatch-only command buffer's kernel args are patched in place;
///   otherwise the command buffer is rebuilt.
/// - **Vulkan / DirectX**: @c update re-records the command list (there is no
///   in-place node-param analog). Correct, but not as cheap as a CUDA/OpenCL
///   in-place patch.
/// - Fallback (replay) backends just swap the recorded command list.
///
/// @ref lastUpdatePatched reports whether the most recent @c update kept the
/// cheap in-place path.
///
/// ## Lifetime contract (keepalive)
///
/// An Executable retains the @c CommandBuffer's recorded resources (strong
/// references to the buffers/images/functions and their native handles). This
/// is what keeps the device pointers baked into the graph / command buffer
/// valid for the object's whole lifetime — even if the caller drops its
/// wrappers, or a pooled allocator would otherwise free and reuse a
/// sub-buffer's address. The caller is still responsible for not *mutating* or
/// freeing the underlying storage out from under a retained Executable, and
/// for not calling @c update while a prior @c submit is still in flight (sync
/// the stream first).
///
/// ## Concurrency note (CUDA)
///
/// A CUDA graph is built by single-stream capture, which serializes the
/// recorded ops into a linear dependency chain. Independent dispatches that a
/// concurrent @c CommandBuffer would let overlap are therefore serialized in
/// the graph; a hand-built multi-stream graph could express more concurrency.
/// For the typical dependency-chained inference pipeline this is a non-issue.
class Executable {
 public:
  /// @brief Default-construct a null Executable.
  Executable() = default;

  /// @brief Construct from a backend implementation.
  Executable(std::shared_ptr<implementation::Executable> impl);

  /// @brief Check whether this Executable holds a valid implementation.
  explicit operator bool() const { return _impl != nullptr; }

  /// @brief Execute the compiled sequence on @p stream.
  void submit(const Stream& stream);

  /// @brief Rebind to a freshly recorded command sequence.
  ///
  /// @p cb should hold the same shape of operations the Executable was
  /// compiled from, with updated arguments / buffers. Native backings patch
  /// in place where the topology matches; otherwise they re-instantiate.
  void update(const CommandBuffer& cb);

  /// @brief Whether this Executable is backed by a native accelerated object
  /// (graph / native command buffer) rather than command replay.
  bool accelerated() const;

  /// @brief Whether the most recent @ref update patched the native object in
  /// place (cheap) rather than rebuilding it. Useful for diagnosing when the
  /// per-frame fast path was lost (e.g. a topology change). Meaningless before
  /// the first @ref update.
  bool lastUpdatePatched() const;

  /// @brief Get the backend implementation.
  std::shared_ptr<implementation::Executable> impl() const { return _impl; }

 private:
  std::shared_ptr<implementation::Executable> _impl;
};

}  // namespace ghost

#endif
