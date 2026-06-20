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

#ifndef GHOST_IMPLEMENTATION_EXECUTABLE_H
#define GHOST_IMPLEMENTATION_EXECUTABLE_H

#include <ghost/implementation/recorded_command_buffer.h>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace ghost {
class Stream;

namespace implementation {

/// @brief Abstract backend interface for a compiled, reusable command
/// sequence (the instantiated counterpart of a @ref CommandBuffer).
///
/// Where a @ref CommandBuffer is the cheap, one-shot recording (the
/// "template"), an Executable is the heavier, retained object a caller holds
/// across many submits. Native backends own a real graph / command-buffer
/// here (e.g. a CUDA @c CUgraphExec); the @ref RecordedExecutable fallback
/// simply replays the recorded commands so the API is uniform on backends
/// (or command sequences) that can't be accelerated.
class Executable {
 protected:
  Executable() {}

  Executable(const Executable&) = delete;

  Executable& operator=(const Executable&) = delete;

 public:
  virtual ~Executable() {}

  /// @brief Execute the compiled sequence on @p stream. Cheap to call
  /// repeatedly on a native backing (one graph launch); equivalent to a
  /// @c CommandBuffer::submit on the fallback.
  virtual void submit(const ghost::Stream& stream) = 0;

  /// @brief Rebind to a freshly recorded command sequence (same shape, new
  /// arguments / buffers). Native backings patch their graph in place where
  /// possible; the fallback swaps its recorded commands.
  virtual void update(const std::vector<Command>& commands) = 0;

  /// @brief Whether this Executable is backed by a native accelerated object
  /// (graph / native command buffer) rather than command replay. Lets callers
  /// that gain nothing from the fallback decide not to retain it.
  virtual bool accelerated() const { return false; }

  /// @brief Whether the most recent @ref update patched the existing native
  /// object in place (cheap) rather than rebuilding it. Lets callers see when
  /// the fast path was lost (e.g. a topology change forced a re-instantiate).
  /// Meaningless before the first @ref update; defaults to false.
  virtual bool lastUpdatePatched() const { return false; }
};

/// @brief Replay-backed @ref Executable used as the universal fallback.
///
/// Holds an independent snapshot of the recorded commands in a command
/// buffer of the originating backend's concrete type (via
/// @ref RecordedCommandBuffer::cloneEmpty), so backend-specific replay hooks
/// (e.g. CUDA's @c encodeNative) still work. Submitting replays the commands
/// onto the target stream exactly as @c CommandBuffer::submit would — correct,
/// but with no launch-overhead reduction (@ref accelerated returns false).
class RecordedExecutable : public Executable {
 public:
  explicit RecordedExecutable(std::shared_ptr<RecordedCommandBuffer> cb)
      : _cb(std::move(cb)) {}

  void submit(const ghost::Stream& stream) override { _cb->submit(stream); }

  void update(const std::vector<Command>& commands) override {
    _cb->commands = commands;
  }

 private:
  std::shared_ptr<RecordedCommandBuffer> _cb;
};

/// @brief An @ref Executable for a @ref CommandBuffer with a marked compiled
/// region (see @c CommandBuffer::beginCompiledRegion). Replays the commands
/// before the region, launches the natively-compiled region, then replays the
/// commands after it — all on the target stream, in order.
///
/// This lets a sequence that brackets compute with non-capturable ops (host
/// uploads/downloads, events) accelerate just the contiguous compute span
/// instead of dropping the whole sequence to replay.
class SegmentedExecutable : public Executable {
 public:
  SegmentedExecutable(std::shared_ptr<Executable> pre,
                      std::shared_ptr<Executable> region,
                      std::shared_ptr<Executable> post, size_t regionBegin,
                      size_t regionEnd)
      : _pre(std::move(pre)),
        _region(std::move(region)),
        _post(std::move(post)),
        _regionBegin(regionBegin),
        _regionEnd(regionEnd) {}

  void submit(const ghost::Stream& stream) override {
    _pre->submit(stream);
    _region->submit(stream);
    _post->submit(stream);
  }

  void update(const std::vector<Command>& commands) override {
    // Same-shape update: re-split at the original region boundaries. (The
    // contract for update() is an unchanged command shape; clamp defensively.)
    size_t b = std::min(_regionBegin, commands.size());
    size_t e = std::min(std::max(_regionEnd, b), commands.size());
    _pre->update({commands.begin(), commands.begin() + b});
    _region->update({commands.begin() + b, commands.begin() + e});
    _post->update({commands.begin() + e, commands.end()});
  }

  // The region is the only part that can be native; report its status.
  bool accelerated() const override { return _region->accelerated(); }

  bool lastUpdatePatched() const override {
    return _region->lastUpdatePatched();
  }

 private:
  std::shared_ptr<Executable> _pre, _region, _post;
  size_t _regionBegin, _regionEnd;
};

}  // namespace implementation
}  // namespace ghost

#endif
