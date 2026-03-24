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

#ifndef GHOST_EVENT_H
#define GHOST_EVENT_H

#include <memory>

namespace ghost {
namespace implementation {
class Event;
}

/// @brief A synchronization primitive for tracking GPU operation completion.
///
/// Events are recorded on a Stream via Stream::record() and can be used for:
/// - CPU-side blocking waits (wait())
/// - Non-blocking completion queries (isComplete())
/// - Cross-stream GPU-side synchronization (Stream::waitForEvent())
/// - Timing measurements (elapsed())
class Event {
 public:
  /// @brief Construct from a backend implementation.
  /// @param impl Shared pointer to the backend-specific event implementation.
  Event(std::shared_ptr<implementation::Event> impl);

  /// @brief Get the backend implementation (const).
  std::shared_ptr<implementation::Event> impl() const { return _impl; }

  /// @brief Get the backend implementation (mutable).
  std::shared_ptr<implementation::Event>& impl() { return _impl; }

  /// @brief Block the calling CPU thread until this event has completed.
  void wait();

  /// @brief Query whether this event has completed without blocking.
  /// @return @c true if the event has completed, @c false otherwise.
  bool isComplete() const;

  /// @brief Measure the elapsed time in seconds between two events.
  /// @param start The earlier event.
  /// @param end The later event.
  /// @return Elapsed time in seconds, or 0 if timing is not supported.
  static double elapsed(const Event& start, const Event& end);

 private:
  std::shared_ptr<implementation::Event> _impl;
};
}  // namespace ghost

#endif
