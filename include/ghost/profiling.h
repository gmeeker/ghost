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

#ifndef GHOST_PROFILING_H
#define GHOST_PROFILING_H

#include <ghost/device.h>
#include <ghost/event.h>
#include <ghost/function.h>

namespace ghost {

/// @brief Dispatch a kernel on a stream and return the elapsed time in seconds.
///
/// Records events before and after the dispatch, synchronizes the stream,
/// and returns the elapsed time. The stream should have profiling enabled
/// (via StreamOptions::profiling) on backends that require it (OpenCL).
///
/// @param s The stream to dispatch on.
/// @param fn The kernel function.
/// @param la Launch configuration.
/// @param args Kernel arguments.
/// @return Elapsed time in seconds.
template <typename... Args>
double timed(Stream& s, Function& fn, const LaunchArgs& la, Args&&... args) {
  Event start = s.record();
  fn(la, s)(std::forward<Args>(args)...);
  Event end = s.record();
  s.sync();
  return Event::elapsed(start, end);
}

}  // namespace ghost

#endif
