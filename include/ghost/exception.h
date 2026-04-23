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

#ifndef GHOST_EXCEPTION_H
#define GHOST_EXCEPTION_H

#include <exception>
#include <stdexcept>

namespace ghost {

/// @brief Exception thrown when an operation is not supported by the current
/// backend.
///
/// Backends throw this when a requested feature (e.g., mapped buffers, image
/// filtering) is not available on the underlying GPU API.
class unsupported_error : public std::runtime_error {
 public:
  unsupported_error() : std::runtime_error("unsupported") {}

  explicit unsupported_error(const char* msg) : std::runtime_error(msg) {}
};

namespace detail {

/// @brief Store an error that occurred in a context where it cannot be thrown
/// (destructors, resource-release paths). The first stashed error per thread
/// is retained; later ones are discarded. Drained by drainErrors().
void stashError(std::exception_ptr e) noexcept;

/// @brief Drain and rethrow any pending stashed error for the current thread.
/// Called at public entry points so that deferred errors surface on the next
/// user-visible Ghost operation.
void drainErrors();

/// @brief RAII helper: drains on construction. Put one at the top of each
/// public Ghost entry point.
struct EntryGuard {
  EntryGuard() { drainErrors(); }
};

}  // namespace detail
}  // namespace ghost

#endif