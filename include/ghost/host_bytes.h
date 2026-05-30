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

#ifndef GHOST_HOST_BYTES_H
#define GHOST_HOST_BYTES_H

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

namespace ghost {

/// @brief A host-memory region with optional ownership semantics, used as the
/// source for host-to-device uploads and as the destination for device-to-host
/// readbacks.
///
/// Construct via the @c borrow / @c adopt / @c wrap factories. The owner is
/// type-erased through @c std::shared_ptr's deleter slot, so any allocation
/// strategy works: @c new[] / @c malloc, @c cuMemAllocHost, @c mmap, a foreign
/// library's allocator, an arena, etc.
///
/// Two contracts:
///
/// - @c borrow (owner is null): the caller keeps @c data valid until the
///   operation it is passed to has finished using it. For uploads, the
///   receiving copy must consume @c data before returning. For readbacks
///   onto a @c Stream, the call is synchronous; onto a @c CommandBuffer,
///   the caller keeps @c data valid until @c stream.sync() returns.
/// - @c adopt / @c wrap (owner is non-null): ownership is shared with Ghost.
///   The deleter runs when the last @c shared_ptr reference drops, which on
///   backends that lifetime-extend the DMA (OpenCL, CUDA-pinned) is after the
///   transfer has completed. Backends that consume @c data synchronously
///   drop their reference at end of call. Either way, the caller may drop
///   their own reference immediately after submitting the work; the memory
///   stays alive long enough for the DMA.
///
/// The pointer type is non-const so the same handle can serve both
/// directions. Callers passing read-only source memory may construct via
/// @c const_cast — only an upload path reads from it, and only a readback
/// path writes to it.
class HostBytes {
 public:
  HostBytes() = default;

  /// @brief Caller-managed lifetime.
  static HostBytes borrow(void* data) {
    HostBytes h;
    h._data = data;
    return h;
  }

  /// @brief Ghost takes ownership. @p deleter(data) runs when the last
  /// reference drops (after any pending async DMA has completed). Suitable
  /// for @c malloc / @c new[], @c cuMemAllocHost, @c mmap, foreign-library
  /// buffers.
  static HostBytes adopt(void* data, std::function<void(void*)> deleter) {
    HostBytes h;
    h._owner = std::shared_ptr<void>(data, std::move(deleter));
    h._data = data;
    return h;
  }

  /// @brief Hand Ghost a shared ref to an existing owned region. @p data
  /// defaults to @c owner.get() but may point into the middle of a larger
  /// arena (the deleter sees @c owner.get(), not @c data).
  static HostBytes wrap(std::shared_ptr<void> owner, void* data = nullptr) {
    HostBytes h;
    h._owner = std::move(owner);
    h._data = data ? data : h._owner.get();
    return h;
  }

  /// @brief Pointer to the host bytes. Upload paths read; readback paths
  /// write. May or may not equal @c owner().get() depending on construction.
  void* data() const noexcept { return _data; }

  /// @brief Lifetime token. Non-null for @c adopt / @c wrap; null for
  /// @c borrow.
  const std::shared_ptr<void>& owner() const noexcept { return _owner; }

  /// @brief Whether ownership was transferred to Ghost (i.e. backends may
  /// lifetime-extend the bytes past the call return).
  bool ownsBytes() const noexcept { return static_cast<bool>(_owner); }

 private:
  std::shared_ptr<void> _owner;
  void* _data = nullptr;
};

}  // namespace ghost

#endif
