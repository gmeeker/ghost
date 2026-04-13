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

#ifndef GHOST_ENCODER_H
#define GHOST_ENCODER_H

#include <ghost/implementation/impl_device.h>

#include <memory>

namespace ghost {

/// @brief Base class for GPU operation targets (streams and command buffers).
///
/// An Encoder represents a destination for GPU operations like kernel
/// dispatches, buffer copies, and image copies. Stream and CommandBuffer
/// both inherit from Encoder, allowing operations to accept either.
///
/// - Stream: serialized execution (every operation waits for the previous).
/// - CommandBuffer: no implicit synchronization (caller manages barriers).
class Encoder {
 public:
  /// @brief Default-construct a null encoder.
  Encoder() = default;

  /// @brief Construct from a backend implementation.
  Encoder(std::shared_ptr<implementation::Encoder> impl);

  /// @brief Check whether this encoder holds a valid implementation.
  explicit operator bool() const { return _impl != nullptr; }

  /// @brief Get the backend implementation (const).
  std::shared_ptr<implementation::Encoder> impl() const { return _impl; }

  /// @brief Get the backend implementation (mutable).
  std::shared_ptr<implementation::Encoder>& impl() { return _impl; }

 protected:
  std::shared_ptr<implementation::Encoder> _impl;
};

}  // namespace ghost

#endif
