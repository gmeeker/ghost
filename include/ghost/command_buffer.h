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

#ifndef GHOST_COMMAND_BUFFER_H
#define GHOST_COMMAND_BUFFER_H

#include <ghost/device.h>
#include <ghost/function.h>

#include <memory>
#include <vector>

namespace ghost {

namespace implementation {

/// @brief Abstract backend interface for a command buffer.
///
/// The default implementation records commands and replays them on submit.
/// Backends may override with native command buffer support.
class CommandBuffer {
 protected:
  CommandBuffer() {}

  CommandBuffer(const CommandBuffer&) = delete;

  virtual ~CommandBuffer() {}

  CommandBuffer& operator=(const CommandBuffer&) = delete;

 public:
  virtual void dispatch(std::shared_ptr<Function> function,
                        const LaunchArgs& launchArgs,
                        const std::vector<Attribute>& args) = 0;

  virtual void dispatchIndirect(std::shared_ptr<Function> function,
                                std::shared_ptr<Buffer> indirectBuffer,
                                size_t indirectOffset,
                                const std::vector<Attribute>& args) = 0;

  virtual void copyBuffer(std::shared_ptr<Buffer> dst,
                          std::shared_ptr<Buffer> src, size_t srcOffset,
                          size_t dstOffset, size_t bytes) = 0;

  virtual void fillBuffer(std::shared_ptr<Buffer> dst, size_t offset,
                          size_t size, uint8_t value) = 0;

  virtual void addBarrier() = 0;

  virtual void addWaitForEvent(std::shared_ptr<Event> e) = 0;

  virtual std::shared_ptr<Event> addRecordEvent(
      const ghost::Stream& stream) = 0;

  virtual void submit(const ghost::Stream& stream) = 0;

  virtual void reset() = 0;

  static std::shared_ptr<CommandBuffer> createDefault();
};

}  // namespace implementation

/// @brief A deferred execution buffer that records GPU operations and submits
/// them as a batch.
///
/// CommandBuffer records kernel dispatches, buffer copies, fills, and
/// synchronization operations. Recorded operations are executed when
/// submit() is called. This enables batching of GPU work for improved
/// performance.
///
/// @code
/// ghost::CommandBuffer cb(device);
/// cb.dispatch(fn, launchArgs, buffer, 42.0f);
/// cb.copyBuffer(dst, src, bytes);
/// cb.submit(stream);
/// @endcode
///
/// On backends with native command buffer support (Metal), recorded
/// operations may be encoded directly into a hardware command buffer.
/// On other backends (OpenCL, CUDA, CPU), operations are recorded and
/// replayed onto a stream at submit time.
class CommandBuffer {
 public:
  /// @brief Create a command buffer for the given device.
  /// @param device The device whose backend determines the recording strategy.
  CommandBuffer(const Device& device);

  /// @brief Construct from a backend implementation.
  /// @param impl Shared pointer to the backend-specific implementation.
  CommandBuffer(std::shared_ptr<implementation::CommandBuffer> impl);

  /// @brief Get the backend implementation (const).
  std::shared_ptr<implementation::CommandBuffer> impl() const { return _impl; }

  /// @brief Get the backend implementation (mutable).
  std::shared_ptr<implementation::CommandBuffer>& impl() { return _impl; }

  /// @brief Record a kernel dispatch.
  /// @tparam ARGS Variadic argument types, each convertible to Attribute.
  /// @param function The kernel to dispatch.
  /// @param launchArgs Global and local work size configuration.
  /// @param args Kernel arguments (buffers, images, scalars, local memory).
  template <typename... ARGS>
  void dispatch(Function& function, const LaunchArgs& launchArgs,
                ARGS&&... args) {
    std::vector<Attribute> attrArgs;
    implementation::Function::addArgs(attrArgs, std::forward<ARGS>(args)...);
    dispatchImpl(function, launchArgs, attrArgs);
  }

  /// @brief Record an indirect kernel dispatch.
  ///
  /// The workgroup counts (X, Y, Z) are read from @p indirectBuffer at
  /// @p indirectOffset at execution time. The buffer must contain three
  /// consecutive @c uint32_t values. Local work size is determined by the
  /// function/pipeline as usual.
  ///
  /// On backends with native indirect dispatch (Metal), this encodes a
  /// single GPU-side indirect dispatch with no CPU round-trip. On other
  /// backends it falls back to sync + readback + regular dispatch.
  ///
  /// @tparam ARGS Variadic argument types, each convertible to Attribute.
  /// @param function The kernel to dispatch.
  /// @param indirectBuffer Buffer containing 3x uint32_t workgroup counts.
  /// @param indirectOffset Byte offset into indirectBuffer.
  /// @param args Kernel arguments (buffers, images, scalars, local memory).
  template <typename... ARGS>
  void dispatchIndirect(Function& function, const Buffer& indirectBuffer,
                        size_t indirectOffset, ARGS&&... args) {
    std::vector<Attribute> attrArgs;
    implementation::Function::addArgs(attrArgs, std::forward<ARGS>(args)...);
    dispatchIndirectImpl(function, indirectBuffer, indirectOffset, attrArgs);
  }

  /// @brief Record a device-to-device buffer copy.
  /// @param dst Destination buffer.
  /// @param src Source buffer.
  /// @param bytes Number of bytes to copy.
  void copyBuffer(Buffer& dst, const Buffer& src, size_t bytes);

  /// @brief Record a device-to-device buffer copy with offsets.
  void copyBuffer(Buffer& dst, size_t dstOff, const Buffer& src, size_t srcOff,
                  size_t bytes);

  /// @brief Record a buffer fill.
  /// @param buf Buffer to fill.
  /// @param offset Byte offset into the buffer.
  /// @param size Number of bytes to fill.
  /// @param value Byte value to fill with.
  void fillBuffer(Buffer& buf, size_t offset, size_t size, uint8_t value);

  /// @brief Record a barrier that ensures all previously recorded operations
  /// complete before subsequent ones begin.
  void barrier();

  /// @brief Record a wait for an event.
  /// @param e The event to wait for.
  void waitForEvent(const Event& e);

  /// @brief Record an event at the current point.
  /// @return An Event that will be signaled when preceding recorded operations
  /// complete.
  Event recordEvent();

  /// @brief Submit all recorded commands for execution on a stream.
  /// @param stream The stream to execute on.
  void submit(const Stream& stream);

  /// @brief Clear all recorded commands for reuse.
  void reset();

 private:
  void dispatchImpl(Function& function, const LaunchArgs& launchArgs,
                    const std::vector<Attribute>& args);

  void dispatchIndirectImpl(Function& function, const Buffer& indirectBuffer,
                            size_t indirectOffset,
                            const std::vector<Attribute>& args);

  std::shared_ptr<implementation::CommandBuffer> _impl;
};

}  // namespace ghost

#endif
