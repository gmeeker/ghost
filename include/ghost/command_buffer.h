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
class CommandBuffer : public Encoder {
 protected:
  CommandBuffer() {}

  CommandBuffer(const CommandBuffer&) = delete;

  virtual ~CommandBuffer() {}

  CommandBuffer& operator=(const CommandBuffer&) = delete;

 public:
  CommandBuffer* asCommandBuffer() override { return this; }

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

  virtual void copyBufferRaw(std::shared_ptr<Buffer> dst, const void* src,
                             size_t dstOffset, size_t bytes) = 0;

  virtual void readBuffer(std::shared_ptr<const Buffer> src, void* dst,
                          size_t srcOffset, size_t bytes) = 0;

  virtual void fillBuffer(std::shared_ptr<Buffer> dst, size_t offset,
                          size_t size, uint8_t value) = 0;

  virtual void fillBufferPattern(std::shared_ptr<Buffer> dst, size_t offset,
                                 size_t size, const void* pattern,
                                 size_t patternSize) = 0;

  virtual void copyImage(std::shared_ptr<Image> dst,
                         std::shared_ptr<const Image> src) = 0;

  virtual void copyImageFromBuffer(std::shared_ptr<Image> dst,
                                   std::shared_ptr<Buffer> src,
                                   const BufferLayout& layout) = 0;

  virtual void copyImageFromHost(std::shared_ptr<Image> dst, const void* src,
                                 const BufferLayout& layout) = 0;

  virtual void copyImageToBuffer(std::shared_ptr<const Image> src,
                                 std::shared_ptr<Buffer> dst,
                                 const BufferLayout& layout) = 0;

  virtual void copyImageToHost(std::shared_ptr<const Image> src, void* dst,
                               const BufferLayout& layout) = 0;

  virtual void submit(const ghost::Stream& stream) = 0;

  virtual void reset() = 0;

  virtual void addBarrier() = 0;

  virtual void addWaitForEvent(std::shared_ptr<Event> e) = 0;

  virtual std::shared_ptr<Event> addRecordEvent(
      const ghost::Stream& stream) = 0;

  static std::shared_ptr<CommandBuffer> createDefault();
};

}  // namespace implementation

/// @brief A deferred execution buffer that records GPU operations and submits
/// them as a batch.
///
/// CommandBuffer inherits from Encoder, so it can be passed anywhere an
/// Encoder is accepted (Buffer::copy, Function::operator(), etc.).
/// Operations are recorded when called through the Encoder interface and
/// executed when submit() is called.
///
/// Unlike Stream (which serializes all operations), CommandBuffer provides
/// no implicit synchronization between operations. Use barrier() for
/// explicit ordering within a batch.
///
/// @code
/// ghost::CommandBuffer cb(device);
/// buf.copy(cb, src, bytes);
/// fn(launch, cb)(buffer, 42.0f);
/// cb.barrier();
/// buf2.copy(cb, buf, bytes);
/// cb.submit(stream);
/// @endcode
class CommandBuffer : public Encoder {
 public:
  /// @brief Create a command buffer for the given device.
  /// @param device The device whose backend determines the recording strategy.
  CommandBuffer(const Device& device);

  /// @brief Construct from a backend implementation.
  /// @param impl Shared pointer to the backend-specific implementation.
  CommandBuffer(std::shared_ptr<implementation::CommandBuffer> impl);

  /// @brief Record an indirect kernel dispatch.
  ///
  /// Workgroup counts are read from @p indirectBuffer at dispatch time.
  template <typename... ARGS>
  void dispatchIndirect(Function& function, const Buffer& indirectBuffer,
                        size_t indirectOffset, ARGS&&... args) {
    std::vector<Attribute> attrArgs;
    implementation::Function::addArgs(attrArgs, std::forward<ARGS>(args)...);
    static_cast<implementation::CommandBuffer*>(_impl.get())
        ->dispatchIndirect(function.impl(), indirectBuffer.impl(),
                           indirectOffset, attrArgs);
  }

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
};

}  // namespace ghost

#endif
