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

#include <functional>
#include <memory>
#include <vector>

namespace ghost {

class Executable;

namespace implementation {

class Executable;

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

  /// @brief Record a host-to-device upload with caller-supplied ownership of
  /// the source bytes. See @c ghost::Buffer::copy(Encoder, HostBytes, ...).
  virtual void copyBufferRaw(std::shared_ptr<Buffer> dst, HostBytes src,
                             size_t dstOffset, size_t bytes) = 0;

  virtual void readBuffer(std::shared_ptr<const Buffer> src, void* dst,
                          size_t srcOffset, size_t bytes) = 0;

  /// @brief Record a device-to-host readback with caller-supplied ownership
  /// of the destination bytes. See @c ghost::Buffer::copyTo(Encoder, HostBytes,
  /// ...).
  virtual void readBuffer(std::shared_ptr<const Buffer> src, HostBytes dst,
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

  /// @brief Owned-handle variant of @c copyImageFromHost.
  virtual void copyImageFromHost(std::shared_ptr<Image> dst, HostBytes src,
                                 const BufferLayout& layout) = 0;

  virtual void copyImageToBuffer(std::shared_ptr<const Image> src,
                                 std::shared_ptr<Buffer> dst,
                                 const BufferLayout& layout) = 0;

  virtual void copyImageToHost(std::shared_ptr<const Image> src, void* dst,
                               const BufferLayout& layout) = 0;

  /// @brief Owned-handle variant of @c copyImageToHost.
  virtual void copyImageToHost(std::shared_ptr<const Image> src, HostBytes dst,
                               const BufferLayout& layout) = 0;

  virtual void submit(const ghost::Stream& stream) = 0;

  /// @brief Compile the recorded sequence into a reusable @ref Executable.
  /// The default (replay) implementation lives on @ref RecordedCommandBuffer;
  /// backends with native graph / command-buffer support override it.
  virtual std::shared_ptr<Executable> compile(
      const CompileOptions& options) = 0;

  /// @brief Compile honoring a marked compiled region (see
  /// @c CommandBuffer::beginCompiledRegion). Default ignores regions and
  /// compiles the whole sequence; @ref RecordedCommandBuffer overrides it to
  /// split into a @ref SegmentedExecutable. This is the entry point the public
  /// @c CommandBuffer::compile calls.
  virtual std::shared_ptr<Executable> compileRegioned(
      const CompileOptions& options) {
    return compile(options);
  }

  /// @brief Mark the start of a compiled region: ops recorded until
  /// @ref endCompiledRegion form the natively-compiled span; ops outside it
  /// replay on submit. Default no-op.
  virtual void beginCompiledRegion() {}

  /// @brief Mark the end of a compiled region. Default no-op.
  virtual void endCompiledRegion() {}

  virtual void reset() = 0;

  virtual void addBarrier() = 0;

  virtual void addWaitForEvent(std::shared_ptr<Event> e) = 0;

  virtual std::shared_ptr<Event> addRecordEvent(
      const ghost::Stream& stream) = 0;

  /// @brief Register a host-side callback to run after this cb's recorded
  /// work has completed on the GPU. See @ref ghost::CommandBuffer::onCompletion
  /// for full semantics.
  virtual void onCompletion(std::function<void()> handler) = 0;

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
  /// @param options Backend-encoder configuration (see CommandBufferOptions).
  CommandBuffer(const Device& device, const CommandBufferOptions& options = {});

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

  /// @brief Compile the recorded sequence into a reusable @ref Executable.
  ///
  /// Use this when the same sequence runs many times (e.g. once per frame):
  /// the returned Executable is retained by the caller and submitted
  /// repeatedly, amortizing instantiation. On backends with native support
  /// (CUDA graphs) submitting it is far cheaper than re-recording; elsewhere
  /// it transparently falls back to command replay (see
  /// @ref Executable::accelerated). The recorded commands are snapshotted, so
  /// this CommandBuffer may be reset and reused afterward.
  /// @param options Compilation options (see @ref CompileOptions).
  /// @return A reusable Executable.
  Executable compile(const CompileOptions& options = {});

  /// @brief Mark the start of a compiled region.
  ///
  /// Operations recorded between @c beginCompiledRegion and
  /// @ref endCompiledRegion form the span that @ref compile turns into a native
  /// graph / command buffer; operations recorded outside the region replay on
  /// the stream at submit time, in order. Use this to accelerate a contiguous
  /// compute span that is bracketed by non-capturable ops (host uploads /
  /// downloads, events) without dropping the whole sequence to replay:
  ///
  /// @code
  /// buf.copy(cb, hostInput, bytes);   // upload — replayed
  /// cb.beginCompiledRegion();
  /// fn(launch, cb)(out, buf);         // compute — compiled to a graph
  /// cb.endCompiledRegion();
  /// out.copyTo(cb, hostOutput, bytes);// download — replayed
  /// ghost::Executable exec = cb.compile();
  /// @endcode
  ///
  /// At most one region is honored (the last begin/end pair). Without a region,
  /// @ref compile compiles the whole sequence as before.
  void beginCompiledRegion();

  /// @brief Mark the end of a compiled region. See @ref beginCompiledRegion.
  void endCompiledRegion();

  /// @brief Clear all recorded commands for reuse.
  void reset();

  /// @brief Register a callback to run after this cb's work has completed.
  ///
  /// The handler is invoked once, after the GPU has finished executing the
  /// recorded commands from the most recent (or next) @ref submit. Handlers
  /// registered before @c submit() fire for that submit's completion;
  /// handlers registered after @c submit() but before @c reset() / next
  /// @c submit() fire for that same in-flight submit. Multiple handlers run
  /// in registration order.
  ///
  /// The handler may run on a backend-internal thread (e.g. Metal's
  /// completion queue, CUDA's host-fn worker). Do not call back into Ghost
  /// from inside the handler other than thread-safe operations on unrelated
  /// objects; in particular, do not call @c reset() / @c submit() on this
  /// cb from the handler.
  ///
  /// Typical use: returning host buffers to a pool, signalling a future,
  /// kicking the next CPU stage in a pipeline without blocking on
  /// @c stream.sync().
  void onCompletion(std::function<void()> handler);
};

}  // namespace ghost

#endif
