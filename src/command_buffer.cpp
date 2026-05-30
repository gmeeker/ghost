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

#include <ghost/command_buffer.h>
#include <ghost/implementation/recorded_command_buffer.h>

namespace ghost {
namespace implementation {

std::shared_ptr<implementation::Event> RecordedCommandBuffer::addRecordEvent(
    const ghost::Stream& stream) {
  auto event =
      static_cast<implementation::Stream*>(stream.impl().get())->record();
  commands.push_back(RecordEventCmd{event});
  return event;
}

void RecordedCommandBuffer::submit(const ghost::Stream& stream) {
  replayInto(stream, stream);
  // The replay enqueues onto the stream but doesn't wait. Since the fallback
  // has no native cb-completion hook, drain the stream and invoke handlers
  // inline. Backends with native cbs override submit() and route handlers
  // through their native completion callback instead.
  if (!pendingCompletionHandlers.empty()) {
    static_cast<implementation::Stream*>(stream.impl().get())->sync();
    auto handlers = std::move(pendingCompletionHandlers);
    pendingCompletionHandlers.clear();
    for (auto& h : handlers) h();
  }
}

void RecordedCommandBuffer::replayInto(const ghost::Encoder& enc,
                                       const ghost::Stream& stream) {
  // Reusable wrapper objects so we don't allocate per command.
  ghost::Buffer srcWrap(nullptr);
  ghost::Buffer dstWrap(nullptr);
  ghost::Image srcImgWrap(nullptr);
  ghost::Image dstImgWrap(nullptr);

  for (auto& command : commands) {
    std::visit(
        [&](auto& cmd) {
          using T = std::decay_t<decltype(cmd)>;
          if constexpr (std::is_same_v<T, DispatchCmd>) {
            cmd.function->execute(enc, cmd.launchArgs, cmd.args);
          } else if constexpr (std::is_same_v<T, DispatchIndirectCmd>) {
            cmd.function->executeIndirect(enc, cmd.indirectBuffer,
                                          cmd.indirectOffset, cmd.args);
          } else if constexpr (std::is_same_v<T, CopyBufferCmd>) {
            dstWrap.impl() = cmd.dst;
            srcWrap.impl() =
                std::const_pointer_cast<implementation::Buffer>(cmd.src);
            cmd.dst->copy(enc, srcWrap, cmd.srcOffset, cmd.dstOffset,
                          cmd.bytes);
          } else if constexpr (std::is_same_v<T, CopyBufferRawCmd>) {
            cmd.dst->copy(enc, cmd.src.data(), cmd.dstOffset, cmd.bytes);
          } else if constexpr (std::is_same_v<T, ReadBufferCmd>) {
            cmd.src->copyTo(enc, cmd.dst, cmd.srcOffset, cmd.bytes);
          } else if constexpr (std::is_same_v<T, FillBufferCmd>) {
            cmd.dst->fill(enc, cmd.offset, cmd.size, cmd.value);
          } else if constexpr (std::is_same_v<T, FillBufferPatternCmd>) {
            cmd.dst->fill(enc, cmd.offset, cmd.size, cmd.pattern.data(),
                          cmd.pattern.size());
          } else if constexpr (std::is_same_v<T, CopyImageCmd>) {
            dstImgWrap.impl() =
                std::const_pointer_cast<implementation::Image>(cmd.dst);
            srcImgWrap.impl() =
                std::const_pointer_cast<implementation::Image>(cmd.src);
            cmd.dst->copy(enc, srcImgWrap);
          } else if constexpr (std::is_same_v<T, CopyImageFromBufferCmd>) {
            srcWrap.impl() = cmd.src;
            cmd.dst->copy(enc, srcWrap, cmd.layout);
          } else if constexpr (std::is_same_v<T, CopyImageFromHostCmd>) {
            cmd.dst->copy(enc, cmd.src.data(), cmd.layout);
          } else if constexpr (std::is_same_v<T, CopyImageToBufferCmd>) {
            dstWrap.impl() = cmd.dst;
            cmd.src->copyTo(enc, dstWrap, cmd.layout);
          } else if constexpr (std::is_same_v<T, CopyImageToHostCmd>) {
            cmd.src->copyTo(enc, cmd.dst, cmd.layout);
          } else if constexpr (std::is_same_v<T, BarrierCmd>) {
            // Delegate to the stream's barrier(). Default drains the stream;
            // backends whose enqueue already serializes (OpenCL) override it
            // with a non-draining variant. Backends with a native cb override
            // submit() and emit a native pipeline barrier instead.
            static_cast<implementation::Stream*>(stream.impl().get())
                ->barrier();
          } else if constexpr (std::is_same_v<T, WaitEventCmd>) {
            static_cast<implementation::Stream*>(stream.impl().get())
                ->waitForEvent(cmd.event);
          } else if constexpr (std::is_same_v<T, RecordEventCmd>) {
            static_cast<implementation::Stream*>(stream.impl().get())->record();
          } else if constexpr (std::is_same_v<T, EncodeNativeCmd>) {
            replayEncodeNative(cmd, stream);
          }
        },
        command);
  }
}

void RecordedCommandBuffer::replayEncodeNative(const EncodeNativeCmd&,
                                               const ghost::Stream&) {
  // Fallback (CPU) has no native API to interop with; backends with
  // native interop override this hook.
  throw ghost::unsupported_error();
}

std::shared_ptr<CommandBuffer> CommandBuffer::createDefault() {
  return std::make_shared<RecordedCommandBuffer>();
}

}  // namespace implementation

// Public CommandBuffer bridge

CommandBuffer::CommandBuffer(const Device& device,
                             const CommandBufferOptions& options)
    : Encoder(device.impl()->createCommandBuffer(options)) {}

CommandBuffer::CommandBuffer(
    std::shared_ptr<implementation::CommandBuffer> impl)
    : Encoder(impl) {}

void CommandBuffer::barrier() {
  static_cast<implementation::CommandBuffer*>(_impl.get())->addBarrier();
}

void CommandBuffer::waitForEvent(const Event& e) {
  static_cast<implementation::CommandBuffer*>(_impl.get())
      ->addWaitForEvent(e.impl());
}

Event CommandBuffer::recordEvent() { throw unsupported_error(); }

void CommandBuffer::submit(const Stream& stream) {
  static_cast<implementation::CommandBuffer*>(_impl.get())->submit(stream);
}

void CommandBuffer::reset() {
  static_cast<implementation::CommandBuffer*>(_impl.get())->reset();
}

void CommandBuffer::onCompletion(std::function<void()> handler) {
  static_cast<implementation::CommandBuffer*>(_impl.get())
      ->onCompletion(std::move(handler));
}

}  // namespace ghost
