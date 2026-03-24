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

#include <variant>
#include <vector>

namespace ghost {
namespace implementation {

struct DispatchCmd {
  std::shared_ptr<implementation::Function> function;
  LaunchArgs launchArgs;
  std::vector<Attribute> args;
  // Keep referenced objects alive until submit
  std::vector<std::shared_ptr<implementation::Buffer>> bufferRefs;
  std::vector<std::shared_ptr<implementation::Image>> imageRefs;
};

struct CopyBufferCmd {
  std::shared_ptr<implementation::Buffer> dst;
  std::shared_ptr<implementation::Buffer> src;
  size_t srcOffset;
  size_t dstOffset;
  size_t bytes;
};

struct FillBufferCmd {
  std::shared_ptr<implementation::Buffer> dst;
  size_t offset;
  size_t size;
  uint8_t value;
};

struct BarrierCmd {};

struct WaitEventCmd {
  std::shared_ptr<implementation::Event> event;
};

struct RecordEventCmd {
  std::shared_ptr<implementation::Event> event;
};

using Command = std::variant<DispatchCmd, CopyBufferCmd, FillBufferCmd,
                             BarrierCmd, WaitEventCmd, RecordEventCmd>;

class DefaultCommandBuffer : public CommandBuffer {
 public:
  std::vector<Command> commands;

  void dispatch(std::shared_ptr<implementation::Function> function,
                const LaunchArgs& launchArgs,
                const std::vector<Attribute>& args) override {
    DispatchCmd cmd;
    cmd.function = function;
    cmd.launchArgs = launchArgs;
    cmd.args = args;
    // Capture shared_ptrs to keep Buffer/Image objects alive
    for (auto& a : cmd.args) {
      if (a.type() == Attribute::Type_Buffer && a.asBuffer()) {
        cmd.bufferRefs.push_back(a.asBuffer()->impl());
      } else if (a.type() == Attribute::Type_Image && a.asImage()) {
        cmd.imageRefs.push_back(a.asImage()->impl());
      }
    }
    commands.push_back(std::move(cmd));
  }

  void copyBuffer(std::shared_ptr<implementation::Buffer> dst,
                  std::shared_ptr<implementation::Buffer> src, size_t srcOffset,
                  size_t dstOffset, size_t bytes) override {
    CopyBufferCmd cmd;
    cmd.dst = dst;
    cmd.src = src;
    cmd.srcOffset = srcOffset;
    cmd.dstOffset = dstOffset;
    cmd.bytes = bytes;
    commands.push_back(std::move(cmd));
  }

  void fillBuffer(std::shared_ptr<implementation::Buffer> dst, size_t offset,
                  size_t size, uint8_t value) override {
    FillBufferCmd cmd;
    cmd.dst = dst;
    cmd.offset = offset;
    cmd.size = size;
    cmd.value = value;
    commands.push_back(std::move(cmd));
  }

  void addBarrier() override { commands.push_back(BarrierCmd{}); }

  void addWaitForEvent(std::shared_ptr<implementation::Event> e) override {
    WaitEventCmd cmd;
    cmd.event = e;
    commands.push_back(std::move(cmd));
  }

  std::shared_ptr<implementation::Event> addRecordEvent(
      const ghost::Stream& stream) override {
    auto event = stream.impl()->record();
    RecordEventCmd cmd;
    cmd.event = event;
    commands.push_back(std::move(cmd));
    return event;
  }

  void submit(const ghost::Stream& stream) override {
    ghost::Buffer srcWrap(nullptr);
    ghost::Buffer dstWrap(nullptr);

    for (auto& command : commands) {
      std::visit(
          [&](auto& cmd) {
            using T = std::decay_t<decltype(cmd)>;
            if constexpr (std::is_same_v<T, DispatchCmd>) {
              cmd.function->execute(stream, cmd.launchArgs, cmd.args);
            } else if constexpr (std::is_same_v<T, CopyBufferCmd>) {
              // Wrap impl ptrs in public Buffer for the copy API
              dstWrap.impl() = cmd.dst;
              srcWrap.impl() = cmd.src;
              cmd.dst->copy(stream, srcWrap, cmd.srcOffset, cmd.dstOffset,
                            cmd.bytes);
            } else if constexpr (std::is_same_v<T, FillBufferCmd>) {
              cmd.dst->fill(stream, cmd.offset, cmd.size, cmd.value);
            } else if constexpr (std::is_same_v<T, BarrierCmd>) {
              stream.impl()->sync();
            } else if constexpr (std::is_same_v<T, WaitEventCmd>) {
              stream.impl()->waitForEvent(cmd.event);
            } else if constexpr (std::is_same_v<T, RecordEventCmd>) {
              // Event was already created during recording; re-record now
              auto ev = stream.impl()->record();
              // Note: the originally returned event won't track this
              // submission. For correct semantics, users should call
              // recordEvent() and submit() together.
            }
          },
          command);
    }
  }

  void reset() override { commands.clear(); }
};

std::shared_ptr<CommandBuffer> CommandBuffer::createDefault() {
  return std::make_shared<DefaultCommandBuffer>();
}

}  // namespace implementation

// Public CommandBuffer bridge

CommandBuffer::CommandBuffer(const Device& device)
    : _impl(implementation::CommandBuffer::createDefault()) {}

CommandBuffer::CommandBuffer(
    std::shared_ptr<implementation::CommandBuffer> impl)
    : _impl(impl) {}

void CommandBuffer::dispatchImpl(Function& function,
                                 const LaunchArgs& launchArgs,
                                 const std::vector<Attribute>& args) {
  _impl->dispatch(function.impl(), launchArgs, args);
}

void CommandBuffer::copyBuffer(Buffer& dst, const Buffer& src, size_t bytes) {
  _impl->copyBuffer(dst.impl(), src.impl(), 0, 0, bytes);
}

void CommandBuffer::copyBuffer(Buffer& dst, size_t dstOff, const Buffer& src,
                               size_t srcOff, size_t bytes) {
  _impl->copyBuffer(dst.impl(), src.impl(), srcOff, dstOff, bytes);
}

void CommandBuffer::fillBuffer(Buffer& buf, size_t offset, size_t size,
                               uint8_t value) {
  _impl->fillBuffer(buf.impl(), offset, size, value);
}

void CommandBuffer::barrier() { _impl->addBarrier(); }

void CommandBuffer::waitForEvent(const Event& e) {
  _impl->addWaitForEvent(e.impl());
}

Event CommandBuffer::recordEvent() {
  // We need a stream for the event, but we don't have one yet during
  // recording. Create a placeholder event that will be signaled on submit.
  // For now, record a placeholder and return a deferred event.
  // The simplest correct approach: defer to submit time via the command list.
  // We return a "pending" event that gets resolved on submit.
  // Implementation: use a special command that records the event during submit.
  // However, since we need to return an Event now, we store a shared_ptr
  // that will be populated later.

  // For the default implementation, we can't create a real event without a
  // stream. Return a placeholder that the submit will attempt to populate.
  // A simpler approach: don't support recordEvent during recording.
  // Instead, users should use stream.record() after submit().
  throw unsupported_error();
}

void CommandBuffer::submit(const Stream& stream) { _impl->submit(stream); }

void CommandBuffer::reset() { _impl->reset(); }

}  // namespace ghost
