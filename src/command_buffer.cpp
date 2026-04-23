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
};

struct DispatchIndirectCmd {
  std::shared_ptr<implementation::Function> function;
  std::shared_ptr<implementation::Buffer> indirectBuffer;
  size_t indirectOffset;
  std::vector<Attribute> args;
};

struct CopyBufferCmd {
  std::shared_ptr<implementation::Buffer> dst;
  std::shared_ptr<implementation::Buffer> src;
  size_t srcOffset;
  size_t dstOffset;
  size_t bytes;
};

struct CopyBufferRawCmd {
  std::shared_ptr<implementation::Buffer> dst;
  const void* src;
  size_t dstOffset;
  size_t bytes;
};

struct ReadBufferCmd {
  std::shared_ptr<const implementation::Buffer> src;
  void* dst;
  size_t srcOffset;
  size_t bytes;
};

struct FillBufferCmd {
  std::shared_ptr<implementation::Buffer> dst;
  size_t offset;
  size_t size;
  uint8_t value;
};

struct FillBufferPatternCmd {
  std::shared_ptr<implementation::Buffer> dst;
  size_t offset;
  size_t size;
  std::vector<uint8_t> pattern;
};

struct CopyImageCmd {
  std::shared_ptr<implementation::Image> dst;
  std::shared_ptr<const implementation::Image> src;
};

struct CopyImageFromBufferCmd {
  std::shared_ptr<implementation::Image> dst;
  std::shared_ptr<implementation::Buffer> src;
  BufferLayout layout;
};

struct CopyImageFromHostCmd {
  std::shared_ptr<implementation::Image> dst;
  const void* src;
  BufferLayout layout;
};

struct CopyImageToBufferCmd {
  std::shared_ptr<const implementation::Image> src;
  std::shared_ptr<implementation::Buffer> dst;
  BufferLayout layout;
};

struct CopyImageToHostCmd {
  std::shared_ptr<const implementation::Image> src;
  void* dst;
  BufferLayout layout;
};

struct BarrierCmd {};

struct WaitEventCmd {
  std::shared_ptr<implementation::Event> event;
};

struct RecordEventCmd {
  std::shared_ptr<implementation::Event> event;
};

using Command =
    std::variant<DispatchCmd, DispatchIndirectCmd, CopyBufferCmd,
                 CopyBufferRawCmd, ReadBufferCmd, FillBufferCmd,
                 FillBufferPatternCmd, CopyImageCmd, CopyImageFromBufferCmd,
                 CopyImageFromHostCmd, CopyImageToBufferCmd, CopyImageToHostCmd,
                 BarrierCmd, WaitEventCmd, RecordEventCmd>;

class DefaultCommandBuffer : public CommandBuffer {
 public:
  std::vector<Command> commands;

  void dispatch(std::shared_ptr<implementation::Function> function,
                const LaunchArgs& launchArgs,
                const std::vector<Attribute>& args) override {
    commands.push_back(DispatchCmd{function, launchArgs, args});
  }

  void dispatchIndirect(std::shared_ptr<implementation::Function> function,
                        std::shared_ptr<implementation::Buffer> indirectBuffer,
                        size_t indirectOffset,
                        const std::vector<Attribute>& args) override {
    commands.push_back(
        DispatchIndirectCmd{function, indirectBuffer, indirectOffset, args});
  }

  void copyBuffer(std::shared_ptr<implementation::Buffer> dst,
                  std::shared_ptr<implementation::Buffer> src, size_t srcOffset,
                  size_t dstOffset, size_t bytes) override {
    commands.push_back(CopyBufferCmd{dst, src, srcOffset, dstOffset, bytes});
  }

  void copyBufferRaw(std::shared_ptr<implementation::Buffer> dst,
                     const void* src, size_t dstOffset, size_t bytes) override {
    commands.push_back(CopyBufferRawCmd{dst, src, dstOffset, bytes});
  }

  void readBuffer(std::shared_ptr<const implementation::Buffer> src, void* dst,
                  size_t srcOffset, size_t bytes) override {
    commands.push_back(ReadBufferCmd{src, dst, srcOffset, bytes});
  }

  void fillBuffer(std::shared_ptr<implementation::Buffer> dst, size_t offset,
                  size_t size, uint8_t value) override {
    commands.push_back(FillBufferCmd{dst, offset, size, value});
  }

  void fillBufferPattern(std::shared_ptr<implementation::Buffer> dst,
                         size_t offset, size_t size, const void* pattern,
                         size_t patternSize) override {
    std::vector<uint8_t> p(static_cast<const uint8_t*>(pattern),
                           static_cast<const uint8_t*>(pattern) + patternSize);
    commands.push_back(FillBufferPatternCmd{dst, offset, size, std::move(p)});
  }

  void copyImage(std::shared_ptr<implementation::Image> dst,
                 std::shared_ptr<const implementation::Image> src) override {
    commands.push_back(CopyImageCmd{dst, src});
  }

  void copyImageFromBuffer(std::shared_ptr<implementation::Image> dst,
                           std::shared_ptr<implementation::Buffer> src,
                           const BufferLayout& layout) override {
    commands.push_back(CopyImageFromBufferCmd{dst, src, layout});
  }

  void copyImageFromHost(std::shared_ptr<implementation::Image> dst,
                         const void* src, const BufferLayout& layout) override {
    commands.push_back(CopyImageFromHostCmd{dst, src, layout});
  }

  void copyImageToBuffer(std::shared_ptr<const implementation::Image> src,
                         std::shared_ptr<implementation::Buffer> dst,
                         const BufferLayout& layout) override {
    commands.push_back(CopyImageToBufferCmd{src, dst, layout});
  }

  void copyImageToHost(std::shared_ptr<const implementation::Image> src,
                       void* dst, const BufferLayout& layout) override {
    commands.push_back(CopyImageToHostCmd{src, dst, layout});
  }

  void addBarrier() override { commands.push_back(BarrierCmd{}); }

  void addWaitForEvent(std::shared_ptr<implementation::Event> e) override {
    commands.push_back(WaitEventCmd{e});
  }

  std::shared_ptr<implementation::Event> addRecordEvent(
      const ghost::Stream& stream) override {
    auto event =
        static_cast<implementation::Stream*>(stream.impl().get())->record();
    commands.push_back(RecordEventCmd{event});
    return event;
  }

  void submit(const ghost::Stream& stream) override {
    // Use the stream directly as the encoder — backends downcast
    // the encoder's impl to their stream type during replay.
    ghost::Buffer srcWrap(nullptr);
    const ghost::Encoder& enc = stream;
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
              cmd.dst->copy(enc, cmd.src, cmd.dstOffset, cmd.bytes);
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
              cmd.dst->copy(enc, cmd.src, cmd.layout);
            } else if constexpr (std::is_same_v<T, CopyImageToBufferCmd>) {
              dstWrap.impl() = cmd.dst;
              cmd.src->copyTo(enc, dstWrap, cmd.layout);
            } else if constexpr (std::is_same_v<T, CopyImageToHostCmd>) {
              cmd.src->copyTo(enc, cmd.dst, cmd.layout);
            } else if constexpr (std::is_same_v<T, BarrierCmd>) {
              static_cast<implementation::Stream*>(stream.impl().get())->sync();
            } else if constexpr (std::is_same_v<T, WaitEventCmd>) {
              static_cast<implementation::Stream*>(stream.impl().get())
                  ->waitForEvent(cmd.event);
            } else if constexpr (std::is_same_v<T, RecordEventCmd>) {
              static_cast<implementation::Stream*>(stream.impl().get())
                  ->record();
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
    : Encoder(implementation::CommandBuffer::createDefault()) {}

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

}  // namespace ghost
