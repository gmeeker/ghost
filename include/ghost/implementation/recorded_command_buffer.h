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

#ifndef GHOST_IMPLEMENTATION_RECORDED_COMMAND_BUFFER_H
#define GHOST_IMPLEMENTATION_RECORDED_COMMAND_BUFFER_H

#include <ghost/command_buffer.h>

#include <cstring>
#include <functional>
#include <variant>
#include <vector>

namespace ghost {
namespace implementation {

/// @brief Tagged command records used by @ref RecordedCommandBuffer.
///
/// One struct per public CommandBuffer entry point. Each captures
/// @c shared_ptr to the involved impl resources, which keeps those resources
/// alive for the lifetime of this record (until the @c commands vector is
/// cleared by @c reset() or the @c RecordedCommandBuffer is destroyed).
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
  // src.ownsBytes() distinguishes:
  // - Recorded from copyBufferRaw(void*, ...): src is HostBytes::adopt of a
  //   Ghost-allocated snapshot (caller's pointer may have been a stack local).
  // - Recorded from copyBufferRaw(HostBytes, ...): src is the caller's
  //   HostBytes verbatim; backends may lifetime-extend src.owner() to keep
  //   the upload async.
  HostBytes src;
  size_t dstOffset;
  size_t bytes;
};

struct ReadBufferCmd {
  std::shared_ptr<const implementation::Buffer> src;
  // dst.ownsBytes() distinguishes:
  // - Recorded from readBuffer(void*, ...): dst is HostBytes::borrow of the
  //   caller's pointer; caller keeps it alive until submit() returns (cb
  //   replay blocks per op for the void* readback path).
  // - Recorded from readBuffer(HostBytes, ...): dst is the caller's HostBytes
  //   verbatim; backends may lifetime-extend dst.owner() to keep the readback
  //   async without burdening the caller with lifetime tracking.
  HostBytes dst;
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
  // See @c CopyBufferRawCmd::src for the ownsBytes() distinction.
  HostBytes src;
  BufferLayout layout;
};

struct CopyImageToBufferCmd {
  std::shared_ptr<const implementation::Image> src;
  std::shared_ptr<implementation::Buffer> dst;
  BufferLayout layout;
};

struct CopyImageToHostCmd {
  std::shared_ptr<const implementation::Image> src;
  // See @c ReadBufferCmd::dst for the ownsBytes() distinction.
  HostBytes dst;
  BufferLayout layout;
};

struct BarrierCmd {};

struct WaitEventCmd {
  std::shared_ptr<implementation::Event> event;
};

struct RecordEventCmd {
  std::shared_ptr<implementation::Event> event;
};

/// @brief Native-API encoding step deferred to submit-time replay.
///
/// Used by per-backend @c encodeNative entry points (see e.g.
/// @c CommandBufferMetal::encodeNative). The body is type-erased here so
/// the shared variant stays closed; each backend's public entry point
/// wraps a typed callback into this form, and the backend's replay site
/// passes the matching native context pointer when invoking @c body.
struct EncodeNativeCmd {
  std::function<void(void* nativeContext)> body;
};

using Command =
    std::variant<DispatchCmd, DispatchIndirectCmd, CopyBufferCmd,
                 CopyBufferRawCmd, ReadBufferCmd, FillBufferCmd,
                 FillBufferPatternCmd, CopyImageCmd, CopyImageFromBufferCmd,
                 CopyImageFromHostCmd, CopyImageToBufferCmd, CopyImageToHostCmd,
                 BarrierCmd, WaitEventCmd, RecordEventCmd, EncodeNativeCmd>;

/// @brief Record-and-replay @ref CommandBuffer used as the fallback for
/// backends without a native command-buffer concept (CUDA, OpenCL, CPU).
///
/// Each public dispatch/copy/fill call appends a tagged @ref Command record
/// to @c commands. On @c submit() the records are replayed in order by
/// invoking the same operation against the target encoder.
///
/// Backends with a native command-buffer wrap this class to reuse the
/// variant-recording machinery and override @ref submit (and any other
/// methods that need backend-native handling) to record the replayed
/// operations into their native object instead of replaying onto the
/// stream's queue directly.
class RecordedCommandBuffer : public CommandBuffer {
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
    // Capture by value: the user's source pointer may be a stack local that
    // dies before submit(). Wrap the snapshot in a HostBytes so the backend's
    // owned-handle replay path can lifetime-extend it past reset().
    auto* buf = new uint8_t[bytes];
    std::memcpy(buf, src, bytes);
    HostBytes owned = HostBytes::adopt(
        buf, [](void* p) { delete[] static_cast<uint8_t*>(p); });
    commands.push_back(
        CopyBufferRawCmd{dst, std::move(owned), dstOffset, bytes});
  }

  void copyBufferRaw(std::shared_ptr<implementation::Buffer> dst, HostBytes src,
                     size_t dstOffset, size_t bytes) override {
    commands.push_back(CopyBufferRawCmd{dst, std::move(src), dstOffset, bytes});
  }

  void readBuffer(std::shared_ptr<const implementation::Buffer> src, void* dst,
                  size_t srcOffset, size_t bytes) override {
    // Borrow: caller is documented to keep @c dst valid until @c stream.sync().
    commands.push_back(
        ReadBufferCmd{src, HostBytes::borrow(dst), srcOffset, bytes});
  }

  void readBuffer(std::shared_ptr<const implementation::Buffer> src,
                  HostBytes dst, size_t srcOffset, size_t bytes) override {
    commands.push_back(ReadBufferCmd{src, std::move(dst), srcOffset, bytes});
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
    // Capture by value (same reason as copyBufferRaw). Size is derived from
    // the destination's pixel layout, mirroring what the immediate path reads.
    size_t pixelSize = dst->description().pixelSize();
    size_t rowStride = layout.rowBytes(pixelSize);
    size_t sliceStride = layout.sliceBytes(rowStride);
    size_t bytes = sliceStride * layout.size.z;
    auto* buf = new uint8_t[bytes];
    std::memcpy(buf, src, bytes);
    HostBytes owned = HostBytes::adopt(
        buf, [](void* p) { delete[] static_cast<uint8_t*>(p); });
    commands.push_back(CopyImageFromHostCmd{dst, std::move(owned), layout});
  }

  void copyImageFromHost(std::shared_ptr<implementation::Image> dst,
                         HostBytes src, const BufferLayout& layout) override {
    commands.push_back(CopyImageFromHostCmd{dst, std::move(src), layout});
  }

  void copyImageToBuffer(std::shared_ptr<const implementation::Image> src,
                         std::shared_ptr<implementation::Buffer> dst,
                         const BufferLayout& layout) override {
    commands.push_back(CopyImageToBufferCmd{src, dst, layout});
  }

  void copyImageToHost(std::shared_ptr<const implementation::Image> src,
                       void* dst, const BufferLayout& layout) override {
    commands.push_back(CopyImageToHostCmd{src, HostBytes::borrow(dst), layout});
  }

  void copyImageToHost(std::shared_ptr<const implementation::Image> src,
                       HostBytes dst, const BufferLayout& layout) override {
    commands.push_back(CopyImageToHostCmd{src, std::move(dst), layout});
  }

  void addBarrier() override { commands.push_back(BarrierCmd{}); }

  void addWaitForEvent(std::shared_ptr<implementation::Event> e) override {
    commands.push_back(WaitEventCmd{e});
  }

  std::shared_ptr<implementation::Event> addRecordEvent(
      const ghost::Stream& stream) override;

  /// @brief Record a native-API encoding step. Backend-specific public
  /// @c encodeNative entry points wrap their typed callback into the
  /// @c std::function<void(void*)> form expected here.
  void addEncodeNative(std::function<void(void*)> body) {
    commands.push_back(EncodeNativeCmd{std::move(body)});
  }

  void submit(const ghost::Stream& stream) override;

  std::shared_ptr<Executable> compile(const CompileOptions& options) override;

  std::shared_ptr<Executable> compileRegioned(
      const CompileOptions& options) override;

  void beginCompiledRegion() override { _regionBegin = commands.size(); }

  void endCompiledRegion() override {
    _regionEnd = commands.size();
    _hasRegion = true;
  }

  /// @brief Create an empty command buffer of this backend's concrete type.
  ///
  /// Used by @ref compile to snapshot the recorded commands into an
  /// independent buffer that still routes backend-specific replay (e.g.
  /// CUDA's @c encodeNative) correctly. Backends that add such hooks override
  /// this to return their own subclass.
  virtual std::shared_ptr<RecordedCommandBuffer> cloneEmpty() const;

  void reset() override {
    commands.clear();
    pendingCompletionHandlers.clear();
    _regionBegin = 0;
    _regionEnd = 0;
    _hasRegion = false;
  }

  void onCompletion(std::function<void()> handler) override {
    pendingCompletionHandlers.push_back(std::move(handler));
  }

 protected:
  /// @brief Marked compiled region [_regionBegin, _regionEnd) within @c
  /// commands (see @c CommandBuffer::beginCompiledRegion). When @c _hasRegion
  /// is set and the region is a proper sub-range, @ref compileRegioned splits
  /// into a @c SegmentedExecutable. 0/0 / false means "whole sequence".
  size_t _regionBegin = 0;
  size_t _regionEnd = 0;
  bool _hasRegion = false;

  /// @brief Handlers registered via onCompletion() that have not yet been
  /// associated with a submission. Moved into @c inFlightCompletionHandlers
  /// at submit() time so subsequent onCompletion() calls accumulate into a
  /// fresh list for the next submit (matches the documented semantics:
  /// handlers fire for the submit they were registered against).
  std::vector<std::function<void()>> pendingCompletionHandlers;

  /// @brief Handlers attached to the most recent submission, fired by the
  /// backend's completion observation point (Metal: native completion
  /// block; Vulkan/DirectX: @c waitForCompletion; fallback: end of submit
  /// after a sync). Backends that have a native async path consume this
  /// list inside their submit() override and clear it; this base storage is
  /// here for backends that drain on a host-side wait.
  std::vector<std::function<void()>> inFlightCompletionHandlers;

  /// @brief Replay the recorded @c commands sequence onto @p enc.
  ///
  /// Default behavior: each command invokes the corresponding native
  /// operation on the encoder. Used by the fallback @c submit and by
  /// backend subclasses that want to replay onto their own native cb.
  void replayInto(const ghost::Encoder& enc, const ghost::Stream& stream);

  /// @brief Invoked from @c replayInto when an @c EncodeNativeCmd is hit.
  ///
  /// Default throws @c ghost::unsupported_error — the fallback (CPU) path
  /// has no native API to interop with. Backends whose @c CommandBuffer
  /// flows through @c RecordedCommandBuffer::submit (i.e. CUDA, OpenCL)
  /// override this to invoke @c cmd.body with the appropriate native
  /// context. Backends with their own @c submit replay loop (Metal,
  /// Vulkan, DirectX) handle @c EncodeNativeCmd inline and never reach
  /// this hook.
  virtual void replayEncodeNative(const EncodeNativeCmd& cmd,
                                  const ghost::Stream& stream);
};

}  // namespace implementation
}  // namespace ghost

#endif
