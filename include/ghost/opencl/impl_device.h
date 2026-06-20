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

#ifndef GHOST_OPENCL_IMPL_DEVICE_H
#define GHOST_OPENCL_IMPL_DEVICE_H

#include <CL/cl_ext.h>
#include <ghost/device.h>
#include <ghost/implementation/executable.h>
#include <ghost/implementation/recorded_command_buffer.h>
#include <ghost/opencl/ptr.h>

#include <functional>
#include <list>
#include <set>
#include <vector>

namespace ghost {
namespace implementation {
class DeviceOpenCL;
class FunctionOpenCL;

// We define the per-command clCommand*KHR function-pointer signatures here
// (using cl_properties, which cl_command_properties_khr aliases) rather than
// rely on the header typedefs, so the recording calls are correct regardless
// of which OpenCL-Headers revision Ghost is built against. The finalized
// cl_khr_command_buffer added a `const cl_command_properties_khr* properties`
// parameter to every clCommand*KHR entry point that the older provisional
// headers lacked; these signatures match the finalized form that shipping
// runtimes (pocl 7.x and any current driver) implement. create / finalize /
// release / enqueue / updateMutable are unchanged across revisions, so those
// keep the header typedefs (the headers gate them behind
// CL_ENABLE_BETA_EXTENSIONS, which the build defines for the OpenCL sources).
typedef cl_command_buffer_khr(CL_API_CALL* ghost_clCreateCommandBufferKHR_fn)(
    cl_uint, const cl_command_queue*, const cl_command_buffer_properties_khr*,
    cl_int*);
typedef cl_int(CL_API_CALL* ghost_clCommandNDRangeKernelKHR_fn)(
    cl_command_buffer_khr, cl_command_queue, const cl_properties*, cl_kernel,
    cl_uint, const size_t*, const size_t*, const size_t*, cl_uint,
    const cl_sync_point_khr*, cl_sync_point_khr*, cl_mutable_command_khr*);
typedef cl_int(CL_API_CALL* ghost_clCommandCopyBufferKHR_fn)(
    cl_command_buffer_khr, cl_command_queue, const cl_properties*, cl_mem,
    cl_mem, size_t, size_t, size_t, cl_uint, const cl_sync_point_khr*,
    cl_sync_point_khr*, cl_mutable_command_khr*);
typedef cl_int(CL_API_CALL* ghost_clCommandFillBufferKHR_fn)(
    cl_command_buffer_khr, cl_command_queue, const cl_properties*, cl_mem,
    const void*, size_t, size_t, size_t, cl_uint, const cl_sync_point_khr*,
    cl_sync_point_khr*, cl_mutable_command_khr*);
typedef cl_int(CL_API_CALL* ghost_clCommandBarrierWithWaitListKHR_fn)(
    cl_command_buffer_khr, cl_command_queue, const cl_properties*, cl_uint,
    const cl_sync_point_khr*, cl_sync_point_khr*, cl_mutable_command_khr*);
typedef cl_int(CL_API_CALL* ghost_clCommandCopyImageKHR_fn)(
    cl_command_buffer_khr, cl_command_queue, const cl_properties*, cl_mem,
    cl_mem, const size_t*, const size_t*, const size_t*, cl_uint,
    const cl_sync_point_khr*, cl_sync_point_khr*, cl_mutable_command_khr*);
typedef cl_int(CL_API_CALL* ghost_clCommandCopyBufferToImageKHR_fn)(
    cl_command_buffer_khr, cl_command_queue, const cl_properties*, cl_mem,
    cl_mem, size_t, const size_t*, const size_t*, cl_uint,
    const cl_sync_point_khr*, cl_sync_point_khr*, cl_mutable_command_khr*);
typedef cl_int(CL_API_CALL* ghost_clCommandCopyImageToBufferKHR_fn)(
    cl_command_buffer_khr, cl_command_queue, const cl_properties*, cl_mem,
    cl_mem, const size_t*, const size_t*, size_t, cl_uint,
    const cl_sync_point_khr*, cl_sync_point_khr*, cl_mutable_command_khr*);

/// @brief Runtime-loaded @c cl_khr_command_buffer entry points.
///
/// The extension's functions are not exported by the ICD loader as link-time
/// symbols, so they're resolved via
/// @c clGetExtensionFunctionAddressForPlatform once per device. @ref loaded
/// is true only when every entry point used by @ref ExecutableOpenCL resolved.
struct CommandBufferExtCL {
  clCreateCommandBufferKHR_fn create = nullptr;
  clFinalizeCommandBufferKHR_fn finalize = nullptr;
  clReleaseCommandBufferKHR_fn release = nullptr;
  clEnqueueCommandBufferKHR_fn enqueue = nullptr;
  ghost_clCommandNDRangeKernelKHR_fn ndrange = nullptr;
  ghost_clCommandCopyBufferKHR_fn copyBuffer = nullptr;
  ghost_clCommandFillBufferKHR_fn fillBuffer = nullptr;
  ghost_clCommandBarrierWithWaitListKHR_fn barrier = nullptr;
  // Optional: image-copy recording (cl_khr_command_buffer too, but gated
  // separately since a driver could omit the image commands).
  ghost_clCommandCopyImageKHR_fn copyImage = nullptr;
  ghost_clCommandCopyBufferToImageKHR_fn copyBufferToImage = nullptr;
  ghost_clCommandCopyImageToBufferKHR_fn copyImageToBuffer = nullptr;
  // Optional: cl_khr_command_buffer_mutable_dispatch — patch recorded kernel
  // args in place on update() instead of rebuilding the command buffer.
  clUpdateMutableCommandsKHR_fn updateMutable = nullptr;
  // Whether the device's CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR advertises
  // CL_MUTABLE_DISPATCH_ARGUMENTS_KHR (some drivers support mutating work sizes
  // but not kernel arguments — e.g. pocl's CPU device).
  bool mutableArgs = false;

  bool loaded() const {
    return create && finalize && release && enqueue && ndrange && copyBuffer &&
           fillBuffer && barrier;
  }

  bool images() const {
    return copyImage && copyBufferToImage && copyImageToBuffer;
  }

  bool mutableDispatch() const { return updateMutable && mutableArgs; }
};

class EventOpenCL : public Event {
 public:
  opencl::ptr<cl_event> event;

  EventOpenCL(opencl::ptr<cl_event> event_);

  virtual void wait() override;
  virtual bool isComplete() const override;
  virtual double timestamp() const override;
  virtual double elapsed(const Event& other) const override;
};

class StreamOpenCL : public Stream {
 public:
  opencl::ptr<cl_command_queue> queue;
  opencl::array<cl_event> events;
  bool outOfOrder;

  StreamOpenCL(opencl::ptr<cl_command_queue> queue_);
  StreamOpenCL(const DeviceOpenCL& dev, const StreamOptions& options = {});
  ~StreamOpenCL();

  virtual void sync() override;
  virtual std::shared_ptr<Event> record() override;
  virtual void waitForEvent(const std::shared_ptr<Event>& e) override;
  virtual void barrier() override;
  void addEvent();
  cl_event* event();

  /// @brief Event slot used by owned-handle (HostBytes) overloads. Unlike
  /// @c event(), this always returns the slot regardless of @c outOfOrder so
  /// the caller can request an event to retain on the pending-memory list,
  /// even on in-order queues.
  cl_event* eventForOwned();

  /// @brief Capture a reference to @p owner alongside @c lastEvent. The
  /// owner is released when the pending-memory list is reaped (sync, or
  /// opportunistic poll at the next enqueue). Caller must have populated
  /// @c lastEvent via @c eventForOwned() in the immediately preceding
  /// enqueue call.
  void retainHostUntilDone(std::shared_ptr<void> owner);

  /// @brief Drop entries whose events have completed. Cheap to call; one
  /// @c clGetEventInfo per pending entry.
  void reapPendingHostMemory();

 protected:
  opencl::ptr<cl_event> lastEvent;

  /// @brief Owners whose host memory must outlive a queued DMA. Populated
  /// by @c retainHostUntilDone; cleared by @c sync() (bulk) and
  /// @c reapPendingHostMemory (opportunistic).
  struct PendingHostMemory {
    opencl::ptr<cl_event> event;
    std::shared_ptr<void> owner;
  };

  std::vector<PendingHostMemory> pendingHostMemory;
};

/// @brief Record-and-replay @c CommandBuffer for OpenCL, adding native
/// interop via @ref encodeNative.
///
/// The OpenCL backend has no native command-buffer concept that maps onto
/// Ghost's recording cb (the @c cl_khr_command_buffer extension records a
/// closed Khronos-blessed command set that external libraries like clBLAS
/// / clBlast / MIOpen do not use), so the cb still replays its variants
/// directly onto the target stream's @c cl_command_queue at submit time.
/// This subclass exists to add @ref encodeNative on top of the default
/// @ref RecordedCommandBuffer machinery.
class CommandBufferOpenCL : public RecordedCommandBuffer {
 public:
  explicit CommandBufferOpenCL(const DeviceOpenCL* device = nullptr)
      : _device(device) {}

  /// @brief Compile to a native @c cl_khr_command_buffer-backed
  /// @ref ExecutableOpenCL when the device supports the extension and the
  /// recorded sequence is fully recordable (dispatch / device-to-device
  /// buffer copy / fill / barrier). Otherwise falls back to command replay,
  /// or throws when @c CompileOptions::requireAccelerated is set.
  std::shared_ptr<Executable> compile(const CompileOptions& options) override;

  std::shared_ptr<RecordedCommandBuffer> cloneEmpty() const override;

  /// @brief Defer a native OpenCL encoding step to submit-time replay.
  ///
  /// At replay, @p body is invoked with the target stream's
  /// @c cl_command_queue. Issue your work via @c clEnqueueXxx on
  /// @p queue. Ordering follows the queue's properties: an in-order
  /// queue (the default) chains automatically with adjacent Ghost
  /// dispatches; an out-of-order queue requires the body to manage
  /// its own event dependencies.
  ///
  /// Body contract:
  ///   1. All work must be enqueued on @p queue.
  ///   2. Do not call @c clFinish / @c clWaitForEvents on @p queue.
  void encodeNative(std::function<void(cl_command_queue queue)> body);

 protected:
  void replayEncodeNative(const EncodeNativeCmd& cmd,
                          const ghost::Stream& stream) override;

 private:
  const DeviceOpenCL* _device;
};

/// @brief @c cl_khr_command_buffer-backed @ref Executable.
///
/// Records the gated command sequence into a @c cl_command_buffer_khr via the
/// @c clCommand*KHR entry points and replays it with a single
/// @c clEnqueueCommandBufferKHR per @ref submit. The native command buffer is
/// built lazily on first submit (it must be created against the target
/// stream's @c cl_command_queue) and rebuilt if the target queue changes or on
/// @ref update.
class ExecutableOpenCL : public Executable {
 public:
  ExecutableOpenCL(const DeviceOpenCL* device, const CommandBufferExtCL* ext,
                   std::vector<Command> commands);
  ~ExecutableOpenCL() override;

  void submit(const ghost::Stream& stream) override;
  void update(const std::vector<Command>& commands) override;

  bool accelerated() const override { return true; }

  bool lastUpdatePatched() const override { return _lastPatched; }

 private:
  // Build (or rebuild) the native command buffer against @p queue.
  void build(cl_command_queue queue);
  void releaseCommandBuffer();
  // Patch the recorded dispatches' kernel args in place via
  // cl_khr_command_buffer_mutable_dispatch. Returns false (→ rebuild) when the
  // cb wasn't built mutable or @p newCommands changed structure.
  bool tryMutableUpdate(const std::vector<Command>& newCommands);

  // A dispatch recorded with a mutable handle (only populated when the cb was
  // built mutable). One entry per DispatchCmd, in record order.
  struct MutableDispatch {
    cl_mutable_command_khr handle;
    cl_kernel kernel;
    cl_uint dims;
    size_t global[3];
    size_t local[3];
    bool localDefined;
  };

  const DeviceOpenCL* _device;
  const CommandBufferExtCL* _ext;
  std::vector<Command> _commands;
  cl_command_buffer_khr _cb = nullptr;
  cl_command_queue _builtForQueue = nullptr;
  std::vector<MutableDispatch> _mutableDispatches;
  bool _mutable = false;
  // Whether the most recent update() patched in place (mutable_dispatch) vs
  // rebuilt the command buffer.
  bool _lastPatched = false;
};

class BufferOpenCL : public Buffer {
 public:
  opencl::ptr<cl_mem> mem;
  size_t _size;

  BufferOpenCL(opencl::ptr<cl_mem> mem_, size_t bytes);
  BufferOpenCL(const DeviceOpenCL& dev, size_t bytes,
               const BufferOptions& opts = {});
  ~BufferOpenCL();

  virtual size_t size() const override;

  virtual void copy(const ghost::Encoder& s, const ghost::Buffer& src,
                    size_t bytes) override;
  virtual void copy(const ghost::Encoder& s, const void* src,
                    size_t bytes) override;
  virtual void copyTo(const ghost::Encoder& s, void* dst,
                      size_t bytes) const override;

  virtual void copy(const ghost::Encoder& s, const ghost::Buffer& src,
                    size_t srcOffset, size_t dstOffset, size_t bytes) override;
  virtual void copy(const ghost::Encoder& s, const void* src, size_t dstOffset,
                    size_t bytes) override;
  virtual void copyTo(const ghost::Encoder& s, void* dst, size_t srcOffset,
                      size_t bytes) const override;

  virtual void copy(const ghost::Encoder& s, HostBytes src, size_t dstOffset,
                    size_t bytes) override;
  virtual void copyTo(const ghost::Encoder& s, HostBytes dst, size_t srcOffset,
                      size_t bytes) const override;

  virtual void fill(const ghost::Encoder& s, size_t offset, size_t size,
                    uint8_t value) override;
  virtual void fill(const ghost::Encoder& s, size_t offset, size_t size,
                    const void* pattern, size_t patternSize) override;

  virtual std::shared_ptr<Buffer> createSubBuffer(
      const std::shared_ptr<Buffer>& self, size_t offset, size_t size) override;
};

class SubBufferOpenCL : public BufferOpenCL {
 public:
  std::shared_ptr<Buffer> _parent;
  size_t _offset;

  SubBufferOpenCL(std::shared_ptr<Buffer> parent, opencl::ptr<cl_mem> mem_,
                  size_t bytes, size_t offset);

  virtual std::shared_ptr<Buffer> createSubBuffer(
      const std::shared_ptr<Buffer>& self, size_t offset, size_t size) override;
};

class MappedBufferOpenCL : public BufferOpenCL {
 public:
  size_t length;
  void* ptr;

  MappedBufferOpenCL(opencl::ptr<cl_mem> mem_, size_t bytes, size_t allocSize);
  MappedBufferOpenCL(const DeviceOpenCL& dev, size_t bytes,
                     const BufferOptions& opts = {});
  ~MappedBufferOpenCL();

  virtual void* map(const ghost::Encoder& s, Access access,
                    bool sync = true) override;
  virtual void unmap(const ghost::Encoder& s) override;
};

class ImageOpenCL : public Image {
 public:
  opencl::ptr<cl_mem> mem;
  ImageDescription descr;

  ImageOpenCL(opencl::ptr<cl_mem> mem_, const ImageDescription& descr);
  ImageOpenCL(const DeviceOpenCL& dev, const ImageDescription& descr);
  ImageOpenCL(const DeviceOpenCL& dev, const ImageDescription& descr,
              BufferOpenCL& buffer);
  ImageOpenCL(const DeviceOpenCL& dev, const ImageDescription& descr,
              ImageOpenCL& image);
  ~ImageOpenCL();

  virtual const ImageDescription& description() const override { return descr; }

  virtual void copy(const ghost::Encoder& s, const ghost::Image& src) override;
  virtual void copy(const ghost::Encoder& s, const ghost::Buffer& src,
                    const BufferLayout& layout) override;
  virtual void copy(const ghost::Encoder& s, const void* src,
                    const BufferLayout& layout) override;
  virtual void copyTo(const ghost::Encoder& s, ghost::Buffer& dst,
                      const BufferLayout& layout) const override;
  virtual void copyTo(const ghost::Encoder& s, void* dst,
                      const BufferLayout& layout) const override;
  virtual void copy(const ghost::Encoder& s, HostBytes src,
                    const BufferLayout& layout) override;
  virtual void copyTo(const ghost::Encoder& s, HostBytes dst,
                      const BufferLayout& layout) const override;
  virtual void copy(const ghost::Encoder& s, const ghost::Buffer& src,
                    const BufferLayout& layout,
                    const Origin3& imageOrigin) override;
  virtual void copyTo(const ghost::Encoder& s, ghost::Buffer& dst,
                      const BufferLayout& layout,
                      const Origin3& imageOrigin) const override;
  virtual void copy(const ghost::Encoder& s, const ghost::Image& src,
                    const Size3& region, const Origin3& srcOrigin,
                    const Origin3& dstOrigin) override;
};

class BufferPool {
 public:
  struct BufferEntry {
    opencl::ptr<cl_mem> mem;
    size_t bytes;
  };

  struct ImageEntry {
    opencl::ptr<cl_mem> mem;
    ImageDescription descr;
    size_t bytes;
  };

  ~BufferPool();

  size_t getLimit() const;
  void setLimit(size_t limit);

  opencl::ptr<cl_mem> lookupBuffer(size_t bytes);
  opencl::ptr<cl_mem> lookupImage(const ImageDescription& descr);

  void reserve(size_t bytes);
  void recycleBuffer(opencl::ptr<cl_mem> mem, size_t bytes);
  void recycleImage(opencl::ptr<cl_mem> mem, const ImageDescription& descr,
                    size_t bytes);
  void clear();

 private:
  void purge(size_t needed = 0);
  static bool imageMatch(const ImageDescription& a, const ImageDescription& b);

  std::list<BufferEntry> _buffers;
  std::list<ImageEntry> _images;
  size_t _current = 0;
  size_t _limit = 0;
};

class PooledBufferOpenCL : public BufferOpenCL {
 public:
  std::shared_ptr<BufferPool> pool;

  PooledBufferOpenCL(opencl::ptr<cl_mem> mem_, size_t bytes,
                     std::shared_ptr<BufferPool> pool_);
  ~PooledBufferOpenCL();
};

class PooledImageOpenCL : public ImageOpenCL {
 public:
  std::shared_ptr<BufferPool> pool;
  size_t imageBytes;

  PooledImageOpenCL(opencl::ptr<cl_mem> mem_, const ImageDescription& descr,
                    size_t bytes, std::shared_ptr<BufferPool> pool_);
  ~PooledImageOpenCL();
};

class DeviceOpenCL : public Device {
 public:
  opencl::ptr<cl_context> context;
  opencl::ptr<cl_command_queue> queue;

  DeviceOpenCL(const SharedContext& share);
  DeviceOpenCL(const GpuInfo& info);
  DeviceOpenCL(cl_platform_id platform, cl_device_id device);

  virtual ghost::Library loadLibraryFromText(
      const std::string& text,
      const CompilerOptions& options = CompilerOptions(),
      bool retainBinary = false) const override;
  virtual ghost::Library loadLibraryFromData(
      const void* data, size_t len,
      const CompilerOptions& options = CompilerOptions(),
      bool retainBinary = false) const override;

  virtual SharedContext shareContext() const override;

  virtual ghost::Stream createStream(
      const StreamOptions& options = {}) const override;

  virtual std::shared_ptr<CommandBuffer> createCommandBuffer(
      const CommandBufferOptions& options = {}) const override;

  virtual size_t getMemoryPoolSize() const override;
  virtual void setMemoryPoolSize(size_t bytes) override;
  virtual ghost::Buffer allocateBuffer(
      size_t bytes, const BufferOptions& opts = {}) const override;
  virtual ghost::MappedBuffer allocateMappedBuffer(
      size_t bytes, const BufferOptions& opts = {}) const override;
  virtual ghost::Image allocateImage(
      const ImageDescription& descr) const override;
  virtual ghost::Image sharedImage(const ImageDescription& descr,
                                   ghost::Buffer& buffer) const override;
  virtual ghost::Image sharedImage(const ImageDescription& descr,
                                   ghost::Image& image) const override;

  virtual ghost::Buffer wrapBuffer(const SharedBuffer& shared) const override;
  virtual ghost::Image wrapImage(const SharedImage& shared) const override;

  virtual Attribute getAttribute(DeviceAttributeId what) const override;
  virtual size_t imageAlignment(const ImageDescription& descr) const override;

  bool checkVersion(const std::string& version) const;
  bool checkExtension(const std::string& extension) const;

  /// @brief The loaded @c cl_khr_command_buffer entry points, or @c nullptr if
  /// the extension is unsupported or its functions failed to resolve.
  const CommandBufferExtCL* commandBufferExt() const;

  std::vector<cl_device_id> getDevices() const;
  cl_platform_id getPlatform() const;
  cl_ulong getInt(cl_device_info param_name) const;
  std::string getString(cl_device_info param_name) const;
  std::string getPlatformString(cl_platform_info param_name) const;

 private:
  std::string _version;
  std::set<std::string> _extensions;
  bool _fullProfile;
  mutable std::shared_ptr<BufferPool> _pool;
  // Resolved lazily on first commandBufferExt() query; _cmdBufExtLoaded guards
  // the one-time resolution.
  mutable CommandBufferExtCL _cmdBufExt;
  mutable bool _cmdBufExtLoaded = false;

  void setVersion();
};
}  // namespace implementation
}  // namespace ghost

#endif
