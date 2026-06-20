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

#if WITH_CUDA

// <delayimp.h> requires immediate-after-<windows.h> ordering, which the
// transitive <cuda.h> include downstream breaks. Forward-declare the one
// helper we use instead of including it.
#if defined(WITH_CUDA_DELAYLOAD) && defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
extern "C" HRESULT __stdcall __HrLoadAllImportsForDll(const char* szDll);
#ifndef FACILITY_VISUALCPP
#define FACILITY_VISUALCPP 0x6d
#endif
#ifndef VcppException
#define VcppException(sev, code) ((sev) | (FACILITY_VISUALCPP << 16) | (code))
#endif
#endif

#include <ghost/allocator.h>
#include <ghost/cuda/device.h>
#include <ghost/cuda/exception.h>
#include <ghost/cuda/impl_device.h>
#include <ghost/cuda/impl_function.h>
#include <ghost/exception.h>

#include <atomic>
#include <cstring>
#include <new>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace ghost {
namespace {
#if defined(WITH_CUDA_DELAYLOAD) && defined(_WIN32)
// SEH-required: __HrLoadAllImportsForDll raises a Win32 exception
// (ERROR_MOD_NOT_FOUND) when nvcuda.dll is absent, not a C++ throw.
bool probeCudaDriverOnce() {
  bool ok = true;
  __try {
    if (FAILED(__HrLoadAllImportsForDll("nvcuda.dll"))) {
      ok = false;
    }
  } __except (GetExceptionCode() ==
                      VcppException(ERROR_SEVERITY_ERROR, ERROR_MOD_NOT_FOUND)
                  ? EXCEPTION_EXECUTE_HANDLER
                  : EXCEPTION_CONTINUE_SEARCH) {
    ok = false;
  }
  return ok;
}

bool isCudaDriverAvailable() {
  static const bool available = probeCudaDriverOnce();
  return available;
}
#else
inline bool isCudaDriverAvailable() { return true; }
#endif
}  // namespace

namespace implementation {
using namespace cu;

namespace {
// Shared state for a deferred CUdeviceptr release. The cu::ptr inside owns
// the device memory; when the last scheduled host-fn callback runs, the
// DeferredRelease is deleted and the cu::ptr destructor calls cuMemFree.
//
// Fallback path only (drivers without stream-ordered allocation): host
// functions are forbidden from making CUDA API calls, so the cuMemFree the
// destructor performs here is off-spec. The preferred path below frees with
// cuMemFreeAsync and involves no host callback at all.
struct DeferredRelease {
  std::atomic<int> remaining{0};
  cu::ptr<CUdeviceptr> mem;

  DeferredRelease() : mem(0, false) {}
};

void CUDA_CB releaseDeferredCallback(void* p) {
  auto* d = static_cast<DeferredRelease*>(p);
  if (d->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    delete d;
  }
}

// Enqueue a stream-ordered free of @p mem, ordered after the work currently
// pending on every stream in @p streams: streams[1..] are chained behind
// streams[0] with events, then cuMemFreeAsync executes on streams[0] as
// soon as the dependent work drains. Pool-backed memory returns to its pool
// for reuse; cuMemAlloc'd memory is released without the device-wide
// synchronization a plain cuMemFree on a busy device implies.
// Returns false (having enqueued at most harmless event waits) when the
// driver or device lacks stream-ordered allocation support, or on any
// API failure — the caller falls back to the host-fn path.
bool tryStreamOrderedRelease(cu::ptr<CUdeviceptr>& mem,
                             const std::vector<CUstream>& streams) {
#if CUDA_VERSION >= 11020
  // cuMemFreeAsync resolving to CUDA_ERROR_NOT_INITIALIZED means the loader
  // found no such symbol (driver < 11.2) — remember that and stop retrying.
  static std::atomic<bool> driverHasFreeAsync{true};
  if (!driverHasFreeAsync.load(std::memory_order_relaxed)) return false;
  CUstream freeStream = streams[0];
  for (size_t i = 1; i < streams.size(); ++i) {
    cu::ptr<CUevent> ev;
    CUresult err = cuEventCreate(&ev, CU_EVENT_DISABLE_TIMING);
    if (err == CUDA_SUCCESS) err = cuEventRecord(ev, streams[i]);
    if (err == CUDA_SUCCESS) err = cuStreamWaitEvent(freeStream, ev, 0);
    if (err != CUDA_SUCCESS) return false;
  }
  CUresult err = cuMemFreeAsync(mem.get(), freeStream);
  if (err != CUDA_SUCCESS) {
    if (err == CUDA_ERROR_NOT_INITIALIZED) {
      driverHasFreeAsync.store(false, std::memory_order_relaxed);
    }
    return false;
  }
  mem.release();
  return true;
#else
  (void)mem;
  (void)streams;
  return false;
#endif
}

// Free @p mem after pending work on each stream in @p streams has completed.
// Preferred: a stream-ordered cuMemFreeAsync (no host callback, free runs as
// part of normal stream progress). Fallback: one cuLaunchHostFunc per stream.
// On total scheduling failure, the cu::ptr's destructor runs synchronously
// (the caller has already violated the lifetime contract).
// @p allocStream orders the free of pool-backed memory that was never used
// on any stream, so it still returns to its pool instead of hitting the
// synchronous cuMemFree path.
void scheduleDeferredRelease(cu::ptr<CUdeviceptr>&& mem,
                             const std::vector<CUstream>& streams,
                             CUstream allocStream = nullptr) {
  if (!mem.get() || !mem.owned()) return;  // nothing to free
  if (streams.empty()) {
    if (allocStream) {
      std::vector<CUstream> alloc{allocStream};
      tryStreamOrderedRelease(mem, alloc);
    }
    return;  // remaining-owner path releases synchronously
  }
  if (tryStreamOrderedRelease(mem, streams)) return;
  auto* d = new DeferredRelease{};
  d->remaining.store(static_cast<int>(streams.size()),
                     std::memory_order_relaxed);
  d->mem = std::move(mem);
  for (CUstream s : streams) {
    CUresult err = cuLaunchHostFunc(s, &releaseDeferredCallback, d);
    if (err != CUDA_SUCCESS) {
      // Decrement on this stream's behalf so the remaining callbacks (or the
      // synchronous fallback below) get the count right.
      if (d->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        delete d;
        return;
      }
    }
  }
}

// Resolve recorded stream uses into the streams that are still alive.
// @p anyExpired is set when at least one use stream has been destroyed —
// in-flight work on it can no longer be ordered against, so the caller must
// fall back to a context-wide drain before freeing.
std::vector<CUstream> liveUseStreams(const std::vector<StreamUse>& uses,
                                     bool* anyExpired) {
  std::vector<CUstream> live;
  live.reserve(uses.size());
  *anyExpired = false;
  for (const auto& use : uses) {
    if (use.alive.expired()) {
      *anyExpired = true;
    } else {
      live.push_back(use.stream);
    }
  }
  return live;
}
}  // namespace

EventCUDA::EventCUDA(cu::ptr<CUevent> event_) : event(event_) {}

void EventCUDA::wait() {
  CUresult err = cuEventSynchronize(event);
  checkError(err);
}

bool EventCUDA::isComplete() const {
  CUresult err = cuEventQuery(event);
  if (err == CUDA_SUCCESS) return true;
  if (err == CUDA_ERROR_NOT_READY) return false;
  checkError(err);
  return false;
}

double EventCUDA::elapsed(const Event& other) const {
  auto& otherCUDA = static_cast<const EventCUDA&>(other);
  float ms = 0.0f;
  CUresult err = cuEventElapsedTime(&ms, event, otherCUDA.event);
  if (err != CUDA_SUCCESS) return 0.0;
  return static_cast<double>(ms) / 1000.0;
}

StreamCUDA::StreamCUDA(cu::ptr<CUstream> queue_) : queue(queue_) {}

StreamCUDA::StreamCUDA(CUcontext dev) {
  CUresult err;
  err = cuStreamCreate(&queue, CU_STREAM_NON_BLOCKING);
  checkError(err);
}

StreamCUDA::~StreamCUDA() {
  // Drain before letting owner deleters run for any host memory still held
  // alive by a queued DMA. Same reasoning as ~StreamOpenCL().
  if (!pendingHostMemory.empty()) {
    if (queue)
      cuStreamSynchronize(queue);
    else
      cuCtxSynchronize();
  }
}

void StreamCUDA::sync() {
  CUresult err;
  if (queue)
    err = cuStreamSynchronize(queue);
  else
    err = cuCtxSynchronize();
  checkError(err);
  // After sync, every enqueued op has completed → drop all retained owners
  // (which will run their user-supplied deleters here on this thread).
  pendingHostMemory.clear();
}

void StreamCUDA::retainHostUntilDone(std::shared_ptr<void> owner) {
  if (!owner) return;
  reapPendingHostMemory();
  cu::ptr<CUevent> ev;
  // CU_EVENT_DISABLE_TIMING: we only need completion signaling, not the
  // per-op latency tracking that timing-enabled events incur.
  CUresult err = cuEventCreate(&ev, CU_EVENT_DISABLE_TIMING);
  if (err == CUDA_SUCCESS) err = cuEventRecord(ev, queue);
  if (err != CUDA_SUCCESS) {
    // Couldn't get a per-op event → still keep the owner alive, but fall back
    // to bulk release on sync() (null-event entry).
    pendingHostMemory.push_back({cu::ptr<CUevent>(), std::move(owner)});
    return;
  }
  pendingHostMemory.push_back({std::move(ev), std::move(owner)});
}

void StreamCUDA::reapPendingHostMemory() {
  auto it = pendingHostMemory.begin();
  while (it != pendingHostMemory.end()) {
    if (it->event.get() == nullptr) {
      // Null event → wait for sync().
      ++it;
      continue;
    }
    CUresult err = cuEventQuery(it->event);
    if (err == CUDA_SUCCESS) {
      it = pendingHostMemory.erase(it);
    } else {
      ++it;
    }
  }
}

std::shared_ptr<Event> StreamCUDA::record() {
  cu::ptr<CUevent> ev;
  CUresult err;
  err = cuEventCreate(&ev, CU_EVENT_DEFAULT);
  checkError(err);
  err = cuEventRecord(ev, queue);
  checkError(err);
  return std::make_shared<EventCUDA>(ev);
}

void StreamCUDA::waitForEvent(const std::shared_ptr<Event>& e) {
  auto eventCUDA = static_cast<EventCUDA*>(e.get());
  CUresult err = cuStreamWaitEvent(queue, eventCUDA->event, 0);
  checkError(err);
}

void StreamCUDA::barrier() {
  // No host drain. A CUDA stream executes its enqueued operations in issue
  // order, so the op after a CommandBuffer barrier already happens-after the
  // op before it. The default Stream::barrier() would cuStreamSynchronize()
  // mid-replay, forcing a full host stall on every cb.barrier(). Host
  // visibility is provided by the caller's Stream::sync() after submit().
}

void CommandBufferCUDA::encodeNative(std::function<void(CUstream)> body) {
  addEncodeNative([body = std::move(body)](void* ctx) {
    body(static_cast<CUstream>(ctx));
  });
}

void CommandBufferCUDA::replayEncodeNative(const EncodeNativeCmd& cmd,
                                           const ghost::Stream& stream) {
  auto* sCu = static_cast<StreamCUDA*>(stream.impl().get());
  cmd.body(sCu->queue.get());
}

namespace {

// Whether a single recorded command can be issued during CUDA stream capture.
// Device-side dispatch / copy / fill / image-blit and the (capture-time no-op)
// barrier are fine. Host transfers (they may stage synchronously), indirect
// dispatch, cross-stream events and native interop are not reliably
// capturable, so their presence forces the replay fallback.
bool isCapturable(const Command& command) {
  return std::visit(
      [](const auto& cmd) -> bool {
        using T = std::decay_t<decltype(cmd)>;
        if constexpr (std::is_same_v<T, DispatchCmd>) {
          // cuLaunchCooperativeKernel cannot be captured.
          return !cmd.launchArgs.is_cooperative();
        } else if constexpr (std::is_same_v<T, CopyBufferCmd> ||
                             std::is_same_v<T, FillBufferCmd> ||
                             std::is_same_v<T, FillBufferPatternCmd> ||
                             std::is_same_v<T, CopyImageCmd> ||
                             std::is_same_v<T, CopyImageFromBufferCmd> ||
                             std::is_same_v<T, CopyImageToBufferCmd> ||
                             std::is_same_v<T, BarrierCmd>) {
          return true;
        } else {
          return false;
        }
      },
      command);
}

// Dispatch/barrier-only sequences are eligible for the in-place node-param
// fast update: every captured node is a kernel node, so we can map them 1:1 to
// the recorded dispatches and patch each via cuGraphExecKernelNodeSetParams.
bool isDispatchOnly(const std::vector<Command>& commands) {
  for (const auto& command : commands) {
    bool ok = std::visit(
        [](const auto& cmd) -> bool {
          using T = std::decay_t<decltype(cmd)>;
          return std::is_same_v<T, DispatchCmd> ||
                 std::is_same_v<T, BarrierCmd>;
        },
        command);
    if (!ok) return false;
  }
  return true;
}

// Reconstruct the record order of a single-stream-captured graph's nodes.
// Capture on one stream yields a linear chain (each node depends only on its
// predecessor); we walk dep→node from the root. Returns {} if the graph isn't
// a single linear chain (branching/disconnected) — the caller then forgoes the
// patch path and uses re-capture.
std::vector<CUgraphNode> orderedNodes(CUgraph graph) {
  size_t num = 0;
  if (cuGraphGetNodes(graph, nullptr, &num) != CUDA_SUCCESS || num == 0)
    return {};
  std::vector<CUgraphNode> nodes(num);
  if (cuGraphGetNodes(graph, nodes.data(), &num) != CUDA_SUCCESS) return {};

  std::unordered_map<CUgraphNode, CUgraphNode> succ;  // predecessor -> node
  CUgraphNode root = nullptr;
  for (CUgraphNode n : nodes) {
    size_t nd = 0;
    if (cuGraphNodeGetDependencies(n, nullptr, &nd) != CUDA_SUCCESS) return {};
    if (nd == 0) {
      if (root) return {};  // more than one root -> not a linear chain
      root = n;
      continue;
    }
    if (nd != 1) return {};  // a node with multiple deps -> branch
    CUgraphNode dep = nullptr;
    if (cuGraphNodeGetDependencies(n, &dep, &nd) != CUDA_SUCCESS) return {};
    if (!succ.emplace(dep, n).second) return {};  // 2 successors -> branch
  }
  if (!root) return {};

  std::vector<CUgraphNode> ordered;
  ordered.reserve(nodes.size());
  for (CUgraphNode cur = root; cur;) {
    ordered.push_back(cur);
    auto it = succ.find(cur);
    cur = (it == succ.end()) ? nullptr : it->second;
  }
  if (ordered.size() != nodes.size()) return {};  // disconnected
  return ordered;
}

// CUDA graph instantiate flags from CompileOptions (no-ops on older drivers
// that lack the flag macros).
unsigned long long graphInstantiateFlags(const CompileOptions& options) {
  unsigned long long flags = 0;
#ifdef CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD
  if (options.uploadOnCompile) flags |= CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD;
#endif
#ifdef CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
  if (options.autoFreeOnLaunch)
    flags |= CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
#endif
  return flags;
}

bool isSequenceCapturable(const std::vector<Command>& commands) {
  for (const auto& c : commands)
    if (!isCapturable(c)) return false;
  return true;
}

}  // namespace

std::shared_ptr<RecordedCommandBuffer> CommandBufferCUDA::cloneEmpty() const {
  return std::make_shared<CommandBufferCUDA>();
}

CUgraph CommandBufferCUDA::captureGraph() {
  cu::ptr<CUstream> capStream;
  checkError(cuStreamCreate(&capStream, CU_STREAM_NON_BLOCKING));
  auto streamImpl = std::make_shared<StreamCUDA>(capStream);
  ghost::Stream capStreamWrap(streamImpl);

  checkError(cuStreamBeginCapture(capStream, CU_STREAM_CAPTURE_MODE_RELAXED));

  CUgraph graph = nullptr;
  try {
    // Reuse the per-op replay machinery: each recorded command issues its
    // native CUDA work onto the capturing stream, building graph nodes.
    replayInto(capStreamWrap, capStreamWrap);
  } catch (...) {
    // Leave the stream out of capture mode before propagating.
    cuStreamEndCapture(capStream, &graph);
    if (graph) cuGraphDestroy(graph);
    throw;
  }

  checkError(cuStreamEndCapture(capStream, &graph));
  return graph;
}

CUgraphExec CommandBufferCUDA::captureGraphExec() {
  CUgraph graph = captureGraph();
  CUgraphExec exec = nullptr;
  CUresult err = cuGraphInstantiateWithFlags(&exec, graph, 0);
  cuGraphDestroy(graph);
  checkError(err);
  return exec;
}

std::shared_ptr<Executable> CommandBufferCUDA::compile(
    const CompileOptions& options) {
  if (!isSequenceCapturable(commands)) {
    if (options.requireAccelerated) throw ghost::unsupported_error();
    // Degrade to the replay-backed Executable (correct, not accelerated).
    return RecordedCommandBuffer::compile(options);
  }
  CUgraph graph = captureGraph();
  CUgraphExec exec = nullptr;
  CUresult err =
      cuGraphInstantiateWithFlags(&exec, graph, graphInstantiateFlags(options));
  if (err != CUDA_SUCCESS) {
    cuGraphDestroy(graph);
    checkError(err);
  }

  // For a dispatch-only graph, map the captured kernel nodes 1:1 to the
  // recorded dispatches and retain the graph so update() can patch node params
  // in place (cuGraphExecKernelNodeSetParams) without re-capturing.
  std::vector<CUgraphNode> dispatchNodes;
  CUgraph retained = nullptr;
  if (isDispatchOnly(commands)) {
    std::vector<CUgraphNode> nodes = orderedNodes(graph);
    size_t numDispatch = 0;
    for (auto& c : commands)
      if (std::get_if<DispatchCmd>(&c)) numDispatch++;
    if (!nodes.empty() && nodes.size() == numDispatch) {
      dispatchNodes = std::move(nodes);
      retained = graph;
    }
  }
  if (!retained) cuGraphDestroy(graph);
  return std::make_shared<ExecutableCUDA>(exec, retained,
                                          std::move(dispatchNodes), commands);
}

ExecutableCUDA::ExecutableCUDA(CUgraphExec exec, CUgraph graph,
                               std::vector<CUgraphNode> dispatchNodes,
                               std::vector<Command> commands)
    : _exec(exec),
      _graph(graph),
      _dispatchNodes(std::move(dispatchNodes)),
      _commands(std::move(commands)) {}

ExecutableCUDA::~ExecutableCUDA() {
  if (_exec) {
    // Draining the context guarantees no in-flight graph launch still
    // references _exec (or the buffers baked into it) before teardown.
    if (_launched) cuCtxSynchronize();
    cuGraphExecDestroy(_exec);
  }
  if (_graph) cuGraphDestroy(_graph);
}

void ExecutableCUDA::submit(const ghost::Stream& stream) {
  auto* s = static_cast<StreamCUDA*>(stream.impl().get());
  checkError(cuGraphLaunch(_exec, s->queue));
  _launched = true;
}

bool ExecutableCUDA::tryPatchUpdate(const std::vector<Command>& newCommands) {
  if (_dispatchNodes.empty()) return false;  // not a patchable graph
  if (newCommands.size() != _commands.size()) return false;

  // Topology must be unchanged and each dispatch must keep its kernel (the
  // captured node is bound to that CUfunction); only args/dims may differ.
  size_t di = 0;
  for (size_t i = 0; i < newCommands.size(); i++) {
    if (newCommands[i].index() != _commands[i].index()) return false;
    if (const auto* d = std::get_if<DispatchCmd>(&newCommands[i])) {
      if (di >= _dispatchNodes.size()) return false;
      const auto* oldD = std::get_if<DispatchCmd>(&_commands[i]);
      auto* fnNew = static_cast<FunctionCUDA*>(d->function.get());
      auto* fnOld = static_cast<FunctionCUDA*>(oldD->function.get());
      if (fnNew->kernel != fnOld->kernel) return false;
      di++;
    }
  }
  if (di != _dispatchNodes.size()) return false;

  // Patch each kernel node. The param-pointer arrays (storages) and the new
  // commands' Attributes/buffers must stay alive across the SetParams calls,
  // which copy the dereferenced values; both do (storages here, newCommands
  // is the caller's argument).
  std::vector<std::vector<void*>> storages(_dispatchNodes.size());
  di = 0;
  for (size_t i = 0; i < newCommands.size(); i++) {
    if (const auto* d = std::get_if<DispatchCmd>(&newCommands[i])) {
      auto* fn = static_cast<FunctionCUDA*>(d->function.get());
      CUDA_KERNEL_NODE_PARAMS np;
      fn->buildKernelNodeParams(d->launchArgs, d->args, storages[di], np);
      if (cuGraphExecKernelNodeSetParams(_exec, _dispatchNodes[di], &np) !=
          CUDA_SUCCESS)
        return false;  // partial patch is fine: caller rebuilds the exec
      di++;
    }
  }
  return true;
}

void ExecutableCUDA::update(const std::vector<Command>& commands) {
  // Fast path: patch the dispatch kernel-node params in place — no re-capture,
  // no re-instantiate. This is the per-frame inference case (same topology,
  // new buffer pointers / scalars). The caller must not call update() while a
  // prior submit() is still executing; sync the stream first (the typical
  // per-frame loop already does before reusing the results).
  if (tryPatchUpdate(commands)) {
    _commands = commands;
    _lastPatched = true;
    return;
  }
  _lastPatched = false;

  // Patch path unavailable/failed → the retained graph's node mapping no longer
  // describes the exec we're about to rebuild; drop it.
  if (_graph) {
    cuGraphDestroy(_graph);
    _graph = nullptr;
  }
  _dispatchNodes.clear();

  // Re-capture and either topology-preserving-update or re-instantiate.
  CommandBufferCUDA tmp;
  tmp.commands = commands;
  CUgraph graph = tmp.captureGraph();

  bool updated = false;
  if (_exec) {
#if CUDA_VERSION >= 12000
    CUgraphExecUpdateResultInfo info{};
    updated = (cuGraphExecUpdate(_exec, graph, &info) == CUDA_SUCCESS);
#else
    CUgraphNode errNode = nullptr;
    CUgraphExecUpdateResult res;
    updated = (cuGraphExecUpdate(_exec, graph, &errNode, &res) == CUDA_SUCCESS);
#endif
  }

  if (updated) {
    cuGraphDestroy(graph);
  } else {
    CUgraphExec next = nullptr;
    CUresult err = cuGraphInstantiateWithFlags(&next, graph, 0);
    cuGraphDestroy(graph);
    checkError(err);
    if (_exec) {
      if (_launched) cuCtxSynchronize();
      cuGraphExecDestroy(_exec);
    }
    _exec = next;
    _launched = false;
  }
  _commands = commands;
}

BufferCUDA::BufferCUDA(cu::ptr<CUdeviceptr> mem_, size_t bytes)
    : mem(mem_), _size(bytes) {}

BufferCUDA::~BufferCUDA() {
  bool expired = false;
  std::vector<CUstream> streams = liveUseStreams(_useStreams, &expired);
  if (_allocator) {
    // The host allocator owns this memory and may hand it to a new consumer
    // the moment freeBuffer returns — drain any device work that still
    // references it first.
    if (expired) cuCtxSynchronize();
    for (CUstream s : streams) cuStreamSynchronize(s);
    _allocator->freeBuffer(
        reinterpret_cast<void*>(static_cast<uintptr_t>(mem.release())), _size);
    return;
  }
  if (expired) {
    // A destroyed use stream may still have in-flight work referencing this
    // memory that we can't order against any more — drain everything, after
    // which no deferral is needed.
    cuCtxSynchronize();
    streams.clear();
  }
  // Defer the free until pending work on each used stream completes.
  // Non-owning mem (sub-buffers, sharedImage donors) falls through to the
  // cu::ptr destructor, which is a no-op.
  scheduleDeferredRelease(std::move(mem), streams, _allocStream);
}

void BufferCUDA::markUsed(const StreamCUDA& s) {
  CUstream handle = s.queue.get();
  for (const StreamUse& existing : _useStreams) {
    if (existing.stream == handle) return;
  }
#if CUDA_VERSION >= 11020
  // Pool-backed: the allocation is stream-ordered on _allocStream. Give every
  // other stream a device-side dependency on allocation completion before its
  // first use — this replaces the former host cuStreamSynchronize per pooled
  // allocation.
  if (_ready.get() && handle != _allocStream) {
    if (cuStreamWaitEvent(handle, _ready, 0) != CUDA_SUCCESS) {
      cuEventSynchronize(_ready);  // degraded: one-time host wait
    }
  }
#endif
  _useStreams.push_back({handle, std::weak_ptr<void>(s.aliveToken)});
}

#if CUDA_VERSION >= 11020
void BufferCUDA::setPoolBacked(const cu::ptr<CUmemoryPool>& pool,
                               CUstream allocStream, cu::ptr<CUevent>&& ready) {
  _pool = pool;  // shares ownership — keeps the pool alive
  _allocStream = allocStream;
  _ready = std::move(ready);
}
#endif

BufferCUDA::BufferCUDA(const DeviceCUDA& dev, size_t bytes,
                       const BufferOptions&)
    : _size(bytes) {
  // TODO: honor opts.hint (Staging → cuMemHostAlloc pinned host)
  CUresult err;
  err = cuMemAlloc(&mem, bytes);
  checkError(err);
}

size_t BufferCUDA::size() const { return _size; }

void BufferCUDA::copy(const ghost::Encoder& s, const ghost::Buffer& src,
                      size_t bytes) {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  auto src_impl = static_cast<implementation::BufferCUDA*>(src.impl().get());
  markUsed(*stream_impl);
  src_impl->markUsed(*stream_impl);
  CUresult err;
  err = cuMemcpyDtoDAsync(mem, src_impl->mem, bytes, stream_impl->queue);
  checkError(err);
}

void BufferCUDA::copy(const ghost::Encoder& s, const void* src, size_t bytes) {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  markUsed(*stream_impl);
  CUresult err;
  err = cuMemcpyHtoDAsync(mem, src, bytes, stream_impl->queue);
  checkError(err);
}

void BufferCUDA::copyTo(const ghost::Encoder& s, void* dst,
                        size_t bytes) const {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  const_cast<BufferCUDA*>(this)->markUsed(*stream_impl);
  CUresult err;
  err = cuMemcpyDtoHAsync(dst, mem, bytes, stream_impl->queue);
  checkError(err);
}

void BufferCUDA::copy(const ghost::Encoder& s, const ghost::Buffer& src,
                      size_t srcOffset, size_t dstOffset, size_t bytes) {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  auto src_impl = static_cast<implementation::BufferCUDA*>(src.impl().get());
  markUsed(*stream_impl);
  src_impl->markUsed(*stream_impl);
  CUresult err;
  err =
      cuMemcpyDtoDAsync(mem.get() + dstOffset, src_impl->mem.get() + srcOffset,
                        bytes, stream_impl->queue);
  checkError(err);
}

void BufferCUDA::copy(const ghost::Encoder& s, const void* src,
                      size_t dstOffset, size_t bytes) {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  markUsed(*stream_impl);
  CUresult err;
  err =
      cuMemcpyHtoDAsync(mem.get() + dstOffset, src, bytes, stream_impl->queue);
  checkError(err);
}

void BufferCUDA::copyTo(const ghost::Encoder& s, void* dst, size_t srcOffset,
                        size_t bytes) const {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  const_cast<BufferCUDA*>(this)->markUsed(*stream_impl);
  CUresult err;
  err =
      cuMemcpyDtoHAsync(dst, mem.get() + srcOffset, bytes, stream_impl->queue);
  checkError(err);
}

void BufferCUDA::copy(const ghost::Encoder& s, HostBytes src, size_t dstOffset,
                      size_t bytes) {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  markUsed(*stream_impl);
  CUresult err = cuMemcpyHtoDAsync(mem.get() + dstOffset, src.data(), bytes,
                                   stream_impl->queue);
  checkError(err);
  if (src.ownsBytes()) stream_impl->retainHostUntilDone(src.owner());
}

void BufferCUDA::copyTo(const ghost::Encoder& s, HostBytes dst,
                        size_t srcOffset, size_t bytes) const {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  const_cast<BufferCUDA*>(this)->markUsed(*stream_impl);
  CUresult err = cuMemcpyDtoHAsync(dst.data(), mem.get() + srcOffset, bytes,
                                   stream_impl->queue);
  checkError(err);
  if (dst.ownsBytes()) stream_impl->retainHostUntilDone(dst.owner());
}

void BufferCUDA::fill(const ghost::Encoder& s, size_t offset, size_t size,
                      uint8_t value) {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  markUsed(*stream_impl);
  CUresult err;
  err = cuMemsetD8Async(mem.get() + offset, value, size, stream_impl->queue);
  checkError(err);
}

void BufferCUDA::fill(const ghost::Encoder& s, size_t offset, size_t size,
                      const void* pattern, size_t patternSize) {
  auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
  markUsed(*stream_impl);
  CUresult err;
  CUdeviceptr dst = mem.get() + offset;
  if (patternSize == 1) {
    err = cuMemsetD8Async(dst, *static_cast<const uint8_t*>(pattern), size,
                          stream_impl->queue);
  } else if (patternSize == 2) {
    unsigned short v;
    memcpy(&v, pattern, 2);
    err = cuMemsetD16Async(dst, v, size / 2, stream_impl->queue);
  } else if (patternSize == 4) {
    unsigned int v;
    memcpy(&v, pattern, 4);
    err = cuMemsetD32Async(dst, v, size / 4, stream_impl->queue);
  } else {
    // For non-standard pattern sizes, fill from host
    std::vector<uint8_t> buf(size);
    for (size_t i = 0; i < size; i += patternSize) {
      size_t n = std::min(patternSize, size - i);
      memcpy(buf.data() + i, pattern, n);
    }
    err = cuMemcpyHtoDAsync(dst, buf.data(), size, stream_impl->queue);
  }
  checkError(err);
}

std::shared_ptr<Buffer> BufferCUDA::createSubBuffer(
    const std::shared_ptr<Buffer>& self, size_t offset, size_t size) {
  cu::ptr<CUdeviceptr> subMem(mem.get() + offset, false);
  return std::make_shared<SubBufferCUDA>(self, subMem, size);
}

SubBufferCUDA::SubBufferCUDA(std::shared_ptr<Buffer> parent,
                             cu::ptr<CUdeviceptr> mem_, size_t bytes)
    : BufferCUDA(mem_, bytes), _parent(parent) {}

void SubBufferCUDA::markUsed(const StreamCUDA& s) {
  // Propagate to the parent so the parent's deferred release waits for any
  // pending work referencing this sub-region. Also record locally so direct
  // uses of this sub-buffer wrapper are tracked (defensive: a sub-buffer's
  // mem is non-owning, but recording is cheap and keeps invariants uniform).
  if (auto* p = static_cast<BufferCUDA*>(_parent.get())) {
    p->markUsed(s);
  }
  BufferCUDA::markUsed(s);
}

MappedBufferCUDA::MappedBufferCUDA(cu::ptr<void*> ptr_)
    : BufferCUDA(cu::ptr<CUdeviceptr>(), 0), ptr(ptr_) {
  CUdeviceptr p;
  CUresult err;
  err = cuMemHostGetDevicePointer(&p, ptr, 0);
  checkError(err);
  mem = cu::ptr<CUdeviceptr>(p, false);  // do not free the device pointer
}

MappedBufferCUDA::~MappedBufferCUDA() {
  if (_allocator) {
    // See ~BufferCUDA: drain device work (async DMA through the mapped
    // device pointer) before returning the handle to the host allocator.
    bool expired = false;
    std::vector<CUstream> streams = liveUseStreams(_useStreams, &expired);
    if (expired) cuCtxSynchronize();
    for (CUstream s : streams) cuStreamSynchronize(s);
    // The base BufferCUDA destructor would also try to free `mem`, but
    // `mem` is non-owning (cuMemHostGetDevicePointer derived). Clearing
    // _allocator prevents BufferCUDA::~BufferCUDA from calling freeBuffer.
    _allocator->freeMappedBuffer(ptr.release(), _size);
    _allocator = nullptr;
  }
}

MappedBufferCUDA::MappedBufferCUDA(const DeviceCUDA& dev, size_t bytes,
                                   const BufferOptions& opts)
    : BufferCUDA(cu::ptr<CUdeviceptr>(), bytes) {
  unsigned int flags = CU_MEMHOSTALLOC_DEVICEMAP;
  if (opts.access == Access::WriteOnly) {
    flags |= CU_MEMHOSTALLOC_WRITECOMBINED;
  }
  CUresult err;
  err = cuMemHostAlloc(&ptr, bytes, flags);
  checkError(err);
  CUdeviceptr p;
  err = cuMemHostGetDevicePointer(&p, ptr, 0);
  checkError(err);
  mem = cu::ptr<CUdeviceptr>(p, false);  // do not free the device pointer
}

void* MappedBufferCUDA::map(const ghost::Encoder& s, Access access, bool sync) {
  if (sync) {
    // TODO
  }
  return ptr;
}

void MappedBufferCUDA::unmap(const ghost::Encoder&) {}

ImageCUDA::ImageCUDA(cu::ptr<CUdeviceptr> mem_, const ImageDescription& descr_,
                     const DeviceCUDA* dev)
    : mem(mem_), descr(descr_) {
  _device = dev;
}

ImageCUDA::~ImageCUDA() {
  bool expired = false;
  std::vector<CUstream> streams = liveUseStreams(_useStreams, &expired);
  if (expired) {
    // A destroyed use stream may still have in-flight work we can't order
    // against — drain everything; nothing is pending afterwards.
    cuCtxSynchronize();
    streams.clear();
  }
  if (_allocator) {
    // See ~BufferCUDA: drain device work before returning the handle to the
    // host allocator.
    for (CUstream s : streams) cuStreamSynchronize(s);
    _textures.clear();  // nothing in flight — destroy cached textures now
    _allocator->freeImage(
        reinterpret_cast<void*>(static_cast<uintptr_t>(mem.release())), descr);
    return;
  }
  if (!_textures.empty()) {
    // cuTexObjectDestroy has no stream-ordered form and must not run inside
    // a cuLaunchHostFunc callback (host functions may not call CUDA APIs),
    // so cached textures with work in flight are parked on the device's
    // reap list behind events recorded on the use streams.
    if (streams.empty()) {
      _textures.clear();  // nothing in flight
    } else if (_device) {
      _device->deferTextureRelease(takeTextures(), streams);
    } else {
      // No device to park them on: drain the streams, then destroy.
      for (CUstream s : streams) cuStreamSynchronize(s);
      _textures.clear();
    }
  }
  scheduleDeferredRelease(std::move(mem), streams);
}

std::vector<TexturePtr> ImageCUDA::takeTextures() {
  std::vector<TexturePtr> out;
  out.reserve(_textures.size());
  for (CachedTexture& t : _textures) out.push_back(std::move(t.tex));
  _textures.clear();
  return out;
}

CUtexObject* ImageCUDA::lookupTexture(CUaddress_mode addressMode,
                                      CUfilter_mode filterMode,
                                      bool normalizedCoords) {
  for (CachedTexture& t : _textures) {
    if (t.address == addressMode && t.filter == filterMode &&
        t.normalized == normalizedCoords) {
      return t.tex.handleAddress();
    }
  }

  CUDA_RESOURCE_DESC resDesc;
  CUDA_TEXTURE_DESC texDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  memset(&texDesc, 0, sizeof(texDesc));

  CUarray_format f;
  switch (descr.type) {
    default:
    case DataType_UInt8:
      f = CU_AD_FORMAT_UNSIGNED_INT8;
      break;
    case DataType_UInt16:
      f = CU_AD_FORMAT_UNSIGNED_INT16;
      break;
    case DataType_Int8:
      f = CU_AD_FORMAT_SIGNED_INT8;
      break;
    case DataType_Int16:
      f = CU_AD_FORMAT_SIGNED_INT16;
      break;
    case DataType_Float16:
      f = CU_AD_FORMAT_HALF;
      break;
    case DataType_Float:
      f = CU_AD_FORMAT_FLOAT;
      break;
  };
  texDesc.addressMode[0] = addressMode;
  texDesc.addressMode[1] = addressMode;
  texDesc.filterMode = filterMode;
  if (normalizedCoords) {
    texDesc.flags |= CU_TRSF_NORMALIZED_COORDINATES;
  }

  resDesc.resType = CU_RESOURCE_TYPE_PITCH2D;
  resDesc.res.pitch2D.devPtr = mem.get();
  resDesc.res.pitch2D.format = f;
  resDesc.res.pitch2D.numChannels = (unsigned int)descr.channels;
  resDesc.res.pitch2D.width = descr.size.x;
  resDesc.res.pitch2D.height = descr.size.y;
  // stride.x == 0 means tight packing; resolve to width*pixelSize.
  resDesc.res.pitch2D.pitchInBytes = descr.rowBytes(descr.pixelSize());

  TexturePtr tex;
  checkError(cuTexObjectCreate(&tex, &resDesc, &texDesc, nullptr));
  _textures.push_back(
      {addressMode, filterMode, normalizedCoords, std::move(tex)});
  return _textures.back().tex.handleAddress();
}

void ImageCUDA::markUsed(const StreamCUDA& s) {
  // Propagate to the sharedImage donor (if any) so the owning side's
  // deferred release waits for work enqueued through this view.
  if (_donorBuffer) static_cast<BufferCUDA*>(_donorBuffer.get())->markUsed(s);
  if (_donorImage) static_cast<ImageCUDA*>(_donorImage.get())->markUsed(s);
  CUstream handle = s.queue.get();
  for (const StreamUse& existing : _useStreams) {
    if (existing.stream == handle) return;
  }
  _useStreams.push_back({handle, std::weak_ptr<void>(s.aliveToken)});
}

ImageCUDA::ImageCUDA(const DeviceCUDA& dev, const ImageDescription& descr_)
    : descr(descr_) {
  _device = &dev;
  CUresult err;
  size_t pitch;
  descr = descr_;
  size_t bytes = descr.pixelSize();
  size_t elementSize = std::max((size_t)4, std::min(bytes, (size_t)16));
  err = cuMemAllocPitch(&mem, &pitch, descr.size.x * bytes,
                        descr.size.y * descr.size.z, elementSize);
  checkError(err);
  descr.stride = Stride2(static_cast<int32_t>(pitch),
                         static_cast<int32_t>(pitch * descr.size.y));
}

ImageCUDA::ImageCUDA(const DeviceCUDA& dev, const ImageDescription& descr_,
                     const std::shared_ptr<Buffer>& buffer)
    : descr(descr_), _donorBuffer(buffer) {
  _device = &dev;
  // Non-owning view; _donorBuffer keeps the allocation alive and its owner
  // remains responsible for the (use-stream-ordered) free.
  auto* b = static_cast<BufferCUDA*>(buffer.get());
  mem = cu::ptr<CUdeviceptr>(b->mem.get(), false);
}

ImageCUDA::ImageCUDA(const DeviceCUDA& dev, const ImageDescription& descr_,
                     const std::shared_ptr<Image>& image)
    : descr(descr_), _donorImage(image) {
  _device = &dev;
  auto* i = static_cast<ImageCUDA*>(image.get());
  mem = cu::ptr<CUdeviceptr>(i->mem.get(), false);
}

void ImageCUDA::copy(const ghost::Encoder& s, const ghost::Image& src) {
  auto src_impl = static_cast<implementation::ImageCUDA*>(src.impl().get());
  {
    auto sc = static_cast<implementation::StreamCUDA*>(s.impl().get());
    markUsed(*sc);
    src_impl->markUsed(*sc);
  }
  size_t srcPixSize = src_impl->descr.pixelSize();
  size_t srcRowBytes = src_impl->descr.rowBytes(srcPixSize);
  size_t srcSliceBytes = src_impl->descr.sliceBytes(srcRowBytes);
  size_t dstPixSize = descr.pixelSize();
  size_t dstRowBytes = descr.rowBytes(dstPixSize);
  size_t dstSliceBytes = descr.sliceBytes(dstRowBytes);
  if (descr.size.z > 1) {
    CUDA_MEMCPY3D a;
    a.srcXInBytes = 0;
    a.srcY = 0;
    a.srcZ = 0;
    a.srcLOD = 0;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = src_impl->mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.srcHeight = srcSliceBytes / srcRowBytes;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstZ = 0;
    a.dstLOD = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.dstHeight = dstSliceBytes / dstRowBytes;
    a.WidthInBytes = descr.size.x * dstPixSize;
    a.Height = descr.size.y;
    a.Depth = descr.size.z;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(*stream_impl);
    CUresult err;
    err = cuMemcpy3DAsync(&a, stream_impl->queue);
    checkError(err);
  } else {
    CUDA_MEMCPY2D a;
    a.srcXInBytes = 0;
    a.srcY = 0;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = src_impl->mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.WidthInBytes = descr.size.x * dstPixSize;
    a.Height = descr.size.y;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(*stream_impl);
    CUresult err;
    err = cuMemcpy2DAsync(&a, stream_impl->queue);
    checkError(err);
  }
}

void ImageCUDA::copy(const ghost::Encoder& s, const ghost::Buffer& src,
                     const BufferLayout& layout) {
  auto src_impl = static_cast<implementation::BufferCUDA*>(src.impl().get());
  {
    auto sc = static_cast<implementation::StreamCUDA*>(s.impl().get());
    markUsed(*sc);
    src_impl->markUsed(*sc);
  }
  size_t pixSize = descr.pixelSize();
  size_t srcRowBytes = layout.rowBytes(pixSize);
  size_t srcSliceBytes = layout.sliceBytes(srcRowBytes);
  size_t dstRowBytes = descr.rowBytes(pixSize);
  size_t dstSliceBytes = descr.sliceBytes(dstRowBytes);
  if (descr.size.z > 1) {
    CUDA_MEMCPY3D a;
    a.srcXInBytes = 0;
    a.srcY = 0;
    a.srcZ = 0;
    a.srcLOD = 0;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = src_impl->mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.srcHeight = srcSliceBytes / srcRowBytes;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstZ = 0;
    a.dstLOD = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.dstHeight = dstSliceBytes / dstRowBytes;
    a.WidthInBytes = descr.size.x * pixSize;
    a.Height = descr.size.y;
    a.Depth = descr.size.z;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(*stream_impl);
    CUresult err;
    err = cuMemcpy3DAsync(&a, stream_impl->queue);
    checkError(err);
  } else {
    CUDA_MEMCPY2D a;
    a.srcXInBytes = 0;
    a.srcY = 0;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = src_impl->mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.WidthInBytes = descr.size.x * pixSize;
    a.Height = descr.size.y;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(*stream_impl);
    CUresult err;
    err = cuMemcpy2DAsync(&a, stream_impl->queue);
    checkError(err);
  }
}

void ImageCUDA::copy(const ghost::Encoder& s, const void* src,
                     const BufferLayout& layout) {
  size_t pixSize = descr.pixelSize();
  size_t srcRowBytes = layout.rowBytes(pixSize);
  size_t srcSliceBytes = layout.sliceBytes(srcRowBytes);
  size_t dstRowBytes = descr.rowBytes(pixSize);
  size_t dstSliceBytes = descr.sliceBytes(dstRowBytes);
  if (descr.size.z > 1) {
    CUDA_MEMCPY3D a;
    a.srcXInBytes = 0;
    a.srcY = 0;
    a.srcZ = 0;
    a.srcLOD = 0;
    a.srcMemoryType = CU_MEMORYTYPE_HOST;
    a.srcHost = src;
    a.srcDevice = (CUdeviceptr)0;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.srcHeight = srcSliceBytes / srcRowBytes;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstZ = 0;
    a.dstLOD = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.dstHeight = dstSliceBytes / dstRowBytes;
    a.WidthInBytes = descr.size.x * pixSize;
    a.Height = descr.size.y;
    a.Depth = descr.size.z;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(*stream_impl);
    CUresult err;
    err = cuMemcpy3DAsync(&a, stream_impl->queue);
    checkError(err);
  } else {
    CUDA_MEMCPY2D a;
    a.srcXInBytes = 0;
    a.srcY = 0;
    a.srcMemoryType = CU_MEMORYTYPE_HOST;
    a.srcHost = src;
    a.srcDevice = (CUdeviceptr)0;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.WidthInBytes = descr.size.x * pixSize;
    a.Height = descr.size.y;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(*stream_impl);
    CUresult err;
    err = cuMemcpy2DAsync(&a, stream_impl->queue);
    checkError(err);
  }
}

void ImageCUDA::copy(const ghost::Encoder& s, HostBytes src,
                     const BufferLayout& layout) {
  this->copy(s, src.data(), layout);
  if (src.ownsBytes()) {
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    stream_impl->retainHostUntilDone(src.owner());
  }
}

void ImageCUDA::copyTo(const ghost::Encoder& s, ghost::Buffer& dst,
                       const BufferLayout& layout) const {
  auto dst_impl = static_cast<implementation::BufferCUDA*>(dst.impl().get());
  {
    auto sc = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(*sc);
    dst_impl->markUsed(*sc);
  }
  size_t pixSize = descr.pixelSize();
  size_t srcRowBytes = descr.rowBytes(pixSize);
  size_t srcSliceBytes = descr.sliceBytes(srcRowBytes);
  size_t dstRowBytes = layout.rowBytes(pixSize);
  size_t dstSliceBytes = layout.sliceBytes(dstRowBytes);
  if (descr.size.z > 1) {
    CUDA_MEMCPY3D a;
    a.srcXInBytes = 0;
    a.srcY = 0;
    a.srcZ = 0;
    a.srcLOD = 0;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.srcHeight = srcSliceBytes / srcRowBytes;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstZ = 0;
    a.dstLOD = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = dst_impl->mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.dstHeight = dstSliceBytes / dstRowBytes;
    a.WidthInBytes = layout.size.x * pixSize;
    a.Height = layout.size.y;
    a.Depth = layout.size.z;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(*stream_impl);
    CUresult err;
    err = cuMemcpy3DAsync(&a, stream_impl->queue);
    checkError(err);
  } else {
    CUDA_MEMCPY2D a;
    a.srcXInBytes = 0;
    a.srcY = 0;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = dst_impl->mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.WidthInBytes = layout.size.x * pixSize;
    a.Height = layout.size.y;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(*stream_impl);
    CUresult err;
    err = cuMemcpy2DAsync(&a, stream_impl->queue);
    checkError(err);
  }
}

void ImageCUDA::copyTo(const ghost::Encoder& s, void* dst,
                       const BufferLayout& layout) const {
  size_t pixSize = descr.pixelSize();
  size_t srcRowBytes = descr.rowBytes(pixSize);
  size_t srcSliceBytes = descr.sliceBytes(srcRowBytes);
  size_t dstRowBytes = layout.rowBytes(pixSize);
  size_t dstSliceBytes = layout.sliceBytes(dstRowBytes);
  if (descr.size.z > 1) {
    CUDA_MEMCPY3D a;
    a.srcXInBytes = 0;
    a.srcY = 0;
    a.srcZ = 0;
    a.srcLOD = 0;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.srcHeight = srcSliceBytes / srcRowBytes;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstZ = 0;
    a.dstLOD = 0;
    a.dstMemoryType = CU_MEMORYTYPE_HOST;
    a.dstHost = dst;
    a.dstDevice = (CUdeviceptr)0;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.dstHeight = dstSliceBytes / dstRowBytes;
    a.WidthInBytes = layout.size.x * pixSize;
    a.Height = layout.size.y;
    a.Depth = layout.size.z;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(*stream_impl);
    CUresult err;
    err = cuMemcpy3DAsync(&a, stream_impl->queue);
    checkError(err);
  } else {
    CUDA_MEMCPY2D a;
    a.srcXInBytes = 0;
    a.srcY = 0;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstMemoryType = CU_MEMORYTYPE_HOST;
    a.dstHost = dst;
    a.dstDevice = (CUdeviceptr)0;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.WidthInBytes = layout.size.x * pixSize;
    a.Height = layout.size.y;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(*stream_impl);
    CUresult err;
    err = cuMemcpy2DAsync(&a, stream_impl->queue);
    checkError(err);
  }
}

void ImageCUDA::copyTo(const ghost::Encoder& s, HostBytes dst,
                       const BufferLayout& layout) const {
  this->copyTo(s, dst.data(), layout);
  if (dst.ownsBytes()) {
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    stream_impl->retainHostUntilDone(dst.owner());
  }
}

void ImageCUDA::copy(const ghost::Encoder& s, const ghost::Buffer& src,
                     const BufferLayout& layout, const Origin3& imageOrigin) {
  auto src_impl = static_cast<implementation::BufferCUDA*>(src.impl().get());
  {
    auto sc = static_cast<implementation::StreamCUDA*>(s.impl().get());
    markUsed(*sc);
    src_impl->markUsed(*sc);
  }
  size_t pixSize = descr.pixelSize();
  size_t srcRowBytes = layout.rowBytes(pixSize);
  size_t srcSliceBytes = layout.sliceBytes(srcRowBytes);
  size_t dstRowBytes = descr.rowBytes(pixSize);
  size_t dstSliceBytes = descr.sliceBytes(dstRowBytes);
  if (descr.size.z > 1) {
    CUDA_MEMCPY3D a;
    a.srcXInBytes = 0;
    a.srcY = 0;
    a.srcZ = 0;
    a.srcLOD = 0;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = src_impl->mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.srcHeight = srcSliceBytes / srcRowBytes;
    a.dstXInBytes = imageOrigin.x * pixSize;
    a.dstY = imageOrigin.y;
    a.dstZ = imageOrigin.z;
    a.dstLOD = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.dstHeight = dstSliceBytes / dstRowBytes;
    a.WidthInBytes = layout.size.x * pixSize;
    a.Height = layout.size.y;
    a.Depth = layout.size.z;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(*stream_impl);
    CUresult err;
    err = cuMemcpy3DAsync(&a, stream_impl->queue);
    checkError(err);
  } else {
    CUDA_MEMCPY2D a;
    a.srcXInBytes = 0;
    a.srcY = 0;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = src_impl->mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.dstXInBytes = imageOrigin.x * pixSize;
    a.dstY = imageOrigin.y;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.WidthInBytes = layout.size.x * pixSize;
    a.Height = layout.size.y;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(*stream_impl);
    CUresult err;
    err = cuMemcpy2DAsync(&a, stream_impl->queue);
    checkError(err);
  }
}

void ImageCUDA::copyTo(const ghost::Encoder& s, ghost::Buffer& dst,
                       const BufferLayout& layout,
                       const Origin3& imageOrigin) const {
  auto dst_impl = static_cast<implementation::BufferCUDA*>(dst.impl().get());
  {
    auto sc = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(*sc);
    dst_impl->markUsed(*sc);
  }
  size_t pixSize = descr.pixelSize();
  size_t srcRowBytes = descr.rowBytes(pixSize);
  size_t srcSliceBytes = descr.sliceBytes(srcRowBytes);
  size_t dstRowBytes = layout.rowBytes(pixSize);
  size_t dstSliceBytes = layout.sliceBytes(dstRowBytes);
  if (descr.size.z > 1) {
    CUDA_MEMCPY3D a;
    a.srcXInBytes = imageOrigin.x * pixSize;
    a.srcY = imageOrigin.y;
    a.srcZ = imageOrigin.z;
    a.srcLOD = 0;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.srcHeight = srcSliceBytes / srcRowBytes;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstZ = 0;
    a.dstLOD = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = dst_impl->mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.dstHeight = dstSliceBytes / dstRowBytes;
    a.WidthInBytes = layout.size.x * pixSize;
    a.Height = layout.size.y;
    a.Depth = layout.size.z;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(*stream_impl);
    CUresult err;
    err = cuMemcpy3DAsync(&a, stream_impl->queue);
    checkError(err);
  } else {
    CUDA_MEMCPY2D a;
    a.srcXInBytes = imageOrigin.x * pixSize;
    a.srcY = imageOrigin.y;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.dstXInBytes = 0;
    a.dstY = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = dst_impl->mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.WidthInBytes = layout.size.x * pixSize;
    a.Height = layout.size.y;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(*stream_impl);
    CUresult err;
    err = cuMemcpy2DAsync(&a, stream_impl->queue);
    checkError(err);
  }
}

void ImageCUDA::copy(const ghost::Encoder& s, const ghost::Image& src,
                     const Size3& region, const Origin3& srcOrigin,
                     const Origin3& dstOrigin) {
  auto src_impl = static_cast<implementation::ImageCUDA*>(src.impl().get());
  {
    auto sc = static_cast<implementation::StreamCUDA*>(s.impl().get());
    markUsed(*sc);
    src_impl->markUsed(*sc);
  }
  size_t srcPixSize = src_impl->descr.pixelSize();
  size_t srcRowBytes = src_impl->descr.rowBytes(srcPixSize);
  size_t srcSliceBytes = src_impl->descr.sliceBytes(srcRowBytes);
  size_t dstPixSize = descr.pixelSize();
  size_t dstRowBytes = descr.rowBytes(dstPixSize);
  size_t dstSliceBytes = descr.sliceBytes(dstRowBytes);
  if (region.z > 1) {
    CUDA_MEMCPY3D a;
    a.srcXInBytes = srcOrigin.x * srcPixSize;
    a.srcY = srcOrigin.y;
    a.srcZ = srcOrigin.z;
    a.srcLOD = 0;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = src_impl->mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.srcHeight = srcSliceBytes / srcRowBytes;
    a.dstXInBytes = dstOrigin.x * dstPixSize;
    a.dstY = dstOrigin.y;
    a.dstZ = dstOrigin.z;
    a.dstLOD = 0;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.dstHeight = dstSliceBytes / dstRowBytes;
    a.WidthInBytes = region.x * dstPixSize;
    a.Height = region.y;
    a.Depth = region.z;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(*stream_impl);
    CUresult err;
    err = cuMemcpy3DAsync(&a, stream_impl->queue);
    checkError(err);
  } else {
    CUDA_MEMCPY2D a;
    a.srcXInBytes = srcOrigin.x * srcPixSize;
    a.srcY = srcOrigin.y;
    a.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    a.srcHost = nullptr;
    a.srcDevice = src_impl->mem;
    a.srcArray = nullptr;
    a.srcPitch = srcRowBytes;
    a.dstXInBytes = dstOrigin.x * dstPixSize;
    a.dstY = dstOrigin.y;
    a.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    a.dstHost = nullptr;
    a.dstDevice = mem;
    a.dstArray = nullptr;
    a.dstPitch = dstRowBytes;
    a.WidthInBytes = region.x * dstPixSize;
    a.Height = region.y;
    auto stream_impl = static_cast<implementation::StreamCUDA*>(s.impl().get());
    const_cast<ImageCUDA*>(this)->markUsed(*stream_impl);
    CUresult err;
    err = cuMemcpy2DAsync(&a, stream_impl->queue);
    checkError(err);
  }
}

thread_local size_t CU_CurrentContext::_pushCount;

CUcontext CU_CurrentContext::get() {
  CUcontext ctx;
  if (cuCtxGetCurrent(&ctx) == CUDA_SUCCESS) return ctx;
  return nullptr;
}

CUresult CU_CurrentContext::set(CUcontext c) {
  CUresult err;
  CUcontext ctx;
  err = cuCtxGetCurrent(&ctx);
  if (err != CUDA_SUCCESS) {
    return err;
  }
  if (c == ctx) {
    return err;
  }
  if (ctx != nullptr) {
    (void)cuCtxSynchronize();
  }
  size_t count = _pushCount;
  if (count > 0) {
    err = cuCtxSetCurrent(c);
  } else {
    err = cuCtxPushCurrent(c);
    if (err == CUDA_SUCCESS) {
      count++;
      _pushCount = count;
    }
  }
  return err;
}

void CU_CurrentContext::pushed() { _pushCount++; }

void CU_CurrentContext::pop() {
  CUresult err;
  CUcontext ctx;
  size_t count = _pushCount;
  if (count > 0) _pushCount = 0;
  while (count > 0) {
    err = cuCtxGetCurrent(&ctx);
    if (err == CUDA_SUCCESS && ctx != nullptr) {
      (void)cuCtxSynchronize();
    }
    cuCtxSynchronize();
    cuCtxPopCurrent(&ctx);
    count--;
  }
}

DeviceCUDA::DeviceCUDA(const SharedContext& share) {
  if (!isCudaDriverAvailable()) {
    checkError(CUDA_ERROR_NOT_INITIALIZED);
  }
  CUresult err = cuInit(0);
  checkError(err);
  context =
      cu::ptr<CUcontext>(reinterpret_cast<CUcontext>(share.device), false);
  queue = cu::ptr<CUstream>(reinterpret_cast<CUstream>(share.queue), false);
  if (!context) {
    CU_CurrentContext::pop();  // clear current stack
    device = (CUdevice)0;
#if CUDA_VERSION >= 13000
    CUctxCreateParams ctxCreateParams = {};
    err = cuCtxCreate(&context, &ctxCreateParams, 0, device);
#else
    err = cuCtxCreate(&context, 0, device);
#endif
    checkError(err);
    CU_CurrentContext::pushed();
  } else {
    err = cuCtxGetDevice(&device);
    checkError(err);
  }
  if (!queue) {
    err = cuStreamCreate(&queue, CU_STREAM_NON_BLOCKING);
    checkError(err);
  }
  checkError(cuDeviceGetAttribute(&computeCapability.major,
                                  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                  device));
  checkError(cuDeviceGetAttribute(&computeCapability.minor,
                                  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                  device));
}

DeviceCUDA::DeviceCUDA(const GpuInfo& info) : DeviceCUDA(info.index) {}

DeviceCUDA::DeviceCUDA(int deviceOrdinal) {
  if (!isCudaDriverAvailable()) {
    checkError(CUDA_ERROR_NOT_INITIALIZED);
  }
  CUresult err;
  err = cuInit(0);
  checkError(err);
  err = cuDeviceGet(&device, deviceOrdinal);
  checkError(err);
  CU_CurrentContext::pop();  // clear current stack
#if CUDA_VERSION >= 13000
  CUctxCreateParams ctxCreateParams = {};
  err = cuCtxCreate(&context, &ctxCreateParams, 0, device);
#else
  err = cuCtxCreate(&context, 0, device);
#endif
  checkError(err);
  CU_CurrentContext::pushed();
  err = cuStreamCreate(&queue, CU_STREAM_NON_BLOCKING);
  checkError(err);
  checkError(cuDeviceGetAttribute(&computeCapability.major,
                                  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                  device));
  checkError(cuDeviceGetAttribute(&computeCapability.minor,
                                  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                  device));
}

DeviceCUDA::~DeviceCUDA() {
  try {
    // Destroy any parked texture objects before the context goes away.
    reapDeferredTextures(/*waitAll=*/true);
    // Need to clear context before we can destroy it.
    if (context.get() && CU_CurrentContext::get() == context.get())
      CU_CurrentContext::pop();
  } catch (...) {
  }
}

void DeviceCUDA::deferTextureRelease(
    std::vector<TexturePtr>&& textures,
    const std::vector<CUstream>& streams) const {
  if (textures.empty()) return;
  // Reap earlier entries first so the list stays bounded even if the host
  // never hits another reap point.
  reapDeferredTextures();
  PendingTextureRelease entry;
  entry.textures = std::move(textures);
  entry.events.reserve(streams.size());
  for (CUstream s : streams) {
    cu::ptr<CUevent> ev;
    CUresult err = cuEventCreate(&ev, CU_EVENT_DISABLE_TIMING);
    if (err == CUDA_SUCCESS) err = cuEventRecord(ev, s);
    if (err != CUDA_SUCCESS) {
      // Can't guard with events — drain the streams and destroy inline
      // (entry.textures release on return).
      for (CUstream drain : streams) cuStreamSynchronize(drain);
      return;
    }
    entry.events.push_back(std::move(ev));
  }
  std::lock_guard<std::mutex> lock(_texReapMutex);
  _texReap.push_back(std::move(entry));
}

void DeviceCUDA::reapDeferredTextures(bool waitAll) const {
  std::vector<PendingTextureRelease> done;  // destroyed outside the lock
  {
    std::lock_guard<std::mutex> lock(_texReapMutex);
    auto it = _texReap.begin();
    while (it != _texReap.end()) {
      bool complete = true;
      for (auto& ev : it->events) {
        if (waitAll) {
          cuEventSynchronize(ev);
        } else if (cuEventQuery(ev) != CUDA_SUCCESS) {
          complete = false;
          break;
        }
      }
      if (complete) {
        done.push_back(std::move(*it));
        it = _texReap.erase(it);
      } else {
        ++it;
      }
    }
  }
}

void DeviceCUDA::activate(void** prevOut) {
  CUcontext prev = CU_CurrentContext::get();
  if (prevOut) *prevOut = reinterpret_cast<void*>(prev);
  checkError(CU_CurrentContext::set(context.get()));
}

void DeviceCUDA::deactivate(void* prev) {
  // Surface any prior async error to the caller instead of losing it.
  CUresult syncErr = cuCtxSynchronize();
  CUcontext prevCtx = reinterpret_cast<CUcontext>(prev);
  if (prevCtx != nullptr && prevCtx != context.get()) {
    (void)CU_CurrentContext::set(prevCtx);
  }
  if (syncErr != CUDA_SUCCESS) checkError(syncErr);
}

ghost::Library DeviceCUDA::loadLibraryFromText(const std::string& text,
                                               const CompilerOptions& options,
                                               bool retainBinary) const {
  auto ptr = std::make_shared<implementation::LibraryCUDA>(*this, retainBinary);
  ptr->loadFromText(text, options);
  return ghost::Library(ptr);
}

ghost::Library DeviceCUDA::loadLibraryFromData(const void* data, size_t len,
                                               const CompilerOptions& options,
                                               bool retainBinary) const {
  auto ptr = std::make_shared<implementation::LibraryCUDA>(*this, retainBinary);
  ptr->loadFromData(data, len, options);
  return ghost::Library(ptr);
}

SharedContext DeviceCUDA::shareContext() const {
  SharedContext c(context.get(), queue.get());
  return c;
}

ghost::Stream DeviceCUDA::createStream(const StreamOptions& options) const {
  auto ptr = std::make_shared<implementation::StreamCUDA>(context.get());
  return ghost::Stream(ptr);
}

std::shared_ptr<CommandBuffer> DeviceCUDA::createCommandBuffer(
    const CommandBufferOptions&) const {
  // CUDA has no native command-buffer concept aligned with Ghost's recording
  // cb (CUDA graphs are a separate model that most CUDA libraries don't use).
  // The subclass exists to expose encodeNative on top of the default
  // record-and-replay machinery.
  return std::make_shared<CommandBufferCUDA>();
}

size_t DeviceCUDA::getMemoryPoolSize() const {
  return Device::getMemoryPoolSize();
}

void DeviceCUDA::setMemoryPoolSize(size_t bytes) {
  Device::setMemoryPoolSize(bytes);
#if CUDA_VERSION >= 11020
  // Drop our reference only — outstanding pool-backed buffers each share
  // ownership, so the old pool stays alive until their stream-ordered frees
  // have been enqueued (cuMemPoolDestroy itself defers reclamation until all
  // queued frees complete).
  memPool.reset();
  if (bytes > 0) {
    CUmemPoolProps poolProps = {};
    poolProps.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    poolProps.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
    poolProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    poolProps.location.id = device;
    CUresult err = cuMemPoolCreate(&memPool, &poolProps);
    if (err == CUDA_SUCCESS) {
      cuuint64_t maxBytes = static_cast<cuuint64_t>(bytes);
      cuMemPoolSetAttribute(memPool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                            &maxBytes);
    } else {
      memPool.reset();
    }
  }
#endif
}

ghost::Buffer DeviceCUDA::allocateBuffer(size_t bytes,
                                         const BufferOptions& opts) const {
  // AllocHint::Staging routes to pinned host memory (mapped buffer path).
  if (opts.hint == AllocHint::Staging) {
    auto ptr =
        std::make_shared<implementation::MappedBufferCUDA>(*this, bytes, opts);
    return ghost::Buffer(ptr);
  }
  // A host-installed allocator overrides every Ghost-internal path including
  // the memory pool; it may decline (return nullptr) to fall through.
  if (auto* a = allocator()) {
    if (void* handle = a->allocateBuffer(bytes, opts)) {
      CUdeviceptr devPtr =
          static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(handle));
      auto ptr = std::make_shared<implementation::BufferCUDA>(
          cu::ptr<CUdeviceptr>(devPtr, /*retainObject=*/true), bytes);
      ptr->setAllocator(a);
      return ghost::Buffer(ptr);
    }
  }
#if CUDA_VERSION >= 11020
  // Use the memory pool for Default/Transient allocations. Persistent
  // allocations bypass the pool to avoid fragmenting long-lived resources.
  if (memPool && opts.hint != AllocHint::Persistent) {
    CUdeviceptr devPtr;
    CUresult err = cuMemAllocFromPoolAsync(&devPtr, bytes, memPool, queue);
    if (err == CUDA_SUCCESS) {
      // The allocation is stream-ordered on `queue`: record its completion
      // so other streams can take a device-side dependency at first use
      // (markUsed) instead of a host sync here.
      cu::ptr<CUevent> ready;
      CUresult evErr = cuEventCreate(&ready, CU_EVENT_DISABLE_TIMING);
      if (evErr == CUDA_SUCCESS) evErr = cuEventRecord(ready, queue);
      if (evErr != CUDA_SUCCESS) {
        ready.reset();
        cuStreamSynchronize(queue);  // degraded: order via host sync
      }
      // The buffer owns the pointer; its destructor returns the memory to
      // the pool with a stream-ordered free, and its pool reference keeps
      // the pool alive until then.
      auto ptr = std::make_shared<implementation::BufferCUDA>(
          cu::ptr<CUdeviceptr>(devPtr, /*retainObject=*/true), bytes);
      ptr->setPoolBacked(memPool, queue, std::move(ready));
      return ghost::Buffer(ptr);
    }
    // Pool allocation failed — fall through to standard allocation.
  }
#endif
  auto ptr = std::make_shared<implementation::BufferCUDA>(*this, bytes, opts);
  return ghost::Buffer(ptr);
}

ghost::MappedBuffer DeviceCUDA::allocateMappedBuffer(
    size_t bytes, const BufferOptions& opts) const {
  if (auto* a = allocator()) {
    if (void* handle = a->allocateMappedBuffer(bytes, opts)) {
      // For CUDA mapped buffers the host returns the host pointer; Ghost
      // derives the device pointer via cuMemHostGetDevicePointer.
      auto ptr = std::make_shared<implementation::MappedBufferCUDA>(
          cu::ptr<void*>(handle, /*retainObject=*/true));
      ptr->_size = bytes;
      ptr->setAllocator(a);
      return ghost::MappedBuffer(ptr);
    }
  }
  auto ptr =
      std::make_shared<implementation::MappedBufferCUDA>(*this, bytes, opts);
  return ghost::MappedBuffer(ptr);
}

ghost::Image DeviceCUDA::allocateImage(const ImageDescription& descr) const {
  reapDeferredTextures();
  if (auto* a = allocator()) {
    if (void* handle = a->allocateImage(descr)) {
      CUdeviceptr devPtr =
          static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(handle));
      auto ptr = std::make_shared<implementation::ImageCUDA>(
          cu::ptr<CUdeviceptr>(devPtr, /*retainObject=*/true), descr, this);
      ptr->setAllocator(a);
      return ghost::Image(ptr);
    }
  }
  auto ptr = std::make_shared<implementation::ImageCUDA>(*this, descr);
  return ghost::Image(ptr);
}

ghost::Image DeviceCUDA::sharedImage(const ImageDescription& descr,
                                     ghost::Buffer& buffer) const {
  auto ptr =
      std::make_shared<implementation::ImageCUDA>(*this, descr, buffer.impl());
  return ghost::Image(ptr);
}

ghost::Image DeviceCUDA::sharedImage(const ImageDescription& descr,
                                     ghost::Image& image) const {
  auto ptr =
      std::make_shared<implementation::ImageCUDA>(*this, descr, image.impl());
  return ghost::Image(ptr);
}

ghost::Buffer DeviceCUDA::wrapBuffer(const SharedBuffer& shared) const {
  CUdeviceptr devPtr =
      static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(shared.handle));
  // owned=false: cu::ptr stores the value but never calls cuMemFree.
  auto ptr = std::make_shared<implementation::BufferCUDA>(
      cu::ptr<CUdeviceptr>(devPtr, /*retainObject=*/false), shared.bytes);
  return ghost::Buffer(ptr);
}

ghost::Image DeviceCUDA::wrapImage(const SharedImage& shared) const {
  CUdeviceptr devPtr =
      static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(shared.handle));
  auto ptr = std::make_shared<implementation::ImageCUDA>(
      cu::ptr<CUdeviceptr>(devPtr, /*retainObject=*/false), shared.descr, this);
  return ghost::Image(ptr);
}

Attribute DeviceCUDA::getAttribute(DeviceAttributeId what) const {
  switch (what) {
    case kDeviceImplementation:
      return "CUDA";
    case kDeviceName: {
      char buf[128];
      checkError(cuDeviceGetName(buf, sizeof(buf), device));
      return buf;
    }
    case kDeviceVendor:
      return "NVIDIA";
    case kDeviceDriverVersion: {
      int version = 0;
      checkError(cuDriverGetVersion(&version));
      std::stringstream stream;
      stream << version;
      return stream.str();
    }
    case kDeviceFamily: {
      std::stringstream stream;
      stream << computeCapability.major << "." << computeCapability.minor;
      return stream.str();
    }
    case kDeviceCount:
      return 1;
    case kDeviceProcessorCount: {
      int multiProcessorCount;
      checkError(cuDeviceGetAttribute(&multiProcessorCount,
                                      CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                      device));
      return multiProcessorCount;
    }
    case kDeviceUnifiedMemory: {
      int v;
      checkError(
          cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_INTEGRATED, device));
      return v != 0;
    }
    case kDeviceMemory: {
      size_t v;
      checkError(cuDeviceTotalMem(&v, device));
      return v;
    }
    case kDeviceLocalMemory: {
      int v;
      checkError(cuDeviceGetAttribute(
          &v, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device));
      return v;
    }
    case kDeviceMaxThreads: {
      int v;
      checkError(cuDeviceGetAttribute(
          &v, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
      return v;
    }
    case kDeviceMaxWorkSize: {
      int x, y, z;
      checkError(cuDeviceGetAttribute(&x, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                                      device));
      checkError(cuDeviceGetAttribute(&y, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
                                      device));
      checkError(cuDeviceGetAttribute(&z, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
                                      device));
      return Attribute(x, y, z);
    }
    case kDeviceMaxRegisters: {
      int v;
      checkError(cuDeviceGetAttribute(
          &v, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device));
      return v;
    }
    case kDeviceMaxImageSize1: {
      int x, y, z;
      checkError(cuDeviceGetAttribute(
          &x, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, device));
      return x;
    }
    case kDeviceMaxImageSize2: {
      int x, y;
      checkError(cuDeviceGetAttribute(
          &x, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, device));
      checkError(cuDeviceGetAttribute(
          &y, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, device));
      return Attribute(x, y);
    }
    case kDeviceMaxImageSize3: {
      int x, y, z;
      checkError(cuDeviceGetAttribute(
          &x, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, device));
      checkError(cuDeviceGetAttribute(
          &y, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, device));
      checkError(cuDeviceGetAttribute(
          &z, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, device));
      return Attribute(x, y, z);
    }
    case kDeviceMaxImageAlignment: {
      int v;
      checkError(cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
                                      device));
      return v;
    }
    case kDeviceSupportsImageIntegerFiltering:
      return true;
    case kDeviceSupportsImageFloatFiltering:
      return true;
    case kDeviceSupportsMappedBuffer: {
      int canMap;
      checkError(cuDeviceGetAttribute(
          &canMap, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, device));
      return canMap != 0;
    }
    case kDeviceSupportsProgramConstants:
      return false;
    case kDeviceSupportsProgramGlobals:
      return true;
    case kDeviceSupportsSubgroup:
      return true;
    case kDeviceSupportsSubgroupShuffle:
      return true;
    case kDeviceSubgroupWidth: {
      int v;
      checkError(
          cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device));
      return v;
    }
    case kDeviceMaxComputeUnits: {
      int v;
      checkError(cuDeviceGetAttribute(
          &v, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
      return v;
    }
    case kDeviceMemoryAlignment: {
      int v;
      checkError(cuDeviceGetAttribute(&v, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
                                      device));
      return v;
    }
    case kDeviceBufferAlignment:
      return 256;
    case kDeviceMaxBufferSize: {
      size_t v;
      checkError(cuDeviceTotalMem(&v, device));
      return (uint64_t)v;
    }
    case kDeviceMaxConstantBufferSize: {
      int v;
      checkError(cuDeviceGetAttribute(
          &v, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, device));
      return v;
    }
    case kDeviceTimestampPeriod:
      return 1000.0f;
    case kDeviceSupportsProfilingTimer:
      return true;
    case kDeviceSupportsCooperativeMatrix:
      // WMMA requires compute capability >= 7.0 (Volta+)
      return computeCapability.major >= 7;
    default:
      return Attribute();
  }
}
}  // namespace implementation

DeviceCUDA::DeviceCUDA(const SharedContext& share)
    : Device(std::make_shared<implementation::DeviceCUDA>(share)) {
  auto cuda = static_cast<implementation::DeviceCUDA*>(impl().get());
  // Non-owning view: passing cuda->queue by value would silently move
  // ownership of the device's queue into the default stream (cu::ptr's
  // lvalue copy steals), leaving teardown correct only by member-order luck.
  setDefaultStream(std::make_shared<implementation::StreamCUDA>(
      cu::ptr<CUstream>(cuda->queue.get(), false)));
}

DeviceCUDA::DeviceCUDA(const GpuInfo& info)
    : Device(std::make_shared<implementation::DeviceCUDA>(info)) {
  auto cuda = static_cast<implementation::DeviceCUDA*>(impl().get());
  setDefaultStream(std::make_shared<implementation::StreamCUDA>(
      cu::ptr<CUstream>(cuda->queue.get(), false)));
}

std::vector<GpuInfo> DeviceCUDA::enumerateDevices() {
  std::vector<GpuInfo> result;
  if (!isCudaDriverAvailable()) return result;
  CUresult err = cuInit(0);
  if (err != CUDA_SUCCESS) return result;

  int count = 0;
  err = cuDeviceGetCount(&count);
  if (err != CUDA_SUCCESS) return result;

  for (int i = 0; i < count; i++) {
    CUdevice dev;
    err = cuDeviceGet(&dev, i);
    if (err != CUDA_SUCCESS) continue;

    GpuInfo info;

    char name[256];
    if (cuDeviceGetName(name, sizeof(name), dev) == CUDA_SUCCESS)
      info.name = name;

    info.vendor = "NVIDIA";
    info.implementation = "CUDA";

    size_t totalMem;
    if (cuDeviceTotalMem(&totalMem, dev) == CUDA_SUCCESS)
      info.memory = totalMem;

    int unified = 0;
    if (cuDeviceGetAttribute(&unified, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
                             dev) == CUDA_SUCCESS)
      info.unifiedMemory = unified != 0;

    info.index = i;
    result.push_back(info);
  }
  return result;
}
}  // namespace ghost
#endif
