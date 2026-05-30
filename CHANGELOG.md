# Changelog

## 1.1

### Added

- `StreamOptions::concurrent` (default `false`) — when `true`, the backend
  skips its per-dispatch barrier (Vulkan `vkCmdPipelineBarrier`, DirectX
  global UAV barrier, Metal serial dispatch type) so successive
  independent dispatches may execute concurrently. Caller takes
  responsibility for inter-dispatch hazards.
- `CommandBufferOptions::concurrent` (default `true`) — when `false`, the
  backend restores the per-dispatch auto-barrier on Vulkan, DirectX, and
  Metal. Useful when porting `Stream` code to `CommandBuffer` without
  retrofitting explicit `cb.barrier()` calls.
- `CommandBuffer(const Device&, const CommandBufferOptions&)` constructor
  overload.
- Native `CommandBufferMetal` backed by `MTLCommandBuffer`. Replaces the
  variant-recording fallback for the Metal backend.
- `StreamMetal` now batches operations into a transient `MTLCommandBuffer`
  between sync points instead of one cb per op.
- `CommandBuffer::onCompletion(std::function<void()>)` — register a
  host-side callback that runs after the cb's recorded work has completed
  on the GPU. Fires after the most recent (or next) `submit`; handlers
  run in registration order. Backed by `MTLCommandBuffer
  addCompletedHandler:` on Metal, fence-observation in
  `waitForCompletion` on Vulkan/DirectX, and `stream.sync()` drain in the
  fallback. Typical uses: returning host buffers to a pool, signalling a
  future, kicking the next CPU stage without blocking on `stream.sync`.
- `Buffer::copyTo(const CommandBuffer&, void* dst, ...)` now works for
  Metal Private and Managed storage (previously threw `unsupported_error`
  during `submit`). The host memcpy is deferred to the cb's completion
  handler, so `stream.sync()` after `cb.submit(stream)` observes the
  bytes in `dst`. The host `dst` pointer must remain valid until that
  `stream.sync()` returns. Enables batching N device-to-host readbacks
  into a single `MTLCommandBuffer` + blit encoder + commit + wait.
- Resource write-intent for CommandBuffer barrier narrowing. `ghost::write(buf)`,
  `ghost::read(buf)`, `ghost::readwrite(buf)` tag a Buffer/Image kernel argument
  (mirroring `ghost::sampler()`); `BoundFunction::writes(n)` is shorthand for
  "the first `n` Buffer/Image args are written, the rest read-only". A
  CommandBuffer barrier then orders only resources some dispatch writes —
  resources that are read-only across the whole batch (e.g. weights) are
  dropped, preserving more dispatch overlap. Write-after-read is still ordered
  (a resource read here but written by a later dispatch stays in the set).
- `Library::setWriteDefault(WriteDefault)` with `WriteDefault { Conservative,
  FirstWritten }` — per-library default for resource args left untagged.
  Default `Conservative` treats every resource as written (no behavior change).
  `FirstWritten` opts into the "first Buffer/Image arg is the output" convention
  so barriers narrow without per-call tagging; kernels that write a non-first or
  multiple resources under `FirstWritten` must tag them (`ghost::write` /
  `writes()`) or they are under-synchronized in concurrent mode.

- `HostBytes` (`ghost/host_bytes.h`) — direction-neutral host-memory handle
  used as the source for uploads and the destination for readbacks. Factories:
  `borrow(ptr)` (caller-managed lifetime, same contract as the existing
  `void*` overloads), `adopt(ptr, deleter)` (Ghost takes ownership; the
  deleter runs after the queued DMA has completed, *and* after the
  recording cb has been reset/destroyed), `wrap(shared_ptr<void>, ptr=null)`
  (share ownership of an arena). Ownership is type-erased through
  `std::shared_ptr<void>`'s deleter slot, so any allocator works
  (`malloc` / `new[]`, `cuMemAllocHost`, `mmap`, foreign-library buffers).

- `Buffer::copy(Encoder, HostBytes, dstOffset, bytes)` /
  `Buffer::copyTo(Encoder, HostBytes, srcOffset, bytes)` and the image
  analogs (`Image::copy(Encoder, HostBytes, layout)` /
  `Image::copyTo(Encoder, HostBytes, layout)`). On `Stream` they retain
  ownership of the bytes via a per-stream pending list keyed by an event,
  releasing when the event reports complete (drained at `stream.sync()` and
  opportunistically polled at the next enqueue); on `CommandBuffer` they
  carry ownership through the recorded command. Lets callers fire async
  uploads/readbacks without tracking host lifetime themselves.

- `CommandBuffer` recorder overloads taking `HostBytes`:
  `copyBufferRaw(Buffer, HostBytes, …)`, `readBuffer(Buffer, HostBytes, …)`,
  `copyImageFromHost(Image, HostBytes, …)`, `copyImageToHost(Image,
  HostBytes, …)`. Skip the implicit snapshot-into-`std::vector` that the
  `void*` overloads do at record time, and let `cb.reset()` happen
  immediately after `submit()` without freeing memory the GPU is still
  reading or writing.

- CUDA pinned-memory uploads/readbacks now have a first-class lifetime
  contract via `HostBytes::adopt`. Previously, passing a pinned host
  pointer to `Buffer::copy(stream, void*, …)` was a silent footgun:
  `cuMemcpyHtoDAsync`/`DtoHAsync` from page-locked memory truly are async
  on the host pointer, and the caller had to keep it alive until
  `stream.sync()`. The new `HostBytes::adopt(pinned, cuMemFreeHost)`
  hands lifetime management to Ghost — the deleter fires after the
  matching `cuEvent` reports complete.

- `AllocHint::Shared` on Metal now allocates the buffer with hazard
  tracking enabled. Required for Metal to automatically synchronize
  writes through the buffer with reads through an aliased texture (per
  Apple's docs, both views must be hazard-tracked for the alias to be
  auto-tracked). Other hints retain Ghost's untracked fast path.

- Buffer-backed `ImageMetal` (`Device::sharedImage(buffer)`) now retains
  a reference to its backing buffer and emits `[computeEncoder
  useResource:buffer]` when the texture is bound to a kernel. This is
  Apple's documented pattern for indirect/aliased access; matches the
  hazard-tracking story above.

### Fixed

- Recorded `CommandBuffer` host-source snapshots are no longer freed
  mid-DMA on backends that perform truly async host→device transfers.
  The cb captured `src` into a `std::vector<uint8_t>` at record time,
  and `cb.reset()` after `submit()` freed that vector before the OpenCL
  driver's `clEnqueueWriteBuffer(blocking=false)` had read it —
  surfaced on NVIDIA OpenCL as stale/zeroed kernel inputs but a spec
  violation on any conforming OpenCL implementation. The cb now captures
  into a `HostBytes::adopt`-style handle, and the OpenCL backend retains
  it on the stream's pending-memory list until the matching `cl_event`
  reports complete. Metal/Vulkan/DirectX consumed the snapshot
  synchronously already; CUDA pageable host bytes go through driver
  staging so they were always safe.

- Buffer-backed Metal kernel image-sampling test (`ImageKernelTest.
  SharedImageKernelSample*`) — the test kernel declared two scalar
  arguments (`W`, `H`) at separate Metal buffer indices, but Ghost's
  Metal binding packs all scalar arguments into a single `MTLBuffer`,
  leaving `buffer(2)` unbound. Updated the kernel to use a single
  packed struct (matches the convention used by every other Metal
  test kernel in `ghost_test.h`).

### Changed

- `Buffer::copy(const CommandBuffer&, const void* src, ...)` and
  `Image::copy(const CommandBuffer&, const void* src, const BufferLayout&)`
  now capture the source bytes by value at call time instead of
  retaining the host pointer until `submit`. Stack-local sources are
  safe; the previous behavior silently corrupted dst when the caller's
  frame was gone by `submit`. Stream-encoded copies are unchanged
  (immediate, no extra allocation).

### Changed (breaking)

- **`CommandBuffer` no longer inserts an implicit barrier between
  dispatches on Vulkan, DirectX, or Metal.** Consumers that recorded
  chains of dependent dispatches into a `CommandBuffer` and relied on
  the backend's per-dispatch barrier must now either:
  - Insert `cb.barrier()` between dependent dispatches, or
  - Construct the cb with `CommandBuffer(dev, CommandBufferOptions{
    /*concurrent=*/false})` to restore the prior auto-barrier behavior.

  `Stream` semantics are unchanged — `Stream` still inserts the
  per-dispatch barrier by default.

- **OpenCL `Buffer::copy(Stream, const void*, ...)` and
  `Buffer::copyTo(Stream, void*, ...)` are now synchronous** (and the
  image analogs). Previously they used `clEnqueueWrite/ReadBuffer` with
  `blocking=false`, so the call returned before the DMA had finished
  reading `src` / writing `dst`. Per the OpenCL spec the caller had to
  keep the host pointer alive until the next `stream.sync()` — a contract
  Ghost didn't advertise on the upload direction and *contradicted* on
  the readback direction (`copyTo(Stream, …)` is documented as
  synchronous). Now matches the contract on every other backend: the
  call returns when the host bytes have been consumed (uploads) /
  produced (readbacks).

  **Performance impact for clients that batched many such calls** (e.g.
  Inferency feeding a sequence of uploads or pulling a sequence of
  readbacks): host-driver pipelining is gone — each call now stalls the
  host. To recover the async path *and* get a documented lifetime
  contract, pass the host buffer through `HostBytes::adopt`:

  ```cpp
  // Was (undocumented async; caller had to keep host alive until sync):
  buf.copy(stream, src_ptr, n);

  // Now (sync; src_ptr can be freed immediately):
  buf.copy(stream, src_ptr, n);

  // Async-with-Ghost-managed-lifetime (matches old perf, documented contract):
  buf.copy(stream, ghost::HostBytes::adopt(src_ptr, free), 0, n);
  ```

  The cb-replay paths (`CommandBuffer::copyBufferRaw(void*)`,
  `readBuffer(void*)`, image equivalents) also now block per op during
  `submit()`. Use the `HostBytes` overloads on `CommandBuffer` for the
  pipelined-batch behavior.

- **`StreamOpenCL::outOfOrder` no longer reflects the queue type;** it
  always means "use Ghost's event-chain machinery to order dependent
  ops" and is always `true` for streams Ghost creates or wraps. A change
  introduced earlier in 1.1 set it from the queue's actual
  `CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE` bit; on drivers that don't
  expose an out-of-order queue (Apple's OpenCL 1.2) the flag became
  false, disabling the event chain and silently racing dependent ops
  enqueued back-to-back on the same stream.

  `StreamOptions::forceEventChain` is now redundant (the chain is always
  on) and kept only for source compatibility.
