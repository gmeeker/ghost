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
