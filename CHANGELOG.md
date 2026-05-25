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
