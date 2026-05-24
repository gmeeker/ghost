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
