# Ghost Coding Conventions

This document captures conventions used across Ghost's backend implementations.
New backends and edits to existing ones should follow these. Existing code that
deviates is being normalized — when in doubt, match `OpenCL` for general C++
style and `cu_ptr.h` for the smart-pointer pattern.

The conventions here exist mostly to keep all six GPU backends (CUDA, OpenCL,
Metal, Vulkan, DirectX, CPU) visually and structurally parallel, so a fix in
one backend can be ported to another by analogy without surprises.

---

## Smart-pointer wrappers for native handles

Every backend wraps its native handle types in a custom smart-pointer template
modeled after `cu::ptr`:

| Backend  | Header                       | Template                  |
|----------|------------------------------|---------------------------|
| CUDA     | `ghost/cuda/cu_ptr.h`        | `cu::ptr<T>`              |
| OpenCL   | `ghost/opencl/ptr.h`         | `opencl::ptr<T>`          |
| Metal    | `ghost/objc/ptr.h`           | `objc::ptr<id<T>>`        |
| Vulkan   | `ghost/vulkan/ptr.h`         | `vk::ptr<T>`              |
| DirectX  | `<wrl/client.h>` (Microsoft) | `Microsoft::WRL::ComPtr<T>` |
| CPU      | n/a                          | raw `void*`               |

These wrappers all share the same essential surface, which mirrors
`std::shared_ptr` / `std::unique_ptr`:

- `T get() const` — returns the underlying handle (by value, since handles are
  pointer-typed already).
- `operator T() const` — implicit conversion for use in API calls.
- `T release()` — gives up ownership without destroying.
- `void reset()` — destroy the held handle now.
- Move-only (`cu::ptr`, `vk::ptr`) or refcounted (`opencl::ptr`, `ComPtr`,
  ARC-managed `objc::ptr`).
- An ownership flag (`_owned` in `cu::ptr`/`vk::ptr`) so that the same wrapper
  can hold borrowed handles for sub-resources / aliases without taking
  ownership.

**`vk::ptr` and `cu::ptr` additionally provide `T* operator&()` for use as the
out-parameter of `vkCreateXxx` / `cuXxxCreate`.** This overload is destructive
(it resets the held value before returning the address), so it's only safe at
allocation/creation sites. For Vulkan APIs that take a `const T*` to an
*existing* handle (e.g. `vkResetFences`, `vkWaitForFences`), use a local
temporary instead:

```cpp
VkFence f = fence;                              // not &fence
vkResetFences(dev.device, 1, &f);
```

This matches what you'd write with `std::shared_ptr` and avoids introducing
a backend-specific accessor.

### When to add a new specialization

When a backend grows a new native handle type that needs RAII cleanup, add
a `detail<T>` specialization to that backend's `ptr.h` rather than writing a
hand-rolled destructor. The whole point of these wrappers is that destructor
ordering is handled automatically by the compiler.

Hand-rolled destructors in impl classes should only exist for state that
*genuinely* can't be wrapped (e.g. POSIX/Win32 handles like
`StreamDirectX::fenceEvent`, or for non-trivial state like `StreamVulkan`'s
sync-on-destroy).

---

## Resource lifetime: device must outlive its children

The convention across all backends is **the user is responsible for destroying
child resources (buffers, images, streams, libraries, functions) before
destroying the Device they were allocated from**. This matches CUDA, Metal,
DirectX, and OpenCL semantics natively, and Vulkan now follows it after the
removal of the `deviceAlive` sentinel pattern.

### Why no automatic enforcement?

We do *not* make impl classes hold a `shared_ptr<Device>` to enforce this
automatically:

- It would require `enable_shared_from_this` plumbing on every Device subclass.
- It would cycle-protect against the rule rather than codify it.
- For backends with internal ref counting (Metal/ARC, DirectX/ComPtr, OpenCL
  via cl_context retain), the parent device is *already* kept alive
  automatically through the smart pointer's underlying type — no Ghost-side
  bookkeeping needed.

For CUDA and Vulkan (which don't have native ref counting on contexts/devices),
the rule is documented and enforced by convention.

### Functions hold a strong reference to their Library

`ghost::Function` (the public bridge class) holds a `shared_ptr<implementation::Library>`
parent. This is the one place where Ghost adds parent ownership above what the
backends provide, because dropping the Library wrapper would unload the
underlying compiled module (CUmodule, cl_program, dlopened .so) while functions
from it are still in use.

### Sub-resources hold a strong reference to their parent

`SubBufferXxx` and image-from-image / image-from-buffer aliases hold a
`shared_ptr<Buffer>` (or per-handle `_owned=false` flag) referring to the
parent resource. This ensures the parent's underlying allocation outlives any
view into it.

---

## Class layout and encapsulation

Impl classes should follow this layout:

```cpp
class FooBackend : public Foo {
 public:
  // Smart-pointer-wrapped native handles, ordered by lifetime dependency
  // (independent handles first, dependents later).
  backend::ptr<NativeHandleA> handleA;
  backend::ptr<NativeHandleB> handleB;

  // Public constructors. Backends usually have an "allocate" ctor that takes
  // (DeviceXxx&, size, opts) and a "wrap" ctor that takes existing handles.
  FooBackend(const DeviceXxx& dev, size_t bytes, const BufferOptions& opts = {});
  FooBackend(backend::ptr<NativeHandleA> handle_, size_t bytes);

  // Override the public Foo interface.
  size_t size() const override;
  void copy(const ghost::Stream& s, /* ... */) override;
  // ...

 private:
  // Internal cached state, configuration, helpers. Member names are
  // underscore-prefixed.
  size_t _someCachedValue;
  void _internalHelper();
};
```

### Member visibility rules

- **Public**: handles other impl classes legitimately need to read (`mem`,
  `program`, `kernel`, `queue`), and the public override interface.
- **Private**: cached state, configuration, internal helpers, parent device
  references, anything other backends shouldn't reach into.

`OpenCL` is the closest match to this convention today; Vulkan and DirectX
device-side classes were originally written with everything public and are
being normalized.

### Member naming

- **Public** members: no underscore prefix (`mem`, `queue`, `program`, `kernel`).
- **Private** members: leading underscore (`_dev`, `_size`, `_pool`).

### Section ordering

`public:` comes first, then `protected:`, then `private:`. Don't put `private:`
or `protected:` before `public:` even if it lets you forward-reference fewer
things — readers expect to see the public surface first.

### `virtual ... override`

Use both keywords together on every override:

```cpp
virtual void execute(...) override;
virtual const ImageDescription& description() const override;
```

`override` alone is technically sufficient in C++17, but the existing codebase
uses both consistently — match that for grep-ability and visual parallel with
the base class declarations.

---

## Parent-device coupling

**Resource impl classes (Buffer, Image) should NOT hold a `const DeviceXxx& dev`
reference**, because:

1. The native handles already know everything they need to clean themselves up:
   `cu::ptr<CUdeviceptr>` calls `cuMemFree`, `opencl::ptr<cl_mem>` calls
   `clReleaseMemObject` (cl_mem internally retains cl_context),
   `objc::ptr<id<MTLBuffer>>` is ARC-managed, `vk::ptr<VkBuffer>` stores its
   own `VkDevice`, `ComPtr<ID3D12Resource>` retains the device through COM.
2. Holding the reference creates a dangling-pointer surface for no benefit:
   the resource silently captures a `Device&` that the user might destroy
   before the resource.
3. It locks the resource to a single Device instance, blocking cross-device
   sharing.

**Stream and CommandBuffer impls MAY hold a device reference**, because
Stream/CommandBuffer are conceptually per-device and need access to multiple
device-level handles (`computeQueue`, `computeQueueFamily`, `descriptorPool`)
that aren't carried by individual handle wrappers.

**Function and Library impls SHOULD hold a device reference** (private
`const DeviceXxx& _dev`), because compilation/specialization touches many
device-level handles and helpers.

### Helpers like `createBuffer` belong on Device

When a backend has a multi-step "allocate buffer + memory + bind" sequence
(Vulkan), expose it as a method on `DeviceXxx` rather than reinventing it in
every Buffer/Image impl. The helper takes `vk::ptr<>&` out-parameters which the
caller pre-initializes with the target device.

### Staging-buffer creation belongs on the Encoder

Resources that need temporary staging buffers for host-side reads/writes
should attach them to the active `Encoder` (Stream or CommandBuffer), not the
Device. The encoder owns the staging buffer's lifetime and tracks it through
`addStagingResource` / `DeferredRead`. In Vulkan this lives on the
`VulkanEncoder` mixin shared by `StreamVulkan` and `CommandBufferVulkan`; in
DirectX the equivalent state lives on the encoder side as well.

---

## Encoder, Stream, and CommandBuffer

Ghost has a three-layer execution-target hierarchy:

- **`ghost::Encoder`** (public) / **`implementation::Encoder`** (impl) — the
  abstract base for "things that can record GPU operations." All
  `Buffer::copy`, `Image::copy*`, and `Function::execute` overloads now take
  a `const ghost::Encoder&` rather than a `const ghost::Stream&`, so the same
  resource methods work against either a live stream or a deferred command
  buffer.
- **`ghost::Stream`** — serialized in-order execution. Each operation waits
  for the previous one before starting.
- **`ghost::CommandBuffer`** — deferred execution. Operations are recorded
  on call, replayed on `submit(stream)`. No implicit synchronization between
  recorded operations — callers insert barriers via `cb.barrier()`.

### Backend Encoder mixins

Backends that share state between their Stream and CommandBuffer impls
introduce a backend-specific Encoder mixin that both inherit from. Vulkan is
the canonical example:

```cpp
class VulkanEncoder {                 // holds VkCommandBuffer, staging,
 public:                              // deferred reads, concurrent flag, etc.
  const DeviceVulkan& dev;
  VkCommandBuffer commandBuffer;
  std::vector<DeferredRead> deferredReads;
  // ...
  virtual void begin() = 0;
  void addStagingResource(vk::ptr<VkBuffer>, vk::ptr<VkDeviceMemory>);
};

class StreamVulkan        : public Stream,                 public VulkanEncoder { /* ... */ };
class CommandBufferVulkan : public RecordedCommandBuffer,  public VulkanEncoder { /* ... */ };
```

Callers cross-cast a `ghost::Encoder` to the backend mixin with a free
function (`vulkanEncoder(enc)`) which throws `ghost::unsupported_error` if
the encoder isn't the expected backend.

### `RecordedCommandBuffer` fallback

`implementation::RecordedCommandBuffer` (in
`include/ghost/implementation/recorded_command_buffer.h`) is the default
record-and-replay command buffer used by backends without a native
command-buffer concept (CUDA, OpenCL, CPU). It stores tagged `Command`
variants in a `std::vector<Command>` and replays them onto the target
encoder in `submit()`.

Backends with native command buffers (Vulkan, DirectX, Metal) subclass
`RecordedCommandBuffer` to reuse the variant-recording machinery, but
override `submit()` (and any other methods that need backend-native handling,
e.g. `BarrierCmd` → `vkCmdPipelineBarrier`) to replay into their native
command buffer object instead of replaying onto the stream's queue directly.
See `CommandBufferVulkan` and the matching DirectX class.

Each `Command` variant captures `shared_ptr` to the involved impl resources,
which keeps them alive between record and submit. Don't store raw pointers
in new command variants.

---

## Constructor patterns

### Allocating constructor

```cpp
BufferXxx(const DeviceXxx& dev, size_t bytes, const BufferOptions& opts = {});
```

Takes the device by reference (used only at construction; not stored), the
size, and a `BufferOptions` value (always with `= {}` default). Allocates new
backing storage.

### Wrapping constructor

```cpp
BufferXxx(backend::ptr<NativeHandle> handle_, size_t bytes);
```

Takes an existing smart-pointer-wrapped handle by value (move semantics for
move-only ptrs, copy for refcounted ones), and the size. Used by sub-buffers
and external-handle wrapping.

### Borrowed-handle constructor

When a sub-resource needs to alias a parent's handle without taking ownership,
construct the smart pointer with the borrow flag:

```cpp
SubBufferVulkan::SubBufferVulkan(std::shared_ptr<Buffer> parent,
                                 const DeviceVulkan& dev, VkBuffer buf,
                                 size_t offset, size_t bytes)
    : BufferVulkan(dev, buf, VK_NULL_HANDLE, bytes, /*owns=*/false),
      _parent(parent),
      _offset(offset) {}
```

The parent is held via `std::shared_ptr<Buffer> _parent` so its destructor
runs after the sub-buffer's.

---

## Attribute lifetime (kernel arguments)

`ghost::Attribute` holds strong references to the underlying impl objects for
`Buffer`, `Image`, and `ArgumentBuffer`-typed attributes. This means an
Attribute outlives the user's wrapper objects, which is required for deferred
execution paths like `CommandBuffer` where the recorded args need to remain
valid between `record()` and `submit()`.

Backends accessing these from inside `Function::execute()` should use:

- `attr.bufferImpl()` — returns `const std::shared_ptr<implementation::Buffer>&`
- `attr.imageImpl()` — returns `const std::shared_ptr<implementation::Image>&`
- `attr.argumentBuffer()` — returns `ArgumentBuffer*` (into the snapshot copy)

**Do NOT use** the historical `asBuffer() / asImage() / asArgumentBuffer()`
accessors that returned wrapper pointers — those have been removed because they
returned addresses that could dangle in deferred-execution paths.

---

## Include conventions

Use absolute include paths for Ghost headers, even within the same directory:

```cpp
// Good
#include <ghost/cuda/cu_ptr.h>
#include <ghost/vulkan/ptr.h>

// Avoid (CUDA and OpenCL still do this; being normalized)
#include "cu_ptr.h"
#include "ptr.h"
```

System and third-party headers (Vulkan, D3D12, CUDA, OpenCL) keep their
native include style.

---

## File organization

Each backend lives in two parallel directories:

- `include/ghost/<backend>/` — public headers visible to consumers
- `src/<backend>/` — implementation .cpp / .mm files

Each backend has:

- `device.h` / `<backend>_device.cpp` — public `Device` subclass and core
  device implementation
- `impl_device.h` / (in same .cpp) — `implementation::DeviceXxx`,
  `BufferXxx`, `ImageXxx`, `StreamXxx`, `EventXxx`, and (where the backend
  has a native command-buffer object) `CommandBufferXxx` impl classes, plus
  any backend-specific `<Backend>Encoder` mixin
- `impl_function.h` / `<backend>_function.cpp` — `implementation::FunctionXxx`,
  `LibraryXxx`
- `exception.h` / `<backend>_exception.cpp` — backend-specific
  `runtime_error` and `checkError` helper
- `ptr.h` (or equivalent) — smart pointer template + handle traits
- `reflect.h` / `<backend>_reflect.cpp` — *optional*, present in Vulkan and
  DirectX for SPIR-V / DXIL reflection used to map kernel arguments to
  descriptor slots

The `<backend>_function.cpp` file is wrapped in `#if WITH_<BACKEND>` so the
TU compiles to a stub when the backend is disabled.

Cross-backend shared types live in `include/ghost/implementation/`:

- `impl_device.h` — abstract `Device`, `Buffer`, `Image`, `Stream`, `Encoder`,
  `Event`, `CommandBuffer` base classes
- `impl_function.h` — abstract `Function`, `Library` base classes
- `recorded_command_buffer.h` — `RecordedCommandBuffer` fallback +
  `Command` variant definitions

---

## Tests

- Tests live in `test/`. The kernel-shader sources live in `test/kernels/`.
- Use the `GhostKernelTest` fixture for tests that need a real device + a
  kernel. Use `GhostTest` for device-only tests.
- Use `GHOST_INSTANTIATE_KERNEL_TESTS(MyTestClass)` at the bottom of each test
  file to instantiate against all available backends.
- Tests that exercise lifetime invariants (Buffer/Function/Library outliving
  their wrappers, vector reallocation mid-batch) are valuable — most lifetime
  bugs in Ghost slipped through because tests kept everything in the same
  outer scope as `submit()`. See `CommandBufferTest.BufferWrappersOutOfScope`
  and `KernelTest.FunctionOutlivesLibrary` for templates.

---

## When to deviate

These conventions exist to make Ghost easier to maintain across five backends.
If you find yourself fighting them, the convention is probably wrong for your
case — flag it in the PR / commit message rather than silently breaking the
pattern. Cosmetic deviation (private/public ordering, naming) is cheap to fix
later; architectural deviation (parent-device coupling, hand-rolled
destructors) tends to ossify.
