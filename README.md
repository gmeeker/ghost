# Ghost

**A unified C++ GPU compute abstraction library**

Ghost provides a single, clean API for GPU compute across CUDA, Metal, OpenCL, Vulkan, DirectX, and CPU backends. Write your host code once, run it on any supported GPU.

Ghost abstracts the host-side API -- device management, memory allocation, kernel dispatch, and synchronization. Writing portable kernel code is outside the scope of this project. Ghost accepts pre-compiled (AOT) GPU binaries, SPIR-V, PTX, or API-specific source code (OpenCL C, Metal Shading Language, etc.). For portable kernel source, consider using [Slang](https://shader-slang.com/) or a similar cross-compilation tool.

## Features

- **Multi-backend** -- CUDA, Metal, OpenCL, Vulkan, DirectX, and a CPU fallback, all behind one API
- **Flexible kernel loading** -- Load pre-compiled GPU binaries, SPIR-V, PTX, or compile from API-specific source at runtime
- **No hard dependencies** -- GPU libraries are loaded dynamically; your app runs even if a driver is missing
- **Binary caching** -- Compiled kernels are automatically cached to disk, eliminating redundant compilation across runs
- **Buffer & image management** -- Allocate, copy, fill, and map GPU memory with a consistent interface
- **Streams & events** -- Asynchronous execution with cross-stream synchronization
- **Command buffers** -- Record and batch GPU operations for deferred execution
- **Device queries** -- Discover available GPUs and query capabilities through a type-safe attribute system
- **Modern C++** -- RAII resource management with `std::shared_ptr`, C++17 throughout

## Quick Start

```cpp
#include <ghost/opencl/device.h>

using namespace ghost;

// Create a device
DeviceOpenCL device;
auto stream = device.defaultStream();

// Compile a kernel from OpenCL C source
auto library = device.loadLibraryFromText(R"(
    __kernel void scale(__global float* out,
                        __global const float* in,
                        float factor) {
        int id = get_global_id(0);
        out[id] = in[id] * factor;
    }
)");
auto kernel = library.lookupFunction("scale");

// Allocate buffers
auto inBuf  = device.allocateBuffer(N * sizeof(float));
auto outBuf = device.allocateBuffer(N * sizeof(float));
inBuf.copy(stream, host_data.data(), N * sizeof(float));

// Dispatch
LaunchArgs args;
args.global_size(N).local_size(256);
kernel(args, stream)(outBuf, inBuf, 2.0f);

// Read back results
outBuf.copyTo(stream, result.data(), N * sizeof(float));
stream.sync();
```

## Building

Ghost uses CMake and requires a C++17 compiler.

**Linux / Windows (CUDA + OpenCL):**

```bash
cmake -B build -DWITH_CUDA=ON -DWITH_OPENCL=ON
cmake --build build
```

**macOS (Metal + OpenCL):**

```bash
cmake -B build -DWITH_METAL=ON -DWITH_OPENCL=ON
cmake --build build
```

### Backend Options

| Option | Default | Notes |
|---|---|---|
| `WITH_CUDA` | OFF | CUDA driver API |
| `WITH_CUDA_LINK` | OFF | Link directly to CUDA driver (vs. dynamic loading) |
| `WITH_CUDA_NVRTC` | OFF | Enable JIT compilation of CUDA source code via NVRTC |
| `WITH_CUDA_NVRTC_STATIC` | OFF | Statically link NVRTC |
| `WITH_METAL` | OFF | macOS only |
| `WITH_OPENCL` | ON | |
| `WITH_VULKAN` | OFF | Uses MoltenVK on macOS |
| `WITH_DIRECTX` | OFF | Windows only |

### Package Management

Ghost supports [Conan](https://conan.io/) via the included `conanfile.py`.

### Running Tests

```bash
cd build && ctest
```

## API Overview

### Devices

Each backend provides a concrete device class: `DeviceCUDA`, `DeviceMetal`, `DeviceOpenCL`, `DeviceVulkan`, `DeviceDirectX`, `DeviceCPU`. All share the same interface.

```cpp
// Enumerate available GPUs
auto gpus = DeviceOpenCL::enumerateDevices();
for (auto& info : gpus) {
    std::cout << info.name << std::endl;
}

// Create a device for a specific GPU
DeviceOpenCL device(gpus[0]);

// Query device capabilities
auto name = device.getAttribute(kDeviceName);
auto mem  = device.getAttribute(kDeviceMemory);
```

### Using Ghost inside a library

When Ghost is used inside a library (as opposed to an application that owns the
thread), the CUDA backend needs to manage the thread's current context so that
calls into other CUDA libraries — or callers that have their own context —
don't see it corrupted. Wrap each public entry point in a `Device::Active`
scope:

```cpp
void mylib_process(ghost::Device& dev, ...) {
    ghost::Device::Active scope(dev);
    // ... allocate buffers, launch kernels, sync streams ...
}
```

On construction, `Active` saves the thread's current CUDA context and makes
`dev`'s context current. On destruction, it synchronizes and restores the
previous context. On non-CUDA backends `Active` is a no-op, so library code
does not need backend-specific branches.

Errors surfaced from prior asynchronous work (e.g. a kernel that failed after
its dispatch returned) are deferred to the next user-visible Ghost call on the
same thread rather than thrown from destructors or silently dropped by
resource-release paths.

### Loading Kernels

Ghost supports multiple ways to load GPU programs:

```cpp
// Compile from API-specific source (OpenCL C, Metal Shading Language, etc.)
auto lib = device.loadLibraryFromText(source_code, options);

// Load a pre-compiled binary (SPIR-V, PTX, metallib, etc.)
auto lib = device.loadLibraryFromData(data, length, options);

// Load from a file
auto lib = device.loadLibraryFromFile("kernels.spv");

auto kernel = lib.lookupFunction("my_kernel");
```

### Buffers

```cpp
// GPU buffer
auto buf = device.allocateBuffer(size);
buf.copy(stream, host_ptr, size);           // Host -> GPU
buf.copyTo(stream, host_ptr, size);         // GPU -> Host
buf.copy(stream, other_buf, bytes);         // GPU -> GPU
buf.fill(stream, 0, size, uint8_t(0));      // Fill with pattern

// Mapped buffer (direct host access to GPU memory)
auto mapped = device.allocateMappedBuffer(size);
float* ptr = static_cast<float*>(mapped.map(stream, Access::WriteOnly));
// ... write directly ...
mapped.unmap(stream);

// Sub-buffers
auto sub = buf.createSubBuffer(offset, size);
```

#### Sub-buffers and arena allocation

Sub-buffers are views into a parent buffer at a given byte offset. They can be passed to kernels and used as copy sources/destinations transparently — backends translate the offset internally (`setBuffer:offset:` on Metal, `VkDescriptorBufferInfo.offset` on Vulkan, base+offset pointer on CUDA/CPU/DirectX, real `clCreateSubBuffer` on OpenCL).

This makes sub-buffers a good fit for arena-style memory planning: allocate one large buffer, then carve it into per-tensor or per-pass views.

```cpp
size_t align = device.getAttribute(kDeviceBufferAlignment).asInt();
auto roundUp = [&](size_t n) { return (n + align - 1) & ~(align - 1); };

auto arena = device.allocateBuffer(totalSize);
auto a = arena.createSubBuffer(0, roundUp(sizeA));
auto b = arena.createSubBuffer(roundUp(sizeA), roundUp(sizeB));
kernel(launch, stream)(a, b, ...);
```

Notes:
- Always pad sub-buffer offsets to `kDeviceBufferAlignment`. OpenCL maps this to `CL_DEVICE_MEM_BASE_ADDR_ALIGN`; Vulkan uses `minStorageBufferOffsetAlignment`.
- Sub-sub-buffers (a sub-buffer of a sub-buffer) are supported on every backend. OpenCL forbids this at the driver level, so Ghost transparently re-roots to the parent and accumulates offsets.
- The parent buffer must outlive its sub-buffers. Sub-buffers hold a `shared_ptr` to the parent, so this is automatic if you keep the sub-buffer alive.

### Images

```cpp
ImageDescription desc(Size3(width, height, 1),
                      PixelOrder_RGBA,
                      DataType_Float,
                      Stride2(row_stride, 0));

auto image = device.allocateImage(desc);
image.copy(stream, pixel_data, desc);       // Host -> GPU
image.copyTo(stream, output, desc);         // GPU -> Host
image.copy(stream, other_image);            // Image -> Image
```

### Streams & Synchronization

```cpp
auto s1 = device.createStream();
auto s2 = device.createStream();

// Enqueue work on parallel streams
kernel(args, s1)(buf1);
kernel(args, s2)(buf2);

// Cross-stream dependency
Event event = s1.record();
s2.waitForEvent(event);

// Timing
Event start = stream.record();
kernel(args, stream)(buf);
Event end = stream.record();
stream.sync();
double ms = Event::elapsed(start, end);
```

### Kernel Dispatch

Kernels use a two-step call syntax: bind launch configuration and encoder, then pass arguments. Both `Stream` and `CommandBuffer` inherit from `Encoder`, so the same kernel works with either.

```cpp
LaunchArgs args;
args.global_size(512, 512).local_size(16, 16);  // 2D dispatch

auto fn = library.lookupFunction("my_kernel");
fn(args, stream)(outputBuf, inputBuf, inputImage, 3.14f);
```

### Command Buffers

Command buffers record GPU operations for deferred, batched execution. They accept the same operations as streams (kernel dispatch, buffer/image copies, fills) through the shared `Encoder` interface.

Unlike streams, command buffers provide no implicit ordering between operations. Use `barrier()` for explicit synchronization within a batch.

```cpp
ghost::CommandBuffer cb(device);
buf.copy(cb, hostData, bytes);
fn(args, cb)(outputBuf, inputBuf, 3.14f);
cb.barrier();
result.copy(cb, outputBuf, bytes);
cb.submit(stream);
```

## Architecture

Ghost uses a bridge/pimpl pattern. Public API classes (`Device`, `Function`, `Library`, `Stream`, `CommandBuffer`, `Buffer`, `Image`) are thin wrappers that delegate to virtual implementation interfaces. `Stream` and `CommandBuffer` both inherit from `Encoder`, so GPU operations (kernel dispatch, buffer/image copies) accept either through a single interface. Each backend provides concrete implementations of these interfaces.

```
include/ghost/
    device.h, function.h, encoder.h,            # Public API
    command_buffer.h, image.h, ...
    implementation/
        impl_device.h, impl_function.h          # Virtual interfaces
    cuda/   metal/   opencl/   vulkan/   ...    # Backend headers
src/
    cuda/   metal/   opencl/   vulkan/   ...    # Backend implementations
```

## License

BSD 3-Clause. See [LICENSE](LICENSE) for details.
