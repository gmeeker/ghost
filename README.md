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
kernel(stream, args, outBuf, inBuf, 2.0f);

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
kernel(s1, args, buf1);
kernel(s2, args, buf2);

// Cross-stream dependency
Event event = s1.record();
s2.waitForEvent(event);

// Timing
Event start = stream.record();
kernel(stream, args, buf);
Event end = stream.record();
stream.sync();
double ms = Event::elapsed(start, end);
```

### Kernel Dispatch

Kernels are dispatched with a function-call syntax. Arguments are passed directly -- buffers, images, and scalar values.

```cpp
LaunchArgs args;
args.global_size(512, 512).local_size(16, 16);  // 2D dispatch

auto fn = library.lookupFunction("my_kernel");
fn(stream, args, outputBuf, inputBuf, inputImage, 3.14f);
```

## Architecture

Ghost uses a bridge/pimpl pattern. Public API classes (`Device`, `Function`, `Library`, `Stream`, `Buffer`, `Image`) are thin wrappers that delegate to virtual implementation interfaces. Each backend provides concrete implementations of these interfaces.

```
include/ghost/
    device.h, function.h, image.h, ...          # Public API
    implementation/
        impl_device.h, impl_function.h          # Virtual interfaces
    cuda/   metal/   opencl/   vulkan/   ...    # Backend headers
src/
    cuda/   metal/   opencl/   vulkan/   ...    # Backend implementations
```

## License

BSD 3-Clause. See [LICENSE](LICENSE) for details.
