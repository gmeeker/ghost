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

#ifndef GHOST_ALLOCATOR_H
#define GHOST_ALLOCATOR_H

#include <ghost/image.h>

#include <cstddef>

namespace ghost {

/// @brief Host-supplied hook that overrides GPU and host memory allocation.
///
/// Subclass @c Allocator and install an instance on a @c Device with
/// @c Device::setAllocator to route Ghost's buffer/image/host-memory
/// allocations through your own pool, recycler, or host application's memory
/// subsystem.
///
/// Each method may return @c nullptr to decline the allocation, in which case
/// Ghost falls back to the device's default path. This is the "selective
/// pooling" use case — a host may intercept the shapes it cares about and let
/// everything else go through.
///
/// **Handle semantics.** The @c void* returned by an @c allocate* method is the
/// backend's native resource handle. Ghost takes ownership of that handle and
/// will pass it back to the matching @c free* method when the resource is
/// destroyed. The handle type is backend-specific:
///
/// | Backend  | Buffer/MappedBuffer/Image handle                         |
/// |----------|----------------------------------------------------------|
/// | Metal    | @c (__bridge_retained void*)id\<MTLBuffer> / @c id\<MTLTexture>
/// | | OpenCL   | @c cl_mem cast to @c void*                                |
/// | CUDA     | @c CUdeviceptr cast to @c (void*)(uintptr_t)              |
/// | Vulkan   | Host-owned pointer to a backend-specific struct (see docs) |
/// | DirectX  | @c ID3D12Resource* (Ghost calls @c Release on free)        |
/// | CPU      | @c void* heap pointer                                     |
///
/// **Lifetime.** The Device holds a @c shared_ptr to the @c Allocator. Buffers
/// and images do not outlive their device, so the allocator is guaranteed to be
/// alive when @c freeBuffer / @c freeImage / @c freeHostMemory is called on
/// resources that survive long enough to need it.
class Allocator {
 public:
  virtual ~Allocator() = default;

  /// @brief Allocate a GPU buffer.
  /// @return The backend's native handle, or @c nullptr to decline.
  virtual void* allocateBuffer(size_t bytes, const BufferOptions& opts) {
    (void)bytes;
    (void)opts;
    return nullptr;
  }

  /// @brief Free a buffer previously returned by @ref allocateBuffer.
  virtual void freeBuffer(void* handle, size_t bytes) {
    (void)handle;
    (void)bytes;
  }

  /// @brief Allocate a mapped (host-visible) GPU buffer.
  virtual void* allocateMappedBuffer(size_t bytes, const BufferOptions& opts) {
    (void)bytes;
    (void)opts;
    return nullptr;
  }

  /// @brief Free a mapped buffer previously returned by
  /// @ref allocateMappedBuffer.
  virtual void freeMappedBuffer(void* handle, size_t bytes) {
    (void)handle;
    (void)bytes;
  }

  /// @brief Allocate a GPU image (texture).
  virtual void* allocateImage(const ImageDescription& descr) {
    (void)descr;
    return nullptr;
  }

  /// @brief Free an image previously returned by @ref allocateImage.
  virtual void freeImage(void* handle, const ImageDescription& descr) {
    (void)handle;
    (void)descr;
  }

  /// @brief Allocate host (CPU) memory, e.g. for pinned-host staging buffers.
  /// @return Pointer to host memory, or @c nullptr to decline.
  virtual void* allocateHostMemory(size_t bytes) {
    (void)bytes;
    return nullptr;
  }

  /// @brief Free host memory previously returned by @ref allocateHostMemory.
  ///
  /// When an allocator is installed, Ghost always routes @c freeHostMemory
  /// through it. Implementations that may receive pointers they did not
  /// allocate are responsible for routing those to @c ::free internally.
  virtual void freeHostMemory(void* ptr) { (void)ptr; }
};

}  // namespace ghost

#endif
