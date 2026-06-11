// Copyright (c) 2025 Digital Anarchy, Inc. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless revuired by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#ifndef GHOST_CU_PTR_H
#define GHOST_CU_PTR_H

#include <cuda.h>
#include <ghost/cuda/exception.h>
#include <ghost/exception.h>

#include <memory>
#include <utility>

namespace ghost {
namespace cu {
template <typename TYPE>
class detail {};

template <>
class detail<void*> {
 public:
  static CUresult release(void* v) { return cuMemFreeHost(v); }
};

template <>
class detail<CUdeviceptr> {
 public:
  static CUresult release(CUdeviceptr v) { return cuMemFree(v); }
};

template <>
class detail<CUarray> {
 public:
  static CUresult release(CUarray v) { return cuArrayDestroy(v); }
};

// NOTE: `CUtexObject` is typedef'd to `unsigned long long`, identical to
// `CUdeviceptr` on 64-bit hosts. Using `cu::ptr<CUtexObject>` here would
// pick up `detail<unsigned long long>` = `detail<CUdeviceptr>`, whose
// destructor calls `cuMemFree()` on the texture object handle.
//
// Use: ptr<CUtexObject, detail<GhostCUtexObject>>

struct GhostCUtexObject {};

template <>
class detail<GhostCUtexObject> {
 public:
  static CUresult release(CUtexObject v) { return cuTexObjectDestroy(v); }
};

template <>
class detail<CUevent> {
 public:
  static CUresult release(CUevent v) { return cuEventDestroy(v); }
};

template <>
class detail<CUcontext> {
 public:
  static CUresult release(CUcontext v) { return cuCtxDestroy(v); }
};

template <>
class detail<CUstream> {
 public:
  static CUresult release(CUstream v) { return cuStreamDestroy(v); }
};

template <>
class detail<CUmodule> {
 public:
  static CUresult release(CUmodule v) { return cuModuleUnload(v); }
};

template <>
class detail<CUlinkState> {
 public:
  static CUresult release(CUlinkState v) { return cuLinkDestroy(v); }
};

#if CUDA_VERSION >= 11020
template <>
class detail<CUmemoryPool> {
 public:
  static CUresult release(CUmemoryPool v) { return cuMemPoolDestroy(v); }
};
#endif

/// @brief Reference-counted handle wrapper for CUDA driver objects.
///
/// Owning ptrs (constructed with @c retainObject=true, assigned a raw
/// handle, or written through @c operator&) share a heap-allocated control
/// node; @c DETAIL::release fires when the last sharing ptr drops. Copies
/// share ownership — copying never transfers or steals it — matching the
/// OpenCL/Metal wrapper semantics. Non-owning ptrs (retainObject=false)
/// carry just the raw value and never release it.
template <typename TYPE, typename DETAIL = detail<TYPE>>
class ptr {
 protected:
  // Owns the handle; releases it when the last sharing ptr drops. Lives on
  // the heap so the storage address is stable for the whole ownership
  // lifetime regardless of how the ptr objects themselves move (see
  // handleAddress()).
  struct Destroyer {
    TYPE value;

    explicit Destroyer(TYPE v = (TYPE)0) : value(v) {}

    Destroyer(const Destroyer&) = delete;
    Destroyer& operator=(const Destroyer&) = delete;

    ~Destroyer() {
      if (value) {
        CUresult err = DETAIL::release(value);
        if (err != CUDA_SUCCESS) {
          try {
            throw cu::runtime_error(err);
          } catch (...) {
            ghost::detail::stashError(std::current_exception());
          }
        }
      }
    }
  };

  std::shared_ptr<Destroyer> _destroy;  // null when non-owning
  TYPE value;                           // used only when non-owning

 public:
  explicit ptr(TYPE v = (TYPE)0, bool retainObject = true) : value(v) {
    if (retainObject && v) {
      _destroy = std::make_shared<Destroyer>(v);
    }
  }

  ptr(const ptr& v) : _destroy(v._destroy), value(v.value) {}

  ptr(ptr&& v) : _destroy(std::move(v._destroy)), value(v.value) {
    v.value = (TYPE)0;
  }

  ~ptr() = default;

  TYPE get() const { return _destroy ? _destroy->value : value; }

  bool owned() const { return _destroy != nullptr; }

  operator TYPE() const { return get(); }

  /// @brief Stable address of the live handle storage, valid for the
  /// lifetime of the ownership node (owning) or of this ptr (non-owning).
  /// CUDA kernel parameter arrays dereference at launch time and need this
  /// rather than a pointer into a relocatable ptr object.
  TYPE* handleAddress() { return _destroy ? &_destroy->value : &value; }

  /// @brief Drop this ptr's reference (and value). The handle is released
  /// only when the last sharing ptr lets go.
  void destroy() {
    _destroy.reset();
    value = (TYPE)0;
  }

  void reset() { destroy(); }

  /// @brief Disarm and return the handle: no sharing ptr will release it,
  /// and every sharer observes the value as cleared. The caller assumes
  /// ownership of the returned handle.
  TYPE release() {
    TYPE v = get();
    if (_destroy) {
      _destroy->value = (TYPE)0;
      _destroy.reset();
    }
    value = (TYPE)0;
    return v;
  }

  /// @brief Out-parameter accessor: drops the current handle and arms a
  /// fresh owning node for the API call to write into.
  TYPE* operator&() {
    destroy();
    _destroy = std::make_shared<Destroyer>();
    return &_destroy->value;
  }

  ptr& operator=(TYPE v) {
    destroy();
    value = v;
    if (v) _destroy = std::make_shared<Destroyer>(v);
    return *this;
  }

  ptr& operator=(const ptr& v) {
    // std::addressof: unary & is overloaded as the out-parameter accessor.
    if (this == std::addressof(v)) return *this;
    _destroy = v._destroy;
    value = v.value;
    return *this;
  }

  ptr& operator=(ptr&& v) {
    if (this == std::addressof(v)) return *this;
    _destroy = std::move(v._destroy);
    value = v.value;
    v.value = (TYPE)0;
    return *this;
  }
};
}  // namespace cu
}  // namespace ghost
#endif
