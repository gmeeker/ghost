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

template <typename TYPE, typename DETAIL = detail<TYPE>>
class ptr {
 protected:
  bool _owned;

 public:
  TYPE value;

  explicit ptr(TYPE v = 0, bool retainObject = true)
      : value(v), _owned(retainObject) {}

  ptr(ptr& v) : value(v.value), _owned(v._owned) { v._owned = false; }

  ptr(ptr&& v) : value(v.value), _owned(v._owned) { v._owned = false; }

  ~ptr() { destroy(); }

  void destroy() {
    if (_owned && value) {
      CUresult err = DETAIL::release(value);
      if (err != CUDA_SUCCESS) {
        try {
          throw cu::runtime_error(err);
        } catch (...) {
          ghost::detail::stashError(std::current_exception());
        }
      }
    }
    _owned = false;
    value = (TYPE)0;
  }

  void reset() { destroy(); }

  TYPE release() {
    TYPE v = value;
    _owned = false;
    return v;
  }

  TYPE get() const { return value; }

  operator TYPE() const { return value; }

  TYPE* operator&() {
    destroy();
    _owned = true;
    return &value;
  }

  ptr& operator=(TYPE v) {
    destroy();
    value = v;
    _owned = true;
    return *this;
  }

  ptr& operator=(ptr& v) {
    destroy();
    value = v.value;
    _owned = v._owned;
    v._owned = false;
    return *this;
  }

  ptr& operator=(ptr&& v) {
    destroy();
    value = v.value;
    _owned = v._owned;
    v._owned = false;
    return *this;
  }
};
}  // namespace cu
}  // namespace ghost
#endif
