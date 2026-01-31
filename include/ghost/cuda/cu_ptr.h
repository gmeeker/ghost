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

namespace ghost {
namespace cu {
template <typename TYPE>
class detail {};

template <>
class detail<void*> {
 public:
  static void release(void* v) { cuMemFreeHost(v); }
};

template <>
class detail<CUdeviceptr> {
 public:
  static void release(CUdeviceptr v) { cuMemFree(v); }
};

template <>
class detail<CUarray> {
 public:
  static void release(CUarray v) { cuArrayDestroy(v); }
};

template <>
class detail<CUevent> {
 public:
  static void release(CUevent v) { cuEventDestroy(v); }
};

template <>
class detail<CUcontext> {
 public:
  static void release(CUcontext v) { cuCtxDestroy(v); }
};

template <>
class detail<CUstream> {
 public:
  static void release(CUstream v) { cuStreamDestroy(v); }
};

template <>
class detail<CUmodule> {
 public:
  static void release(CUmodule v) { cuModuleUnload(v); }
};

template <>
class detail<CUlinkState> {
 public:
  static void release(CUlinkState v) { cuLinkDestroy(v); }
};

template <typename TYPE, typename DETAIL = detail<TYPE>>
class ptr {
 protected:
  bool _owned;

 public:
  TYPE value;

  explicit ptr(TYPE v = 0, bool retainObject = true)
      : value(v), _owned(retainObject) {}

  ptr(ptr& v) : value(v.value), _owned(v._owned) { v._owned = false; }

  ~ptr() { destroy(); }

  void destroy() {
    if (_owned) {
      DETAIL::release(value);
      _owned = false;
    }
    value = (TYPE)0;
  }

  TYPE release() {
    TYPE v = value;
    _owned = false;
    return v;
  }

  void reset() { destroy(); }

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
};
}  // namespace cu
}  // namespace ghost
#endif