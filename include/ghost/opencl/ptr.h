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

#ifndef GHOST_OPENCL_PTR_H
#define GHOST_OPENCL_PTR_H

#if __APPLE_CC__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace ghost {
namespace opencl {
template <typename T>
class detail {};

template <>
class detail<cl_command_queue> {
 public:
  static void release(cl_command_queue v) { clReleaseCommandQueue(v); }

  static void retain(cl_command_queue v) { clRetainCommandQueue(v); }
};

template <>
class detail<cl_context> {
 public:
  static void release(cl_context v) { clReleaseContext(v); }

  static void retain(cl_context v) { clRetainContext(v); }
};

template <>
class detail<cl_device_id> {
 public:
  static void release(cl_device_id v) { clReleaseDevice(v); }

  static void retain(cl_device_id v) { clRetainDevice(v); }
};

template <>
class detail<cl_event> {
 public:
  static void release(cl_event v) { clReleaseEvent(v); }

  static void retain(cl_event v) { clRetainEvent(v); }
};

template <>
class detail<cl_kernel> {
 public:
  static void release(cl_kernel v) { clReleaseKernel(v); }

  static void retain(cl_kernel v) { clRetainKernel(v); }
};

template <>
class detail<cl_mem> {
 public:
  static void release(cl_mem v) { clReleaseMemObject(v); }

  static void retain(cl_mem v) { clRetainMemObject(v); }
};

template <>
class detail<cl_program> {
 public:
  static void release(cl_program v) { clReleaseProgram(v); }

  static void retain(cl_program v) { clRetainProgram(v); }
};

template <>
class detail<cl_sampler> {
 public:
  static void release(cl_sampler v) { clReleaseSampler(v); }

  static void retain(cl_sampler v) { clRetainSampler(v); }
};

template <typename T, typename DETAIL = detail<T>>
class ptr {
 public:
  typedef T cl_type;

 protected:
  cl_type object_;

  void retain() {
    if (object_) DETAIL::retain(object_);
  }

 public:
  explicit ptr(cl_type obj = nullptr, bool retainObject = false)
      : object_(obj) {
    if (!retainObject) retain();
  }

  ptr(const ptr& rhs) : object_(rhs.object_) { retain(); }

  ptr(ptr&& rhs) : object_(rhs.object_) { rhs.object_ = nullptr; }

  ~ptr() { reset(); }

  void reset() {
    if (object_) {
      DETAIL::release(object_);
      object_ = nullptr;
    }
  }

  cl_type release() {
    cl_type obj = object_;
    object_ = nullptr;
    return obj;
  }

  cl_type get() const { return object_; }

  operator cl_type() const { return object_; }

  cl_type* operator&() {
    reset();
    return &object_;
  }

  ptr& operator=(cl_type rhs) {
    reset();
    object_ = rhs;
    retain();
    return *this;
  }

  ptr& operator=(const ptr& rhs) {
    if (get() != rhs.get()) {
      reset();
      object_ = rhs.object_;
      retain();
    }
    return *this;
  }

  ptr& operator=(ptr&& rhs) {
    if (get() != rhs.get()) {
      reset();
      object_ = rhs.object_;
      rhs.object_ = nullptr;
    }
    return *this;
  }
};

template <typename T, typename DETAIL = detail<T>>
class array {
 public:
  typedef T cl_type;

 protected:
  std::vector<cl_type> objects_;

  void retain(cl_type obj) {
    if (obj) DETAIL::retain(obj);
  }

 public:
  explicit array(cl_type obj = nullptr, bool retainObject = false) {
    if (obj) {
      objects_.push_back(obj);
      if (!retainObject) retain(obj);
    }
  }

  array(const array& rhs) : objects_(rhs.objects_) {
    for (auto i = objects_.begin(); i != objects_.end(); ++i) {
      retain(*i);
    }
  }

  array(array&& rhs) : objects_(rhs.objects_) { rhs.objects_.clear(); }

  ~array() { reset(); }

  void reset() {
    for (auto i = objects_.begin(); i != objects_.end(); ++i) {
      DETAIL::release(*i);
    }
    objects_.clear();
  }

  const cl_type* get() const {
    return objects_.empty() ? nullptr : &objects_[0];
  }

  size_t size() const { return objects_.size(); }

  operator const cl_type*() const { return get(); }

  cl_type* operator&() {
    reset();
    objects_.resize(1);
    objects_[0] = nullptr;
    return &objects_[0];
  }

  array& push(cl_type rhs) {
    if (rhs) {
      objects_.push_back(rhs);
      retain(rhs);
    }
    return *this;
  }

  array& push(const ptr<T>& rhs) {
    cl_type obj = rhs.get();
    if (obj) {
      objects_.push_back(obj);
      retain(obj);
    }
    return *this;
  }

  array& push(ptr<T>&& rhs) {
    if (rhs.get()) {
      objects_.push_back(nullptr);  // reserve memory first
      objects_[objects_.size() - 1] = rhs.release();
    }
    return *this;
  }

  array& push(const array& rhs) {
    objects_.reserve(objects_.size() + rhs.objects_.size());
    for (auto i = rhs.objects_.begin(); i != rhs.objects_.end(); ++i) {
      objects_.push_back(*i);
      retain(*i);
    }
    return *this;
  }

  array& push(array&& rhs) {
    objects_ = rhs.objects_;
    rhs.objects_.clear();
    return *this;
  }
};
}  // namespace opencl
}  // namespace ghost
#endif