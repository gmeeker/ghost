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

#ifndef GHOST_OBJC_PTR_H
#define GHOST_OBJC_PTR_H

namespace ghost {
namespace objc {
template <typename T> class ptr {
public:
  typedef T obj_type;

protected:
  obj_type object_;

  void retain() {
#if !__has_feature(objc_arc)
    if (object_)
      [object_ retain];
#endif
  }

public:
  explicit ptr(obj_type obj = nil, bool retainObject = true) : object_(obj) {
    if (!retainObject)
      retain();
  }
  ptr(const ptr &rhs) : object_(rhs.object_) { retain(); }
  ptr(ptr &&rhs) : object_(rhs.object_) { rhs.object_ = nil; }
  ~ptr() { reset(); }

  void reset() {
    if (object_) {
#if !__has_feature(objc_arc)
      [object_ release];
#endif
      object_ = nil;
    }
  }

  obj_type release() {
    obj_type obj = object_;
    object_ = nil;
    return obj;
  }

  obj_type get() const { return object_; }
  operator obj_type() const { return object_; }

  ptr &operator=(obj_type rhs) {
    reset();
    object_ = rhs;
    retain();
    return *this;
  }
  ptr &operator=(const ptr &rhs) {
    if (this != &rhs) {
      reset();
      object_ = rhs.object_;
      retain();
    }
    return *this;
  }
  ptr &operator=(ptr &&rhs) {
    if (this != &rhs) {
      reset();
      object_ = rhs.object_;
      rhs.object_ = nil;
    }
    return *this;
  }
};
} // namespace objc
} // namespace ghost
#endif

// vim: ts=2:sw=2:et:ft=mm
// -*- mode: objective-c++; indent-tabs-mode: nil; tab-width: 2 -*-
// code: language=objective-c++ insertSpaces=true tabSize=2
