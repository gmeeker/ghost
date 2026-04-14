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

#include <ghost/exception.h>

namespace ghost {
namespace detail {

namespace {
thread_local std::exception_ptr t_deferred;
}

void stashError(std::exception_ptr e) noexcept {
  if (!e) return;
  if (!t_deferred) t_deferred = e;
}

void drainErrors() {
  if (t_deferred) {
    std::exception_ptr e = t_deferred;
    t_deferred = nullptr;
    std::rethrow_exception(e);
  }
}

}  // namespace detail
}  // namespace ghost
