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

#include <ghost/function.h>

#include <memory>
#include <string>

namespace ghost {
Function::Function(std::shared_ptr<implementation::Function> impl)
    : _impl(impl) {}

Library::Library(std::shared_ptr<implementation::Library> impl) : _impl(impl) {}

Function Library::lookupFunction(const std::string& name) const {
  return _impl->lookupFunction(name);
}
}  // namespace ghost
