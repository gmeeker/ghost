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

#ifndef GHOST_IMPL_FUNCTION_H
#define GHOST_IMPL_FUNCTION_H

#include <ghost/attribute.h>
#include <ghost/exception.h>
#include <stdlib.h>

#include <memory>
#include <string>
#include <vector>

namespace ghost {
class Function;
class Library;
class Stream;
class LaunchArgs;

namespace implementation {
class Function {
 public:
  class Arg {};

  Function() {}

  Function(const Function& rhs) = delete;

  virtual ~Function() {}

  Function& operator=(const Function& rhs) = delete;

  virtual void execute(const ghost::Stream& s, const LaunchArgs& launchArgs,
                       const std::vector<Attribute>& args) = 0;

  static void addArgs(std::vector<Attribute>&) {}

  template <typename ARG>
  static void addArgs(std::vector<Attribute>& args, ARG&& head) {
    args.push_back(std::forward<ARG>(head));
  }

  template <typename ARG, typename... ARGS>
  static void addArgs(std::vector<Attribute>& args, ARG&& head,
                      ARGS&&... tail) {
    args.push_back(std::forward<ARG>(head));
    addArgs(args, std::forward<ARGS>(tail)...);
  }

  template <typename... ARGS>
  void operator()(const Stream& s, const LaunchArgs& launchArgs,
                  ARGS&&... tail) {
    std::vector<Attribute> args;
    addArgs(args, std::forward<ARGS>(tail)...);
    execute(s, launchArgs, args);
  }
};

class Library {
 public:
  Library() {}

  Library(const Library& rhs) = delete;

  virtual ~Library() {}

  Library& operator=(const Library& rhs) = delete;

  virtual ghost::Function lookupFunction(const std::string& name) const = 0;
  virtual ghost::Function specializeFunction(
      const std::string& name, const std::vector<Attribute>& args) const;
};
}  // namespace implementation
}  // namespace ghost

#endif