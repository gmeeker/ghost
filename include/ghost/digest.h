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

#ifndef GHOST_DIGEST_H
#define GHOST_DIGEST_H

#include <stdint.h>
#include <stdlib.h>

#include <string>

namespace ghost {
class Digest {
 protected:
  void* data;

 public:
  static constexpr size_t length = 20;

  Digest();
  Digest(const Digest&) = delete;
  ~Digest();

  void update(const void* buf, size_t len);
  void get(uint8_t digest[length]);
  std::string get();

  Digest& operator=(const Digest&) = delete;
};
}  // namespace ghost

#endif