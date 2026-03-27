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

#include <ghost/digest.h>
#include <stdio.h>

#include "sha256.h"

namespace ghost {
Digest::Digest() {
  data = new sha256();
  sha256_init(reinterpret_cast<sha256*>(data));
}

Digest::~Digest() { delete reinterpret_cast<sha256*>(data); }

void Digest::update(const void* buf, size_t len) {
  sha256_append(reinterpret_cast<sha256*>(data), buf, len);
}

void Digest::get(uint8_t digest[length]) {
  sha256_finalize_bytes(reinterpret_cast<sha256*>(data), digest);
}

std::string Digest::get() {
  char hex[SHA256_HEX_SIZE];
  sha256_finalize_hex(reinterpret_cast<sha256*>(data), hex);
  return std::string(hex);
}
}  // namespace ghost
