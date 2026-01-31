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

#include "sha1.h"

namespace ghost {
Digest::Digest() {
  data = new SHA1_CTX();
  PD_SHA1_Init(reinterpret_cast<SHA1_CTX*>(data));
}

Digest::~Digest() { delete reinterpret_cast<SHA1_CTX*>(data); }

void Digest::update(const void* buf, size_t len) {
  PD_SHA1_Update(reinterpret_cast<SHA1_CTX*>(data),
                 reinterpret_cast<const uint8_t*>(buf), (unsigned)len);
}

void Digest::get(uint8_t digest[length]) {
  PD_SHA1_Final(reinterpret_cast<SHA1_CTX*>(data), digest);
}

std::string Digest::get() {
  uint8_t buf[length];
  PD_SHA1_Final(reinterpret_cast<SHA1_CTX*>(data), buf);
  std::string v;
  for (size_t i = 0; i < length; i++) {
    char buf[4];
    snprintf(buf, sizeof(buf), "%02x", (unsigned int)buf[i]);
    v += buf;
  }
  return v;
}
}  // namespace ghost
