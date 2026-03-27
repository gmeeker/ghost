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

/// @brief SHA-256 message digest for hashing data.
///
/// Used internally by BinaryCache to generate content-addressed keys for
/// compiled GPU binaries. Not copyable.
class Digest {
 protected:
  /// @brief Opaque handle to the underlying hash state.
  void* data;

 public:
  /// @brief SHA-256 digest length in bytes.
  static constexpr size_t length = 32;

  /// @brief Construct a new digest, initializing the hash state.
  Digest();
  Digest(const Digest&) = delete;
  /// @brief Destroy the digest and free the hash state.
  ~Digest();

  /// @brief Feed data into the hash.
  /// @param buf Pointer to the data to hash.
  /// @param len Number of bytes to hash.
  void update(const void* buf, size_t len);

  /// @brief Finalize the hash and write the raw digest.
  /// @param[out] digest Output buffer of at least @c length bytes.
  void get(uint8_t digest[length]);

  /// @brief Finalize the hash and return the digest as a hexadecimal string.
  /// @return A 64-character hex string representing the SHA-256 digest.
  std::string get();

  Digest& operator=(const Digest&) = delete;
};
}  // namespace ghost

#endif