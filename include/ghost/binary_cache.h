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

#ifndef GHOST_BINARY_CACHE_H
#define GHOST_BINARY_CACHE_H

#include <stdlib.h>

#include <string>
#include <vector>

namespace ghost {
namespace implementation {
class Device;
}
class Digest;

/// @brief Number of hex characters of the digest to use in cache filenames.
#define GHOST_DIGEST_FILENAME_LENGTH 32

/// @brief Content-addressed cache for compiled GPU binaries.
///
/// Stores and retrieves pre-compiled GPU program binaries keyed by a SHA-256
/// digest of the source data, compiler options, and device identity. Cached
/// files are stored under @c cachePath and are subject to a time-to-live
/// purge policy. Set the cachePath member to enable caching.
class BinaryCache {
 protected:
  /// @brief Remove cached files older than @p days from @p subfolder.
  /// @param subfolder Subdirectory within the cache path.
  /// @param days Maximum age in days; files older than this are deleted.
  /// @return @c true if files were successfully purged.
  static bool purgeFiles(const std::string& subfolder, int days);

 public:
  typedef std::string PathString;
  /// @brief Root directory for cached binary files.
  PathString cachePath;

  /// @brief Build a SHA-256 digest key for a compiled program.
  /// @param[in,out] d Digest object to update with the key material.
  /// @param dev The device whose identity is included in the digest.
  /// @param count Device count or index used as part of the key.
  /// @param data Pointer to the source data (e.g., program text).
  /// @param length Length of @p data in bytes.
  /// @param options Compiler options string included in the digest.
  static void makeDigest(Digest& d, const implementation::Device& dev,
                         size_t count, const void* data, size_t length,
                         const std::string& options);

  /// @brief Check whether the binary cache is enabled.
  /// @return @c true if @c cachePath is non-empty.
  bool isEnabled() const;

  /// @brief Remove cached binaries older than @p days for the given device.
  /// @param dev The device whose cache subfolder to purge.
  /// @param days Maximum age in days (default 30).
  void purgeBinaries(const implementation::Device& dev, int days) const;

  /// @brief Load previously cached compiled binaries.
  /// @param[out] binaries Vector of binary blobs, one per device/program.
  /// @param[out] sizes Corresponding sizes of each binary blob.
  /// @param dev The device to look up cached binaries for.
  /// @param data Pointer to the source data used to compute the cache key.
  /// @param length Length of @p data in bytes.
  /// @param options Compiler options used to compute the cache key.
  /// @return @c true if cached binaries were found and loaded.
  bool loadBinaries(std::vector<std::vector<unsigned char>>& binaries,
                    std::vector<size_t>& sizes,
                    const implementation::Device& dev, const void* data,
                    size_t length, const std::string& options) const;

  /// @brief Save compiled binaries to the cache.
  /// @param dev The device the binaries were compiled for.
  /// @param binaries Vector of raw binary pointers.
  /// @param sizes Corresponding sizes of each binary.
  /// @param data Pointer to the source data used to compute the cache key.
  /// @param length Length of @p data in bytes.
  /// @param options Compiler options used to compute the cache key.
  void saveBinaries(const implementation::Device& dev,
                    const std::vector<unsigned char*>& binaries,
                    const std::vector<size_t>& sizes, const void* data,
                    size_t length, const std::string& options) const;
};

}  // namespace ghost

#endif