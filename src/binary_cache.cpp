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

#include <ghost/binary_cache.h>
#include <ghost/device.h>
#include <ghost/digest.h>
#include <ghost/implementation/impl_function.h>
#include <ghost/io.h>
#include <string.h>
#include <time.h>

#include <chrono>
#include <filesystem>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace ghost {

namespace fs = std::filesystem;

std::string CompilerOptions::buildFlags() const {
  std::ostringstream result;
  result << flags;
  for (auto& arg : arguments) {
    if (result.tellp() > 0) result << ' ';
    result << arg;
  }
  for (auto& def : defines) {
    if (result.tellp() > 0) result << ' ';
    result << "-D" << def.first;
    if (!def.second.empty()) result << '=' << def.second;
  }
  return result.str();
}

void CompilerOptions::updateDigest(Digest& d) const {
  static const char sep = '\0';
  if (!flags.empty()) d.update(flags.c_str(), flags.size());
  for (auto& arg : arguments) {
    d.update(&sep, 1);
    d.update(arg.c_str(), arg.size());
  }
  for (auto& def : defines) {
    d.update(&sep, 1);
    d.update(def.first.c_str(), def.first.size());
    d.update("=", 1);
    d.update(def.second.c_str(), def.second.size());
  }
  for (auto& hdr : headers) {
    d.update(&sep, 1);
    d.update(hdr.first.c_str(), hdr.first.size());
    d.update(&sep, 1);
    d.update(hdr.second.c_str(), hdr.second.size());
  }
}

bool BinaryCache::isEnabled() const { return !cachePath.empty(); }

void BinaryCache::makeDigest(Digest& d, const implementation::Device& dev,
                             size_t count, const void* data, size_t length,
                             const CompilerOptions& options) {
  for (size_t i = 0; i < count; i++) {
    std::string vendor = dev.getAttribute(kDeviceVendor).asString();
    std::string name = dev.getAttribute(kDeviceName).asString();
    std::string driverVersion =
        dev.getAttribute(kDeviceDriverVersion).asString();
    d.update(vendor.c_str(), vendor.size());
    d.update(name.c_str(), name.size());
    if (!driverVersion.empty())
      d.update(driverVersion.c_str(), driverVersion.size());
  }
  options.updateDigest(d);
  if (data) d.update(data, length);
}

bool BinaryCache::purgeFiles(const fs::path& dirname, int days) {
  std::error_code ec;
  if (!fs::is_directory(dirname, ec)) return true;
  auto cutoff = fs::file_time_type::clock::now() -
                std::chrono::hours(24 * static_cast<int64_t>(days));
  for (const auto& entry : fs::directory_iterator(dirname, ec)) {
    if (ec) break;
    auto mtime = fs::last_write_time(entry.path(), ec);
    if (ec) {
      ec.clear();
      continue;
    }
    if (mtime < cutoff) {
      fs::remove(entry.path(), ec);
      ec.clear();
    }
  }
  return true;
}

void BinaryCache::purgeBinaries(const implementation::Device& dev,
                                int days) const {
  if (cachePath.empty()) return;
  purgeFiles(cachePath, days);
}

bool BinaryCache::loadBinaries(
    std::vector<std::vector<unsigned char>>& binaries,
    std::vector<size_t>& sizes, const implementation::Device& dev,
    const void* data, size_t length, const CompilerOptions& options) const {
  if (cachePath.empty()) return false;
  size_t count = (size_t)dev.getAttribute(kDeviceCount).asInt();
  Digest f, d;
  makeDigest(d, dev, count, nullptr, 0, CompilerOptions());
  makeDigest(f, dev, count, data, length, options);
  fs::path filePath =
      cachePath / f.get().substr(0, GHOST_DIGEST_FILENAME_LENGTH);
  FileWrapper file;
  file = fopen(filePath.string().c_str(), "rb");
  if (!file.okay()) return false;

  uint8_t digest1[Digest::length];
  uint8_t digest2[Digest::length];
  d.get(digest1);
  file.read(digest2, sizeof(digest2));
  if (memcmp(digest1, digest2, sizeof(digest1)) != 0) return false;
  file.read(digest2, sizeof(digest2));

  uint64_t v;
  size_t i;
  file.read(&v, sizeof(v));
  if (v == 0 || v != count) return false;
  binaries.resize(size_t(v));
  sizes.resize(size_t(v));
  for (i = 0; i < sizes.size(); i++) {
    file.read(&v, sizeof(v));
  }
  for (i = 0; i < sizes.size(); i++) {
    if (sizes[i] > 0) {
      binaries[i].resize(sizes[i]);
      file.read(&binaries[i][0], sizes[i]);
    }
  }

  Digest b;
  for (i = 0; i < sizes.size(); i++) {
    if (!binaries[i].empty()) b.update(&binaries[i][0], binaries[i].size());
  }
  b.get(digest1);
  if (memcmp(digest1, digest2, sizeof(digest1)) != 0) return false;
  return true;
}

void BinaryCache::saveBinaries(const implementation::Device& dev,
                               const std::vector<unsigned char*>& binaries,
                               const std::vector<size_t>& sizes,
                               const void* data, size_t length,
                               const CompilerOptions& options) const {
  if (cachePath.empty()) return;
  Digest f, d;
  makeDigest(d, dev, binaries.size(), nullptr, 0, CompilerOptions());
  makeDigest(f, dev, binaries.size(), data, length, options);
  size_t i;

  fs::path filePath =
      cachePath / f.get().substr(0, GHOST_DIGEST_FILENAME_LENGTH);
  std::error_code ec;
  fs::create_directories(cachePath, ec);
  FileWrapper file;
  file = fopen(filePath.string().c_str(), "wb");
  if (file.okay()) {
    uint8_t digest[Digest::length];
    d.get(digest);
    file.write(digest, sizeof(digest));
    Digest b;
    for (i = 0; i < binaries.size(); i++) {
      b.update(&binaries[i], sizes[i]);
    }
    b.get(digest);
    file.write(digest, sizeof(digest));

    uint64_t v;
    v = (uint64_t)sizes.size();
    file.write(&v, sizeof(v));
    for (i = 0; i < sizes.size(); i++) {
      v = (uint64_t)sizes[i];
      file.write(&v, sizeof(v));
    }
    for (i = 0; i < binaries.size(); i++) {
      file.write(&binaries[i], sizes[i]);
    }
  }
}
}  // namespace ghost
