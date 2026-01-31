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
#include <ghost/io.h>
#include <time.h>

#if WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <string>
#include <vector>

namespace ghost {
bool BinaryCache::isEnabled() const { return !cachePath.empty(); }

void BinaryCache::makeDigest(Digest& d, const implementation::Device& dev,
                             size_t count, const void* data, size_t length,
                             const std::string& options) {
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
  if (!options.empty()) d.update(options.c_str(), options.size());
  if (data) d.update(data, length);
}

bool BinaryCache::purgeFiles(const std::string& dirname, int days) {
  time_t oldest = time(NULL);
  oldest -= 60 * 60 * 24 * days;  // Remove anything older than 'days'.
#if WIN32
  std::basic_string<TCHAR> name;
  if (!getFileName(name, "", subfolder)) return false;
  HANDLE hFind = INVALID_HANDLE_VALUE;
  WIN32_FIND_DATA finddata;
  hFind = FindFirstFile((name + _T("\\*")).c_str(), &finddata);
  while (hFind != INVALID_HANDLE_VALUE) {
    std::basic_string<TCHAR> filename = name + _T("\\") + finddata.cFileName;
    DALStructStat s;
    if (DALStat(filename.c_str(), &s) == 0) {
      if (s.st_ctime < oldest) {
        DeleteFile(filename.c_str());
      }
    }
    if (!FindNextFile(hFind, &finddata)) {
      FindClose(hFind);
      hFind = INVALID_HANDLE_VALUE;
    }
  }
  if (hFind != INVALID_HANDLE_VALUE) FindClose(hFind);
  return true;
#else
  DIR* dir = ::opendir(dirname.c_str());
  if (dir) {
    struct dirent* d;
    struct stat s;
    while ((d = ::readdir(dir))) {
      std::string path = dirname + "/" + d->d_name;
      if (::stat(path.c_str(), &s) >= 0) {
        if (s.st_ctime < oldest) {
          ::unlink(path.c_str());
        }
      }
    }
    ::closedir(dir);
  }
  return true;
#endif
}

void BinaryCache::purgeBinaries(const implementation::Device& dev,
                                int days) const {
  if (cachePath.empty()) return;
  purgeFiles(cachePath, days);
}

bool BinaryCache::loadBinaries(
    std::vector<std::vector<unsigned char>>& binaries,
    std::vector<size_t>& sizes, const implementation::Device& dev,
    const void* data, size_t length, const std::string& options) const {
  if (cachePath.empty()) return false;
  size_t count = (size_t)dev.getAttribute(kDeviceCount).asInt();
  Digest f, d;
  makeDigest(d, dev, count, nullptr, 0, "");
  makeDigest(f, dev, count, data, length, options);
  FileWrapper file;
  file = fopen((cachePath + f.get()).c_str(), "rb");
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
                               const std::string& options) const {
  if (cachePath.empty()) return;
  Digest f, d;
  makeDigest(d, dev, binaries.size(), nullptr, 0, "");
  makeDigest(f, dev, binaries.size(), data, length, options);
  size_t i;

  FileWrapper file;
  file = fopen((cachePath + f.get()).c_str(), "wb");
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
