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

class BinaryCache {
 protected:
  static bool purgeFiles(const std::string& subfolder, int days);

 public:
  typedef std::string PathString;
  PathString cachePath;

  static void makeDigest(Digest& d, const implementation::Device& dev,
                         size_t count, const void* data, size_t length,
                         const std::string& options);

  bool isEnabled() const;
  void purgeBinaries(const implementation::Device& dev, int days) const;
  bool loadBinaries(std::vector<std::vector<unsigned char>>& binaries,
                    std::vector<size_t>& sizes,
                    const implementation::Device& dev, const void* data,
                    size_t length, const std::string& options) const;
  void saveBinaries(const implementation::Device& dev,
                    const std::vector<unsigned char*>& binaries,
                    const std::vector<size_t>& sizes, const void* data,
                    size_t length, const std::string& options) const;
};

}  // namespace ghost

#endif