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

#ifndef GHOST_IO_H
#define GHOST_IO_H

#include <stdio.h>

#include <stdexcept>

namespace ghost {
class FileWrapper {
 protected:
  FILE* _fp;

 public:
  FileWrapper() : _fp(nullptr) {}

  FileWrapper(const FileWrapper&) = delete;

  explicit FileWrapper(FILE* fp) : _fp(fp) {}

  ~FileWrapper() { close(); }

  void close() {
    if (_fp) {
      fclose(_fp);
      _fp = nullptr;
    }
  }

  void read(void* p, size_t len) {
    if (_fp == nullptr || fread(p, len, 1, _fp) != 1)
      throw std::runtime_error("read error");
  }

  void write(const void* p, size_t len) {
    if (_fp == nullptr || fwrite(p, len, 1, _fp) != 1)
      throw std::runtime_error("write error");
  }

  bool okay() const { return _fp != nullptr; }

  FileWrapper& operator=(const FileWrapper&) = delete;

  FileWrapper& operator=(FILE* f) {
    close();
    _fp = f;
    return *this;
  }
};
}  // namespace ghost

#endif