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

/// @brief RAII wrapper around a C @c FILE* handle.
///
/// Automatically closes the file on destruction. Not copyable, but supports
/// assignment from a raw @c FILE* pointer (closes any previously held file).
/// Used internally for binary cache I/O.
class FileWrapper {
 protected:
  /// @brief Underlying C file handle, or @c nullptr if not open.
  FILE* _fp;

 public:
  /// @brief Construct a wrapper with no open file.
  FileWrapper() : _fp(nullptr) {}

  FileWrapper(const FileWrapper&) = delete;

  /// @brief Construct a wrapper taking ownership of @p fp.
  /// @param fp An open file handle, or @c nullptr.
  explicit FileWrapper(FILE* fp) : _fp(fp) {}

  /// @brief Close the file if open.
  ~FileWrapper() { close(); }

  /// @brief Close the file handle if open, setting it to @c nullptr.
  void close() {
    if (_fp) {
      fclose(_fp);
      _fp = nullptr;
    }
  }

  /// @brief Read exactly @p len bytes from the file.
  /// @param[out] p Destination buffer.
  /// @param len Number of bytes to read.
  /// @throws std::runtime_error if the file is not open or the read fails.
  void read(void* p, size_t len) {
    if (_fp == nullptr || fread(p, len, 1, _fp) != 1)
      throw std::runtime_error("read error");
  }

  /// @brief Write exactly @p len bytes to the file.
  /// @param p Source buffer.
  /// @param len Number of bytes to write.
  /// @throws std::runtime_error if the file is not open or the write fails.
  void write(const void* p, size_t len) {
    if (_fp == nullptr || fwrite(p, len, 1, _fp) != 1)
      throw std::runtime_error("write error");
  }

  /// @brief Check whether a file handle is open.
  /// @return @c true if the file is open.
  bool okay() const { return _fp != nullptr; }

  FileWrapper& operator=(const FileWrapper&) = delete;

  /// @brief Assign a new file handle, closing any previously held file.
  /// @param f The new file handle to take ownership of.
  /// @return Reference to this wrapper.
  FileWrapper& operator=(FILE* f) {
    close();
    _fp = f;
    return *this;
  }
};
}  // namespace ghost

#endif