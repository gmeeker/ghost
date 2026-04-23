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

#include <ghost/exception.h>
#include <ghost/kernel_source.h>

#include <iomanip>
#include <sstream>

namespace ghost {

KernelSource::KernelSource(const std::string& text,
                           const CompilerOptions& baseOptions, bool useDefines)
    : _mode(Mode::Text),
      _forceDefines(useDefines),
      _text(text),
      _baseOptions(baseOptions),
      _baseLibrary(nullptr) {}

KernelSource::KernelSource(const void* data, size_t len,
                           const CompilerOptions& baseOptions)
    : _mode(Mode::Binary),
      _forceDefines(false),
      _binaryData(static_cast<const uint8_t*>(data),
                  static_cast<const uint8_t*>(data) + len),
      _baseOptions(baseOptions),
      _baseLibrary(nullptr) {}

std::string KernelSource::makeKey(
    const std::string& functionName,
    const std::vector<std::pair<std::string, Attribute>>& constants) {
  std::ostringstream key;
  key << functionName;
  for (auto& c : constants) {
    key << ':' << c.first << '=';
    auto& attr = c.second;
    switch (attr.type()) {
      case Attribute::Type_Float:
        key << std::setprecision(9) << attr.asFloat() << 'f';
        break;
      case Attribute::Type_Int:
        key << attr.asInt();
        break;
      case Attribute::Type_UInt:
        key << attr.asUInt() << 'u';
        break;
      case Attribute::Type_Bool:
        key << (attr.asBool() ? '1' : '0');
        break;
      default:
        key << "?";
        break;
    }
  }
  return key.str();
}

static void appendAttrToKey(std::ostringstream& key, const Attribute& attr) {
  switch (attr.type()) {
    case Attribute::Type_Float:
      key << std::setprecision(9) << attr.asFloat() << 'f';
      break;
    case Attribute::Type_Int:
      key << attr.asInt();
      break;
    case Attribute::Type_UInt:
      key << attr.asUInt() << 'u';
      break;
    case Attribute::Type_Bool:
      key << (attr.asBool() ? '1' : '0');
      break;
    default:
      key << "?";
      break;
  }
}

std::string KernelSource::makeKey(const std::string& functionName,
                                  const std::vector<Attribute>& constants) {
  std::ostringstream key;
  key << functionName;
  for (size_t i = 0; i < constants.size(); i++) {
    key << ':' << i << '=';
    appendAttrToKey(key, constants[i]);
  }
  return key.str();
}

void KernelSource::constantsToDefines(
    const std::vector<std::pair<std::string, Attribute>>& constants,
    CompilerOptions& opts) {
  for (auto& c : constants) {
    std::string value;
    auto& attr = c.second;
    switch (attr.type()) {
      case Attribute::Type_Float: {
        std::ostringstream ss;
        ss << std::setprecision(9) << attr.asFloat();
        value = ss.str();
        if (value.find('.') == std::string::npos &&
            value.find('e') == std::string::npos) {
          value += ".0";
        }
        value += "f";
        break;
      }
      case Attribute::Type_Int:
        value = std::to_string(attr.asInt());
        break;
      case Attribute::Type_UInt:
        value = std::to_string(attr.asUInt()) + "u";
        break;
      case Attribute::Type_Bool:
        value = attr.asBool() ? "1" : "0";
        break;
      default:
        break;
    }
    opts.defines.push_back({c.first, value});
  }
}

std::vector<Attribute> KernelSource::constantsToPositional(
    const std::vector<std::pair<std::string, Attribute>>& constants) {
  std::vector<Attribute> result;
  result.reserve(constants.size());
  for (auto& c : constants) {
    result.push_back(c.second);
  }
  return result;
}

// Text mode: OpenCL C, Metal Shading Language, CUDA NVRTC.
//
// Strategy selection (first call only) — chosen from device capability
// attributes so no speculative compile is ever performed:
//   1. kDeviceSupportsProgramConstants (Metal, Vulkan): compile once,
//      specialize per variant via function/spec constants.
//   2. kDeviceSupportsProgramGlobals (CUDA): compile once with
//      retainBinary, reuse the binary per variant via loadLibraryFromData
//      + setGlobals. Avoids repeated source compilation.
//   3. Otherwise (OpenCL, CPU, ...): compile per variant with -D defines.
//
// Strategy 3 is also used when useDefines=true is passed to the constructor.
Function KernelSource::getFunctionFromText(
    Device& device, const std::string& functionName,
    const std::vector<std::pair<std::string, Attribute>>& constants) {
  if (!_capabilityChecked) {
    _capabilityChecked = true;
    if (_forceDefines) {
      // Caller explicitly requested -D defines for all backends.
      _useSpecialization = false;
    } else if (device.getAttribute(kDeviceSupportsProgramConstants).asBool()) {
      // Metal / Vulkan: compile once, specialize per variant.
      _useSpecialization = true;
      _baseLibrary = device.loadLibraryFromText(_text, _baseOptions);
      _hasBaseLibrary = true;
    } else if (device.getAttribute(kDeviceSupportsProgramGlobals).asBool()) {
      // CUDA: compile once, reuse binary per variant with setGlobals.
      Library base = device.loadLibraryFromText(_text, _baseOptions, true);
      _binaryData = base.getBinary();
      _useSpecialization = false;
      _baseLibrary = base;
      _hasBaseLibrary = true;
    } else {
      // OpenCL / other: compile per variant with -D defines.
      _useSpecialization = false;
    }
  }

  // Path 1: specialization (Metal, Vulkan).
  if (_useSpecialization && _hasBaseLibrary) {
    if (constants.empty()) {
      return _baseLibrary.lookupFunction(functionName);
    }
    try {
      return _baseLibrary.lookupSpecializedFunction(functionName, constants);
    } catch (const ghost::unsupported_error&) {
      // Named not supported; try positional (Vulkan).
    }
    try {
      const std::vector<Attribute> positional =
          constantsToPositional(constants);
      return _baseLibrary.lookupSpecializedFunction(functionName, positional);
    } catch (const ghost::unsupported_error&) {
      _useSpecialization = false;
    }
  }

  // Path 2: binary reuse + setGlobals (CUDA).
  if (!_binaryData.empty()) {
    if (constants.empty() && _hasBaseLibrary) {
      return _baseLibrary.lookupFunction(functionName);
    }
    Library lib = device.loadLibraryFromData(_binaryData.data(),
                                             _binaryData.size(), _baseOptions);
    if (!constants.empty()) {
      lib.setGlobals(constants);
    }
    return lib.lookupFunction(functionName);
  }

  // Path 3: compile per variant with -D defines (OpenCL, forced).
  CompilerOptions opts = _baseOptions;
  constantsToDefines(constants, opts);
  Library lib = device.loadLibraryFromText(_text, opts);
  return lib.lookupFunction(functionName);
}

// Binary mode: Metal metallib, CUDA fatbin/PTX, Vulkan SPIR-V.
Function KernelSource::getFunctionFromBinary(
    Device& device, const std::string& functionName,
    const std::vector<std::pair<std::string, Attribute>>& constants) {
  if (!_capabilityChecked) {
    _capabilityChecked = true;
    _useSpecialization =
        device.getAttribute(kDeviceSupportsProgramConstants).asBool();
    if (_useSpecialization) {
      // Metal metallib / Vulkan SPIR-V: load once, specialize per
      // variant.
      _baseLibrary = device.loadLibraryFromData(
          _binaryData.data(), _binaryData.size(), _baseOptions);
      _hasBaseLibrary = true;
    }
  }

  if (_useSpecialization && _hasBaseLibrary) {
    if (constants.empty()) {
      return _baseLibrary.lookupFunction(functionName);
    }
    // Try named specialization (Metal).
    try {
      return _baseLibrary.lookupSpecializedFunction(functionName, constants);
    } catch (const ghost::unsupported_error&) {
      // Not supported as named; try positional (Vulkan).
    }
    // Positional specialization (Vulkan): names are stripped, values
    // passed in declaration order.
    try {
      const std::vector<Attribute> positional =
          constantsToPositional(constants);
      return _baseLibrary.lookupSpecializedFunction(functionName, positional);
    } catch (const ghost::unsupported_error&) {
      // Shouldn't happen if kDeviceSupportsProgramConstants is true,
      // but fall through to setGlobals path just in case.
      _useSpecialization = false;
    }
  }

  // CUDA fatbin / other: load a separate library per variant, then
  // set __constant__ globals. Each variant gets its own CUmodule so
  // concurrent dispatch with different constants is safe.
  Library lib = device.loadLibraryFromData(_binaryData.data(),
                                           _binaryData.size(), _baseOptions);
  if (!constants.empty()) {
    lib.setGlobals(constants);
  }
  return lib.lookupFunction(functionName);
}

Function KernelSource::getFunction(
    Device& device, const std::string& functionName,
    const std::vector<std::pair<std::string, Attribute>>& constants) {
  std::string key = makeKey(functionName, constants);

  // Fast path: cache hit under shared lock.
  {
    std::shared_lock<std::shared_mutex> lock(_mutex);
    auto it = _cache.find(key);
    if (it != _cache.end()) {
      return it->second;
    }
  }

  // Slow path: compile under exclusive lock.
  std::unique_lock<std::shared_mutex> lock(_mutex);

  // Double-check after acquiring exclusive lock.
  auto it = _cache.find(key);
  if (it != _cache.end()) {
    return it->second;
  }

  Function fn = (_mode == Mode::Text)
                    ? getFunctionFromText(device, functionName, constants)
                    : getFunctionFromBinary(device, functionName, constants);

  _cache.emplace(key, fn);
  return fn;
}

Function KernelSource::getSpecializedFunction(
    Device& device, const std::string& functionName,
    const std::vector<Attribute>& constants) {
  std::string key = makeKey(functionName, constants);

  // Fast path: cache hit under shared lock.
  {
    std::shared_lock<std::shared_mutex> lock(_mutex);
    auto it = _cache.find(key);
    if (it != _cache.end()) {
      return it->second;
    }
  }

  // Slow path: compile under exclusive lock.
  std::unique_lock<std::shared_mutex> lock(_mutex);

  auto it = _cache.find(key);
  if (it != _cache.end()) {
    return it->second;
  }

  // Ensure base library exists. Positional constants only make sense on
  // specialization backends (Metal, Vulkan).
  if (!_capabilityChecked) {
    _capabilityChecked = true;
    _useSpecialization =
        device.getAttribute(kDeviceSupportsProgramConstants).asBool();
    if (_useSpecialization) {
      if (_mode == Mode::Text) {
        _baseLibrary = device.loadLibraryFromText(_text, _baseOptions);
      } else {
        _baseLibrary = device.loadLibraryFromData(
            _binaryData.data(), _binaryData.size(), _baseOptions);
      }
      _hasBaseLibrary = true;
    }
  }

  if (!_hasBaseLibrary) {
    throw ghost::unsupported_error();
  }

  Function fn(nullptr);
  if (constants.empty()) {
    fn = _baseLibrary.lookupFunction(functionName);
  } else {
    fn = _baseLibrary.lookupSpecializedFunction(functionName, constants);
  }

  _cache.emplace(key, fn);
  return fn;
}

}  // namespace ghost
