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

#if WITH_DIRECTX

#include <ghost/directx/exception.h>

#include <string>

namespace ghost {
namespace dx {

runtime_error::runtime_error(int32_t hr)
    : std::runtime_error(std::string("DirectX error: ") + errorString(hr)),
      _hr(hr) {}

int32_t runtime_error::error() const noexcept { return _hr; }

const char* runtime_error::errorString(int32_t hr) {
  switch (hr) {
    case 0:
      return "S_OK";
    case 1:
      return "S_FALSE";
    // DXGI errors
    case (int32_t)0x887A0001:
      return "DXGI_ERROR_INVALID_CALL";
    case (int32_t)0x887A0002:
      return "DXGI_ERROR_NOT_FOUND";
    case (int32_t)0x887A0004:
      return "DXGI_ERROR_MORE_DATA";
    case (int32_t)0x887A0005:
      return "DXGI_ERROR_UNSUPPORTED";
    case (int32_t)0x887A0006:
      return "DXGI_ERROR_DEVICE_REMOVED";
    case (int32_t)0x887A0007:
      return "DXGI_ERROR_DEVICE_HUNG";
    case (int32_t)0x887A000A:
      return "DXGI_ERROR_DEVICE_RESET";
    case (int32_t)0x887A002B:
      return "DXGI_ERROR_WAS_STILL_DRAWING";
    // Common HRESULT errors
    case (int32_t)0x80004001:
      return "E_NOTIMPL";
    case (int32_t)0x80004002:
      return "E_NOINTERFACE";
    case (int32_t)0x80004003:
      return "E_POINTER";
    case (int32_t)0x80004004:
      return "E_ABORT";
    case (int32_t)0x80004005:
      return "E_FAIL";
    case (int32_t)0x8007000E:
      return "E_OUTOFMEMORY";
    case (int32_t)0x80070057:
      return "E_INVALIDARG";
    default:
      return "Unknown HRESULT error";
  }
}

}  // namespace dx
}  // namespace ghost

#endif
