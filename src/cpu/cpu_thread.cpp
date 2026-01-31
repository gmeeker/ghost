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

#include <ghost/cpu/device.h>
#include <ghost/cpu/impl_device.h>

#if WIN32
#include <windows.h>
#elif __sgi__
#include <sys/sysmp.h>
#elif __linux__
#include <sys/sysinfo.h>
#elif __APPLE_CC__
#include <sys/sysctl.h>
#else
#include <unistd.h>

#include <thread>
#endif

namespace ghost {
namespace implementation {
size_t DeviceCPU::getNumberOfCores() {
#if WIN32
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  return (size_t)sysinfo.dwNumberOfProcessors;
#elif __sgi__
  return (size_t)sysmp(MP_NAPROCS);
#elif __linux__
  return (size_t)get_nprocs();
#elif __APPLE_CC__
  int cpus = 0;
  size_t length = sizeof(cpus);
  int error = sysctlbyname("hw.activecpu", &cpus, &length, NULL, 0);
  if (error < 0) cpus = 1;
  return (size_t)cpus;
#elif defined(_SC_NPROCESSORS_ONLN)
  return (size_t)sysconf(_SC_NPROCESSORS_ONLN);
#else
  auto count = std::thread::hardware_concurrency();
  return count > 0 ? count : 1;
#endif
}
}  // namespace implementation
}  // namespace ghost
