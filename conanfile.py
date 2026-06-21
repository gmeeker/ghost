import os

from conan import ConanFile
from conan.tools.apple import is_apple_os
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.scm import Version

required_conan_version = ">=1.54.0"


class GhostConan(ConanFile):
    name = "ghost"
    version = "2.0"
    license = "BSD-3-Clause"
    author = "Digital Anarchy"
    description = "Compute engine"
    topics = ("GPU", "computing")

    exports_sources = "LICENSE", "conanfile.py", "CMakeLists.txt", "cmake/**", "src/**", "include/**", "test/**", "test_package"
    settings = "os", "arch", "compiler", "build_type", "cuda"
    options = {"shared": [True, False],
               "fPIC": [True, False],
               "with_cuda": [True, False],
               "with_cuda_link": [False, True, "delaylib"],
               "with_cuda_nvrtc": [True, False],
               "static_nvrtc": [True, False],
               "with_directx": [True, False],
               "with_gcd": [True, False],
               "with_metal": [True, False],
               "with_opencl": [True, False],
               "with_opencl_command_buffers": [True, False],
               "with_vulkan": [True, False],
               "with_pocl_tests": [True, False]}
    default_options = {
               "shared": False,
               "fPIC": True,
               "with_cuda": True,
               "with_cuda_link": False,
               "with_cuda_nvrtc": False,
               "static_nvrtc": False,
               "with_directx": False,
               "with_gcd": False,
               "with_metal": True,
               "with_opencl": True,
               "with_opencl_command_buffers": True,
               "with_vulkan": False,
               "with_pocl_tests": False}
    short_paths = True

    # pocl is not on ConanCenter (it builds against LLVM), so the ICD is supplied
    # out-of-band (conda-forge / distro / CI build). with_pocl_tests just adds a
    # CTest target that runs the OpenCL tests against pocl and asserts this exact
    # runtime version, so the command-buffer extension level is deterministic.
    _pocl_version = "7.0"

    def validate(self):
        check_min_cppstd(self, "17")

    def _supports_cuda(self):
        return self.settings.os in ("Windows", "Linux")

    def _supports_directx(self):
        return self.settings.os in ("Windows",)

    def _supports_opencl(self):
        return self.settings.os in ("Windows", "Linux", "Macos")

    def _cuda_version(self):
        version = self.settings.get_safe("cuda.version")
        return Version(version) if version else None

    @property
    def cuda_toolkit_path(self):
        if self.settings.os == "Windows":
            cuda_toolkit = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v' + str(self.settings.cuda.version)
        else:
            cuda_toolkit = '/usr/local/cuda-' + str(self.settings.cuda.version)
        if os.path.exists(cuda_toolkit):
            return cuda_toolkit
        else:
            return None

    def configure(self):
        if not self._supports_cuda() or not self.options.get_safe("with_cuda", False):
            self.settings.rm_safe("cuda")
        # delaylib is MSVC-specific; drop on other platforms.
        if (self.settings.os != "Windows"
                and self.options.get_safe("with_cuda_link") == "delaylib"):
            self.options.rm_safe("with_cuda_link")
        # CUDA 12+ ships cuda.lib as an import wrapper, so /DELAYLOAD against
        # nvcuda.dll no longer resolves the way it did with the old static
        # cuda.lib. Fall back to plain linking (the closest equivalent, since
        # our own wrapper doesn't cover every CUDA entry point) and warn.
        cuda_version = self._cuda_version()
        if (self.options.get_safe("with_cuda_link") == "delaylib"
                and cuda_version is not None
                and cuda_version >= "12"):
            self.output.warning(
                "with_cuda_link=delaylib is not supported with CUDA "
                + str(self.settings.cuda.version)
                + " (cuda.lib is now an import wrapper and delay loading "
                "nvcuda.dll no longer works); switching to with_cuda_link=True.")
            self.options.with_cuda_link = True

    def config_options(self):
        if self.settings.os == 'Windows':
            self.options.rm_safe("fPIC")
        if not (self._supports_cuda() and self.settings.cuda):
            self.options.rm_safe("with_cuda")
            self.options.rm_safe("with_cuda_link")
            self.options.rm_safe("with_cuda_nvrtc")
        if not self._supports_directx():
            self.options.rm_safe("with_directx")
        if not is_apple_os(self):
            self.options.rm_safe("with_gcd")
            self.options.rm_safe("with_metal")
        else:
            # Apple's OpenCL is frozen at 1.2 and deprecated; cl_khr_command_buffer
            # is unavailable, so the acceleration can't be built there.
            self.options.rm_safe("with_opencl_command_buffers")
        if not self._supports_opencl():
            self.options.rm_safe("with_opencl")
            self.options.rm_safe("with_pocl_tests")
            self.options.rm_safe("with_opencl_command_buffers")

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        if self.options.get_safe("with_opencl", False):
            if not is_apple_os(self):
                self.requires("opencl-headers/2025.07.22")
                self.requires("opencl-icd-loader/2025.07.22")
        if self.options.get_safe("with_vulkan", False):
            if is_apple_os(self):
                self.requires("moltenvk/1.2.2")
            else:
                self.requires("vulkan-headers/1.4.313.0")
                self.requires("vulkan-loader/1.4.313.0")

    def build_requirements(self):
        self.test_requires("gtest/[>=1.14.0 <2]")
        self.tool_requires("spirv-tools/[>=1.4.313.0 <2]")

    def generate(self):
        tc = CMakeToolchain(self)

        if self.options.get_safe("with_cuda", False):
            tc.variables['WITH_CUDA'] = 'ON'
            if self.options.get_safe("with_cuda_link"):
                tc.variables['WITH_CUDA_LINK'] = 'ON'
                if self.options.with_cuda_link == "delaylib":
                    tc.variables['WITH_CUDA_DELAYLOAD'] = 'ON'
            if self.options.get_safe("with_cuda_nvrtc", False):
                tc.variables['WITH_CUDA_NVRTC'] = 'ON'
            if self.options.get_safe("static_nvrtc", False):
                tc.variables['WITH_CUDA_NVRTC_STATIC'] = 'ON'
            if self.settings.cuda:
                cuda_toolkit = self.cuda_toolkit_path
                if cuda_toolkit:
                    tc.variables['CUDAToolkit_ROOT'] = cuda_toolkit.replace('\\', '/')
                    include_path = os.path.join(cuda_toolkit, 'include')
                    if os.path.exists(include_path):
                        tc.variables['WITH_CUDA_NVRTC_INCLUDE_PATH'] = include_path.replace('\\', '/')
                        for prefix in [[], ['cccl']]:
                            libcxx_path = os.path.join(*([include_path] + prefix + ['cuda', 'std', 'detail', 'libcxx', 'include']))
                            if os.path.exists(libcxx_path):
                                tc.variables['WITH_CUDA_NVRTC_STD_INCLUDE_PATH'] = libcxx_path.replace('\\', '/')
                                break
        if self.options.get_safe("with_directx", False):
            tc.variables['WITH_DIRECTX'] = 'ON'
        if self.options.get_safe("with_gcd", False):
            tc.variables['WITH_GCD'] = 'ON'
        tc.variables['WITH_OPENCL'] = 'ON' if self.options.get_safe("with_opencl", False) else 'OFF'
        tc.variables['WITH_OPENCL_COMMAND_BUFFERS'] = 'ON' if self.options.get_safe("with_opencl_command_buffers", False) else 'OFF'
        if self.options.get_safe("with_pocl_tests", False):
            tc.variables['GHOST_POCL_TESTS'] = 'ON'
            tc.variables['GHOST_POCL_VERSION'] = self._pocl_version
        if self.options.get_safe("with_metal", False):
            tc.variables['WITH_METAL'] = 'ON'
        if self.options.get_safe("with_vulkan", False):
            if is_apple_os(self):
                tc.variables['WITH_VULKAN'] = self.dependencies["moltenvk"].package_folder.replace('\\', '/')
            else:
                tc.variables['WITH_VULKAN'] = self.dependencies["vulkan-headers"].package_folder.replace('\\', '/')
        tc.generate()
        cd = CMakeDeps(self)
        cd.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.set_property("cmake_find_mode", "both")
        self.cpp_info.set_property("cmake_module_file_name", "Ghost")
        self.cpp_info.set_property("cmake_file_name", "Ghost")
        self.cpp_info.set_property("cmake_target_name", "Ghost::Ghost")
        self.cpp_info.set_property("pkg_config_name", "ghost")
        self.cpp_info.libs = ["Ghost"]
        if (self.options.get_safe("with_cuda_link")
                and not self.options.shared
                and self.settings.os == "Windows"):
            self.cpp_info.libs.append("cuda")
            cuda_toolkit = self.cuda_toolkit_path
            if cuda_toolkit:
                self.cpp_info.libdirs.append(
                    os.path.join(cuda_toolkit, "lib", "x64"))
            if self.options.with_cuda_link == "delaylib":
                self.cpp_info.system_libs.append("delayimp")
                self.cpp_info.sharedlinkflags.append("/DELAYLOAD:nvcuda.dll")
                self.cpp_info.exelinkflags.append("/DELAYLOAD:nvcuda.dll")
        if self.options.get_safe("with_cuda_nvrtc", False) and not self.options.shared:
            if self.options.get_safe("static_nvrtc", False):
                self.cpp_info.libs += ['nvrtc_static', 'nvrtc-builtins_static', 'nvptxcompiler_static']
            else:
                self.cpp_info.libs += ['nvrtc']
            cuda_toolkit = self.cuda_toolkit_path
            if cuda_toolkit:
                if self.settings.os == "Windows":
                    libpath = os.path.join(cuda_toolkit, 'lib', 'x64')
                else:
                    libpath = os.path.join(cuda_toolkit, 'lib64')
                self.cpp_info.libdirs += [libpath]
        if self.options.get_safe("with_opencl", False):
            self.cpp_info.frameworks.append('OpenCL')
        if self.options.get_safe("with_metal", False):
            self.cpp_info.frameworks += ['Metal', 'Foundation']
