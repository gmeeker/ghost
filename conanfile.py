import os

from conan import ConanFile
from conan.tools.apple import is_apple_os
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout

required_conan_version = ">=1.54.0"


class GhostConan(ConanFile):
    name = "ghost"
    version = "1.0"
    license = "BSD-3-Clause"
    author = "Digital Anarchy"
    description = "Compute engine"
    topics = ("GPU", "computing")

    exports_sources = "LICENSE", "conanfile.py", "CMakeLists.txt", "cmake/**", "src/**", "include/**", "test/**", "test_package"
    settings = "os", "arch", "compiler", "build_type", "cuda"
    options = {"shared": [True, False],
               "fPIC": [True, False],
               "with_cuda": [True, False],
               "with_cuda_nvrtc": [True, False],
               "static_nvrtc": [True, False],
               "with_directx": [True, False],
               "with_metal": [True, False],
               "with_opencl": [True, False],
               "with_vulkan": [True, False]}
    default_options = {
               "shared": False,
               "fPIC": True,
               "with_cuda": True,
               "with_cuda_nvrtc": False,
               "static_nvrtc": False,
               "with_directx": False,
               "with_metal": True,
               "with_opencl": True,
               "with_vulkan": False}
    short_paths = True

    def validate(self):
        check_min_cppstd(self, "17")

    def _supports_cuda(self):
        return self.settings.os in ("Windows", "Linux")

    def _supports_directx(self):
        return self.settings.os in ("Windows",)

    def _supports_opencl(self):
        return self.settings.os in ("Windows", "Linux", "Macos")

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

    def config_options(self):
        if self.settings.os == 'Windows':
            self.options.rm_safe("fPIC")
        if not (self._supports_cuda() and self.settings.cuda):
            self.options.rm_safe("with_cuda")
            self.options.rm_safe("with_cuda_nvrtc")
        if not self._supports_directx():
            self.options.rm_safe("with_directx")
        if not is_apple_os(self):
            self.options.rm_safe("with_metal")
        if not self._supports_opencl():
            self.options.rm_safe("with_opencl")

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        if self.options.get_safe("with_opencl", False):
            if not is_apple_os(self):
                self.requires("opencl-headers/2023.12.14")
                self.requires("opencl-icd-loader/2023.12.14")
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
            tc.variables['WITH_CUDA_DRIVER'] = 'ON'
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
                    if os.path.exists(os.path.join(include_path, 'cuda', 'std')):
                        tc.variables['WITH_CUDA_NVRTC_STD_INCLUDE_PATH'] = os.path.join(include_path, 'cuda', 'std').replace('\\', '/')
                    elif os.path.exists(os.path.join(include_path, 'cccl', 'cuda', 'std')):
                        tc.variables['WITH_CUDA_NVRTC_STD_INCLUDE_PATH'] = os.path.join(include_path, 'cccl', 'cuda', 'std').replace('\\', '/')
        if self.options.get_safe("with_directx", False):
            tc.variables['WITH_DIRECTX'] = 'ON'
        tc.variables['WITH_OPENCL'] = 'ON' if self.options.get_safe("with_opencl", False) else 'OFF'
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
