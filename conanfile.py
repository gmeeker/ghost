import os

from conan import ConanFile
from conan.tools.apple import is_apple_os
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout

required_conan_version = ">=1.54.0"


class GhostConan(ConanFile):
    name = "crosscompute"
    version = "1.0"
    license = "Commercial"
    author = "Digital Anarchy"
    description = "Compute engine"
    topics = ("GPU", "computing")

    generators = "CMakeToolchain", "CMakeDeps"
    exports_sources = "conanfile.py", "CMakeLists.txt", "src/**", "include/**", "test_package"
    settings = "os", "arch", "compiler", "build_type", "cuda"
    options = {"shared": [True, False],
               "fPIC": [True, False],
               "with_cuda": [True, False],
               "with_directx": [True, False],
               "with_metal": [True, False],
               "with_opencl": [True, False],
               "with_vulkan": [True, False]}
    default_options = {
               "shared": False,
               "fPIC": True,
               "with_cuda": True,
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

    def configure(self):
        if not self._supports_cuda() or not self.options.get_safe("with_cuda", False):
            self.settings.rm_safe("cuda")

    def config_options(self):
        if self.settings.os == 'Windows':
            self.options.rm_safe("fPIC")
        if not self._supports_cuda():
            self.options.rm_safe("with_cuda")
        elif not self.settings.cuda:
            self.options.rm_safe("with_cuda")
        if not self._supports_directx():
            self.options.rm_safe("with_directx")
        if not is_apple_os(self):
            self.options.rm_safe("with_metal")
        if not self._supports_opencl():
            self.options.rm_safe("with_opencl")

    def layout(self):
        cmake_layout(self, src_folder="src")

    def requirements(self):
        if self.options.get_safe("with_opencl", False):
            self.requires("opencl-headers/2023.04.17")
            self.requires("opencl-icd-loader/2023.04.17")
        if self.options.get_safe("with_vulkan", False):
            if is_apple_os(self):
                self.requires("moltenvk/1.2.2")
            else:
                self.requires("vulkan-headers/1.3.250.0")
                self.requires("vulkan-loader/1.3.243.0")

    def generate(self):
        tc = CMakeToolchain(self)

        if self.options.get_safe("with_cuda", False):
            tc.variables['USE_CUDA'] = 'ON'
            tc.variables['USE_CUDA_DRIVER'] = 'ON'
        if self.options.get_safe("with_directx", False):
            tc.variables['USE_DIRECTX'] = 'ON'
        if self.options.get_safe("with_opencl", False):
            tc.variables['USE_OPENCL'] = 'ON'
        if self.options.get_safe("with_metal", False):
            tc.variables['USE_METAL'] = 'ON'
        if self.options.get_safe("with_vulkan", False):
            if self.settings.os == "Macos":
                tc.variables['USE_VULKAN'] = self.deps_cpp_info["moltenvk"].rootpath
            else:
                tc.variables['USE_VULKAN'] = self.deps_cpp_info["vulkan-headers"].rootpath
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
