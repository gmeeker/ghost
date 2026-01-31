cmake_minimum_required(VERSION 3.5)
project(Ghost VERSION 0.1 LANGUAGES C CXX OBJC OBJCXX)

include(CTest)

option(WITH_CUDA "Build support for CUDA" OFF)
option(WITH_CUDA_LINK "Link directly to CUDA driver" OFF)
option(WITH_DIRECTX "Build support for DirectX" OFF)
option(WITH_METAL "Build support for Metal" OFF)
option(WITH_ARC "Build with Objective-C ARC" ON)
option(WITH_OPENCL "Build support for OpenCL" ON)
option(WITH_VULKAN "Build support for Vulkan" OFF)

if(WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()
if(WITH_OPENCL)
  find_package(OpenCL REQUIRED)
  # find_package(OpenCLHeadersCpp REQUIRED CONFIG)
endif()
if(WITH_VULKAN)
  find_package(Vulkan REQUIRED)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
if(WITH_CUDA AND CUDAToolkit_FOUND)
  add_definitions(-DWITH_CUDA)
  if(WITH_CUDA_LINK)
    add_definitions(-DWITH_CUDA_LINK)
  endif()
  include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()
if(WITH_DIRECTX AND WIN32)
  add_definitions(-DWITH_DIRECTX)
endif()
if(WITH_METAL AND APPLE)
  add_definitions(-DWITH_METAL)
endif()
if(WITH_OPENCL AND OpenCL_FOUND)
  add_definitions(-DWITH_OPENCL)
  include_directories(${OPENCL_INCLUDE_DIRS})
endif()
if(WITH_VULKAN AND Vulkan_FOUND)
  add_definitions(-DWITH_VULKAN)
  include_directories(${VULKAN_INCLUDE_DIRS})
endif()
if(WITH_ARC AND APPLE)
  add_compile_options(-fobjc-arc)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GHOST_PUBLIC_HEADERS
    include/ghost/attribute.h
    include/ghost/binary_cache.h
    include/ghost/device.h
    include/ghost/function.h
    include/ghost/io.h
    include/ghost/cpu/device.h
    include/ghost/cuda/cu_ptr.h
    include/ghost/cuda/device.h
    include/ghost/cuda/exception.h
    include/ghost/cuda/impl_device.h
    include/ghost/cuda/impl_function.h
    include/ghost/directx/device.h
    include/ghost/metal/device.h
    include/ghost/metal/exception.h
    include/ghost/metal/impl_device.h
    include/ghost/metal/impl_function.h
    include/ghost/objc/ptr.h
    include/ghost/opencl/device.h
    include/ghost/opencl/exception.h
    include/ghost/opencl/impl_device.h
    include/ghost/opencl/impl_function.h
    include/ghost/opencl/ptr.h
    include/ghost/vulkan/device.h
    include/ghost/implementation/impl_device.h
    include/ghost/implementation/impl_function.h
  )

set(GHOST_SOURCES
    src/binary_cache.cpp
    src/device.cpp
    src/digest.cpp
    src/function.cpp
    src/sha1.c
    src/cpu/cpu_device.cpp
    src/cpu/cpu_function.cpp
    src/cpu/cpu_thread.cpp
    src/cuda/cuda_device.cpp
    src/cuda/cuda_exception.cpp
    src/cuda/cuda_function.cpp
    src/cuda/cuda_wrapper.cpp
    src/directx/directx_device.cpp
    src/opencl/opencl_device.cpp
    src/opencl/opencl_exception.cpp
    src/opencl/opencl_function.cpp
    src/metal/metal_device.mm
    src/metal/metal_function.mm
    src/vulkan/vulkan_device.cpp
  )

add_library(${PROJECT_NAME}
    "${GHOST_SOURCES}"
  )
if(WITH_OPENCL AND OpenCL_FOUND)
  # target_link_libraries(${PROJECT_NAME} PUBLIC OpenCL::OpenCL OpenCL::HeadersCpp)
  target_link_libraries(${PROJECT_NAME} PUBLIC OpenCL::OpenCL)
endif()
if(WITH_CUDA AND WITH_CUDA_LINK)
  target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cuda_driver)
endif()
if(WITH_VULKAN)
  target_link_libraries(${PROJECT_NAME} PUBLIC Vulkan::Vulkan)
endif()
if(WITH_METAL AND APPLE)
  target_link_libraries(${PROJECT_NAME} INTERFACE "-framework Metal" "-framework Foundation")
endif()

add_executable(TestGhost test/TestGhost.cpp)
target_link_libraries(TestGhost ${PROJECT_NAME})
add_test(NAME TestGhost
         COMMAND TestGhost)

install(TARGETS ${PROJECT_NAME}
    EXPORT GhostTargets
    RUNTIME DESTINATION bin   # For Windows DLLs
    LIBRARY DESTINATION lib   # For shared libraries (.so, .dylib)
    ARCHIVE DESTINATION lib   # For static libraries (.a, .lib)
)

# install(FILES ${GHOST_PUBLIC_HEADERS}
#     DESTINATION include
# )
install(DIRECTORY include/ghost DESTINATION include)

install(EXPORT GhostTargets
        FILE GhostTargets.cmake
        NAMESPACE Ghost::
        DESTINATION lib/cmake
)
include(CMakePackageConfigHelpers)
# generate the config file that includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/GhostConfig.cmake"
  INSTALL_DESTINATION "lib/cmake"
#  NO_SET_AND_CHECK_MACRO
#  NO_CHECK_REQUIRED_COMPONENTS_MACRO
  )
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/GhostConfigVersion.cmake"
  VERSION "${Ghost_VERSION_MAJOR}.${Ghost_VERSION_MINOR}"
  COMPATIBILITY AnyNewerVersion
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/GhostConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/GhostConfigVersion.cmake
  DESTINATION lib/cmake
  )

# Set usage requirements for consumers within the build tree
target_include_directories(${PROJECT_NAME}
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
