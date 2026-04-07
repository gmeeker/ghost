cmake_minimum_required(VERSION 3.10...4.1)
project(Ghost VERSION 1.0 LANGUAGES C CXX)
if(APPLE)
    enable_language(OBJC OBJCXX)
endif()

include(CTest)

option(WITH_CUDA "Build support for CUDA" OFF)
option(WITH_CUDA_LINK "Link directly to CUDA driver" OFF)
option(WITH_CUDA_NVRTC "Build support for CUDA runtime compiler" OFF)
option(WITH_CUDA_NVRTC_STATIC "Static link CUDA runtime compiler" OFF)
option(WITH_CUDA_NVRTC_INCLUDE_PATH "Hardcoded CUDA runtime compiler include path" "")
option(WITH_CUDA_NVRTC_STD_INCLUDE_PATH "Hardcoded CUDA runtime compiler STD include path" "")
option(WITH_DIRECTX "Build support for DirectX" OFF)
option(WITH_METAL "Build support for Metal" OFF)
option(WITH_ARC "Build with Objective-C ARC" ON)
option(WITH_OPENCL "Build support for OpenCL" ON)
option(WITH_VULKAN "Build support for Vulkan" OFF)

if(WITH_OPENCL)
  find_package(OpenCL REQUIRED)
endif()
if(WITH_VULKAN)
  if(APPLE)
    find_package(moltenvk REQUIRED)
  else()
    find_package(Vulkan REQUIRED)
  endif()
endif()
if(WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
if(WITH_DIRECTX AND WIN32)
  add_compile_definitions(WITH_DIRECTX)
endif()
if(WITH_METAL AND APPLE)
  find_program(XCRUN_EXECUTABLE xcrun)
  if(XCRUN_EXECUTABLE)
    execute_process(
        COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metal -v
        RESULT_VARIABLE METAL_TOOLCHAIN_RESULT
        OUTPUT_QUIET
        ERROR_QUIET)
    if(NOT METAL_TOOLCHAIN_RESULT EQUAL 0)
      message(FATAL_ERROR
          "Metal Toolchain is not installed. "
          "Install it with: xcodebuild -downloadComponent MetalToolchain")
    endif()
  else()
    message(FATAL_ERROR "xcrun not found. Please install Xcode command line tools.")
  endif()
  add_compile_definitions(WITH_METAL)
endif()
if(WITH_OPENCL AND OpenCL_FOUND)
  add_compile_definitions(WITH_OPENCL)
  include_directories(${OPENCL_INCLUDE_DIRS})
endif()
if(WITH_VULKAN)
  if(APPLE AND moltenvk_FOUND)
    add_compile_definitions(WITH_VULKAN)
  elseif(Vulkan_FOUND)
    add_compile_definitions(WITH_VULKAN)
    include_directories(${Vulkan_INCLUDE_DIRS})
  endif()
endif()
if(WITH_ARC AND APPLE)
  add_compile_options(-fobjc-arc)
endif()
if(WITH_CUDA AND CUDAToolkit_FOUND)
  add_compile_definitions(WITH_CUDA)
  if(WITH_CUDA_LINK)
    add_compile_definitions(WITH_CUDA_LINK)
  endif()
  if(WITH_CUDA_NVRTC)
    add_compile_definitions(WITH_CUDA_NVRTC)
    if(WITH_CUDA_NVRTC_INCLUDE_PATH)
      add_compile_definitions(WITH_CUDA_NVRTC_INCLUDE_PATH="${WITH_CUDA_NVRTC_INCLUDE_PATH}")
    endif()
    if(WITH_CUDA_NVRTC_STD_INCLUDE_PATH)
      add_compile_definitions(WITH_CUDA_NVRTC_STD_INCLUDE_PATH="${WITH_CUDA_NVRTC_STD_INCLUDE_PATH}")
    endif()
  endif()
  include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GHOST_PUBLIC_HEADERS
    include/ghost/attribute.h
    include/ghost/binary_cache.h
    include/ghost/device.h
    include/ghost/event.h
    include/ghost/exception.h
    include/ghost/argument_buffer.h
    include/ghost/command_buffer.h
    include/ghost/gpu_info.h
    include/ghost/function.h
    include/ghost/io.h
    include/ghost/cpu/device.h
    include/ghost/cuda/cu_ptr.h
    include/ghost/cuda/device.h
    include/ghost/cuda/exception.h
    include/ghost/cuda/impl_device.h
    include/ghost/cuda/impl_function.h
    include/ghost/directx/device.h
    include/ghost/directx/exception.h
    include/ghost/directx/impl_device.h
    include/ghost/directx/impl_function.h
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
    include/ghost/vulkan/exception.h
    include/ghost/vulkan/impl_device.h
    include/ghost/vulkan/impl_function.h
    include/ghost/implementation/impl_device.h
    include/ghost/implementation/impl_function.h
  )

if(APPLE)
    set(GHOST_OBJC_SOURCES
        src/metal/metal_device.mm
        src/metal/metal_function.mm
      )
else()
    set(GHOST_OBJC_SOURCES )
endif()

set(GHOST_SOURCES
    src/argument_buffer.cpp
    src/binary_cache.cpp
    src/command_buffer.cpp
    src/device.cpp
    src/digest.cpp
    src/function.cpp
    src/image.cpp
    src/sha256.c
    src/cpu/cpu_device.cpp
    src/cpu/cpu_function.cpp
    src/cpu/cpu_thread.cpp
    src/cuda/cuda_device.cpp
    src/cuda/cuda_exception.cpp
    src/cuda/cuda_function.cpp
    src/cuda/cuda_wrapper.cpp
    src/directx/directx_device.cpp
    src/directx/directx_exception.cpp
    src/directx/directx_function.cpp
    src/opencl/opencl_device.cpp
    src/opencl/opencl_exception.cpp
    src/opencl/opencl_function.cpp
    src/vulkan/vulkan_device.cpp
    src/vulkan/vulkan_exception.cpp
    src/vulkan/vulkan_function.cpp
    ${GHOST_OBJC_SOURCES}
  )

add_library(${PROJECT_NAME}
    "${GHOST_SOURCES}"
  )
if(WITH_OPENCL AND OpenCL_FOUND)
  if(APPLE)
    # On macOS, link directly against the system framework.
    target_link_libraries(${PROJECT_NAME} PUBLIC "-framework OpenCL")
  else()
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenCL::OpenCL)
  endif()
endif()

if(WITH_CUDA)
  if(WITH_CUDA_LINK)
    set(GHOST_CUDA_LIBS CUDA::cuda_driver)
  else()
    set(GHOST_CUDA_LIBS)
  endif()
  if(WITH_CUDA_NVRTC)
    if(WITH_CUDA_NVRTC_STATIC)
      set(GHOST_CUDA_NVRTC_LIBS CUDA::nvrtc_static CUDA::nvrtc_builtins_static)
    else()
      set(GHOST_CUDA_NVRTC_LIBS CUDA::nvrtc)
    endif()
  else()
    set(GHOST_CUDA_NVRTC_LIBS)
  endif()
  target_link_libraries(${PROJECT_NAME} PUBLIC ${GHOST_CUDA_LIBS} ${GHOST_CUDA_NVRTC_LIBS})
endif()
if(WITH_DIRECTX AND WIN32)
  target_link_libraries(${PROJECT_NAME} PUBLIC d3d12 dxgi dxguid)
endif()
if(WITH_VULKAN)
  if(APPLE AND moltenvk_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC moltenvk::moltenvk)
  elseif(Vulkan_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC Vulkan::Vulkan)
  endif()
endif()
if(WITH_METAL AND APPLE)
  target_link_libraries(${PROJECT_NAME} INTERFACE "-framework Metal" "-framework Foundation")
endif()

add_executable(TestGhost test/TestGhost.cpp)
target_link_libraries(TestGhost ${PROJECT_NAME})
add_test(NAME TestGhost
         COMMAND TestGhost)

# Google Test based tests
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  set(GTest_FOUND TRUE)
endif()
option(WITH_SPIRV_ASM "Use spirv-as for SPIR-V generation (simpler build)" ON)

if(GTest_FOUND)
  add_executable(TestGhostGTest
    test/test_attribute.cpp
    test/test_binary_cache_access.cpp
    test/test_binary_load.cpp
    test/test_buffer.cpp
    test/test_command_buffer.cpp
    test/test_cpu_kernel.cpp
    test/test_device.cpp
    test/test_image.cpp
    test/test_kernel.cpp
    test/test_stream.cpp
  )
  target_link_libraries(TestGhostGTest ${PROJECT_NAME} GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(TestGhostGTest)

  # Generated headers go here
  set(GHOST_TEST_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/test/generated)
  file(MAKE_DIRECTORY ${GHOST_TEST_GENERATED_DIR})
  target_include_directories(TestGhostGTest PRIVATE ${GHOST_TEST_GENERATED_DIR})

  # -----------------------------------------------------------------------
  # CPU test kernel shared library
  # -----------------------------------------------------------------------
  add_library(ghost_test_kernels SHARED test/kernels/cpu_kernels.cpp)
  target_include_directories(ghost_test_kernels PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(ghost_test_kernels PRIVATE ${PROJECT_NAME})
  set_target_properties(ghost_test_kernels PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test/shaders
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test/shaders)
  add_dependencies(TestGhostGTest ghost_test_kernels)
  target_compile_definitions(TestGhostGTest PRIVATE
      GHOST_TEST_CPU_LIBRARY="$<TARGET_FILE:ghost_test_kernels>")

  # -----------------------------------------------------------------------
  # CUDA PTX compilation (build-time)
  # -----------------------------------------------------------------------
  if(WITH_CUDA AND CUDAToolkit_FOUND)
    if(${CUDAToolkit_VERSION_MAJOR} LESS 11)
      set(CUDA_MINIMUM_ARCH sm_30)
    elif(${CUDAToolkit_VERSION_MAJOR} LESS 12)
      set(CUDA_MINIMUM_ARCH sm_35)
    elif(${CUDAToolkit_VERSION_MAJOR} LESS 13)
      set(CUDA_MINIMUM_ARCH sm_50)
    else()
      set(CUDA_MINIMUM_ARCH sm_75)
    endif()
    find_program(NVCC_EXECUTABLE nvcc HINTS ${CUDAToolkit_BIN_DIR})
    if(NVCC_EXECUTABLE)
      set(CUDA_PTX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/test/kernels/test_kernel.cu)
      set(CUDA_PTX_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test/shaders/test_kernel.ptx)
      set(CUDA_PTX_HEADER ${GHOST_TEST_GENERATED_DIR}/ghost_test_cuda_ptx.h)
      add_custom_command(
          OUTPUT ${CUDA_PTX_OUTPUT}
          COMMAND ${NVCC_EXECUTABLE} -ptx -arch=${CUDA_MINIMUM_ARCH} -o ${CUDA_PTX_OUTPUT} ${CUDA_PTX_SOURCE}
          DEPENDS ${CUDA_PTX_SOURCE}
          COMMENT "Compiling CUDA kernel to PTX")
      add_custom_command(
          OUTPUT ${CUDA_PTX_HEADER}
          COMMAND ${CMAKE_COMMAND}
              -DINPUT_FILE=${CUDA_PTX_OUTPUT}
              -DOUTPUT_FILE=${CUDA_PTX_HEADER}
              -DVAR_NAME=ghost_test_cuda_ptx
              -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2h.cmake
          DEPENDS ${CUDA_PTX_OUTPUT} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2h.cmake
          COMMENT "Generating CUDA PTX header")
      add_custom_target(ghost_test_ptx DEPENDS ${CUDA_PTX_HEADER})
      add_dependencies(TestGhostGTest ghost_test_ptx)
      target_compile_definitions(TestGhostGTest PRIVATE
          GHOST_TEST_CUDA_PTX="ghost_test_cuda_ptx.h")
    endif()
  endif()

  # -----------------------------------------------------------------------
  # Metal metallib compilation (build-time)
  # -----------------------------------------------------------------------
  if(WITH_METAL AND APPLE)
    # XCRUN_EXECUTABLE already found and Metal toolchain validated above
    set(METAL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/test/kernels/test_kernel.metal)
    set(METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/test/shaders/test_kernel.air)
    set(METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/test/shaders/test_kernel.metallib)
    set(METAL_HEADER ${GHOST_TEST_GENERATED_DIR}/ghost_test_metallib.h)
    add_custom_command(
        OUTPUT ${METAL_AIR}
        COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metal -c ${METAL_SOURCE} -o ${METAL_AIR}
        DEPENDS ${METAL_SOURCE}
        COMMENT "Compiling Metal shader to AIR")
    add_custom_command(
        OUTPUT ${METAL_LIB}
        COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
        DEPENDS ${METAL_AIR}
        COMMENT "Linking Metal library")
    add_custom_command(
        OUTPUT ${METAL_HEADER}
        COMMAND ${CMAKE_COMMAND}
            -DINPUT_FILE=${METAL_LIB}
            -DOUTPUT_FILE=${METAL_HEADER}
            -DVAR_NAME=ghost_test_metallib
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2h.cmake
        DEPENDS ${METAL_LIB} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2h.cmake
        COMMENT "Generating Metal metallib header")
    add_custom_target(ghost_test_metallib DEPENDS ${METAL_HEADER})
    add_dependencies(TestGhostGTest ghost_test_metallib)
    target_compile_definitions(TestGhostGTest PRIVATE
        GHOST_TEST_METALLIB="ghost_test_metallib.h")
  endif()

  # -----------------------------------------------------------------------
  # Vulkan SPIR-V compilation (build-time)
  # -----------------------------------------------------------------------
  if(WITH_VULKAN)
    set(VULKAN_SPV_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test/shaders/test_kernel_vulkan.spv)
    set(VULKAN_SPV_HEADER ${GHOST_TEST_GENERATED_DIR}/ghost_test_vulkan_spirv.h)
    if(WITH_SPIRV_ASM)
      find_program(SPIRV_AS_EXECUTABLE spirv-as)
      if(SPIRV_AS_EXECUTABLE)
        set(VULKAN_SPVASM_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/test/kernels/test_kernel_vulkan.spvasm)
        add_custom_command(
            OUTPUT ${VULKAN_SPV_OUTPUT}
            COMMAND ${SPIRV_AS_EXECUTABLE} --target-env vulkan1.1
                    ${VULKAN_SPVASM_SOURCE} -o ${VULKAN_SPV_OUTPUT}
            DEPENDS ${VULKAN_SPVASM_SOURCE}
            COMMENT "Assembling Vulkan SPIR-V from .spvasm")
      endif()
    else()
      find_program(DXC_EXECUTABLE dxc)
      if(DXC_EXECUTABLE)
        set(HLSL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/test/kernels/test_kernel.hlsl)
        add_custom_command(
            OUTPUT ${VULKAN_SPV_OUTPUT}
            COMMAND ${DXC_EXECUTABLE} -T cs_6_0 -E mult_const_f -spirv
                    -fspv-target-env=vulkan1.1 ${HLSL_SOURCE} -Fo ${VULKAN_SPV_OUTPUT}
            DEPENDS ${HLSL_SOURCE}
            COMMENT "Compiling HLSL to Vulkan SPIR-V")
      endif()
    endif()
    if(EXISTS ${VULKAN_SPV_OUTPUT} OR TRUE)
      add_custom_command(
          OUTPUT ${VULKAN_SPV_HEADER}
          COMMAND ${CMAKE_COMMAND}
              -DINPUT_FILE=${VULKAN_SPV_OUTPUT}
              -DOUTPUT_FILE=${VULKAN_SPV_HEADER}
              -DVAR_NAME=ghost_test_vulkan_spirv
              -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2h.cmake
          DEPENDS ${VULKAN_SPV_OUTPUT} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2h.cmake
          COMMENT "Generating Vulkan SPIR-V header")
      add_custom_target(ghost_test_vulkan_spirv DEPENDS ${VULKAN_SPV_HEADER})
      add_dependencies(TestGhostGTest ghost_test_vulkan_spirv)
      target_compile_definitions(TestGhostGTest PRIVATE
          GHOST_TEST_VULKAN_SPIRV="ghost_test_vulkan_spirv.h")
    endif()
  endif()

  # -----------------------------------------------------------------------
  # DirectX DXIL compilation (build-time)
  # -----------------------------------------------------------------------
  if(WITH_DIRECTX AND WIN32)
    find_program(DXC_EXECUTABLE dxc)
    if(DXC_EXECUTABLE)
      set(HLSL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/test/kernels/test_kernel.hlsl)
      set(DXIL_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test/shaders/test_kernel.cso)
      set(DXIL_HEADER ${GHOST_TEST_GENERATED_DIR}/ghost_test_directx_dxil.h)
      add_custom_command(
          OUTPUT ${DXIL_OUTPUT}
          COMMAND ${DXC_EXECUTABLE} -T cs_6_0 -E mult_const_f
                  ${HLSL_SOURCE} -Fo ${DXIL_OUTPUT}
          DEPENDS ${HLSL_SOURCE}
          COMMENT "Compiling HLSL to DirectX DXIL")
      add_custom_command(
          OUTPUT ${DXIL_HEADER}
          COMMAND ${CMAKE_COMMAND}
              -DINPUT_FILE=${DXIL_OUTPUT}
              -DOUTPUT_FILE=${DXIL_HEADER}
              -DVAR_NAME=ghost_test_directx_dxil
              -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2h.cmake
          DEPENDS ${DXIL_OUTPUT} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2h.cmake
          COMMENT "Generating DirectX DXIL header")
      add_custom_target(ghost_test_dxil DEPENDS ${DXIL_HEADER})
      add_dependencies(TestGhostGTest ghost_test_dxil)
      target_compile_definitions(TestGhostGTest PRIVATE
          GHOST_TEST_DIRECTX_DXIL="ghost_test_directx_dxil.h")
    endif()
  endif()

  # -----------------------------------------------------------------------
  # OpenCL SPIR-V compilation (build-time)
  # -----------------------------------------------------------------------
  if(WITH_OPENCL)
    set(OCL_SPV_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test/shaders/test_kernel_opencl.spv)
    set(OCL_SPV_HEADER ${GHOST_TEST_GENERATED_DIR}/ghost_test_opencl_spirv.h)
    if(WITH_SPIRV_ASM)
      find_program(SPIRV_AS_EXECUTABLE spirv-as)
      if(SPIRV_AS_EXECUTABLE)
        set(OCL_SPVASM_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/test/kernels/test_kernel_opencl.spvasm)
        add_custom_command(
            OUTPUT ${OCL_SPV_OUTPUT}
            COMMAND ${SPIRV_AS_EXECUTABLE} --target-env opencl2.0
                    ${OCL_SPVASM_SOURCE} -o ${OCL_SPV_OUTPUT}
            DEPENDS ${OCL_SPVASM_SOURCE}
            COMMENT "Assembling OpenCL SPIR-V from .spvasm")
      endif()
    else()
      find_program(CLANG_EXECUTABLE clang)
      find_program(LLVM_SPIRV_EXECUTABLE llvm-spirv)
      if(CLANG_EXECUTABLE AND LLVM_SPIRV_EXECUTABLE)
        set(OCL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/test/kernels/test_kernel.cl)
        set(OCL_BC ${CMAKE_CURRENT_BINARY_DIR}/test/shaders/test_kernel.bc)
        add_custom_command(
            OUTPUT ${OCL_BC}
            COMMAND ${CLANG_EXECUTABLE} -cc1 -triple spir64 -O2 -emit-llvm-bc
                    -o ${OCL_BC} ${OCL_SOURCE}
            DEPENDS ${OCL_SOURCE}
            COMMENT "Compiling OpenCL C to LLVM bitcode")
        add_custom_command(
            OUTPUT ${OCL_SPV_OUTPUT}
            COMMAND ${LLVM_SPIRV_EXECUTABLE} ${OCL_BC} -o ${OCL_SPV_OUTPUT}
            DEPENDS ${OCL_BC}
            COMMENT "Converting LLVM bitcode to SPIR-V")
      endif()
    endif()
    if(EXISTS ${OCL_SPV_OUTPUT} OR TRUE)
      add_custom_command(
          OUTPUT ${OCL_SPV_HEADER}
          COMMAND ${CMAKE_COMMAND}
              -DINPUT_FILE=${OCL_SPV_OUTPUT}
              -DOUTPUT_FILE=${OCL_SPV_HEADER}
              -DVAR_NAME=ghost_test_opencl_spirv
              -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2h.cmake
          DEPENDS ${OCL_SPV_OUTPUT} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2h.cmake
          COMMENT "Generating OpenCL SPIR-V header")
      add_custom_target(ghost_test_opencl_spirv DEPENDS ${OCL_SPV_HEADER})
      add_dependencies(TestGhostGTest ghost_test_opencl_spirv)
      target_compile_definitions(TestGhostGTest PRIVATE
          GHOST_TEST_OPENCL_SPIRV="ghost_test_opencl_spirv.h")
    endif()
  endif()
endif()

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
