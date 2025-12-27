cmake_minimum_required(VERSION 3.16)
project(lutmac VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options
option(LUTMAC_BUILD_TESTS "Build tests" ON)
option(LUTMAC_BUILD_TOOLS "Build CLI tools" ON)
option(LUTMAC_ENABLE_AVX2 "Enable AVX2 optimizations" ON)
option(LUTMAC_ENABLE_AVX512 "Enable AVX-512 optimizations" OFF)
option(LUTMAC_ENABLE_NEON "Enable ARM NEON optimizations" ON)

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
    set(LUTMAC_ARCH_ARM TRUE)
    message(STATUS "Detected ARM architecture")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686")
    set(LUTMAC_ARCH_X86 TRUE)
    message(STATUS "Detected x86 architecture")
endif()

# SIMD flags
if(LUTMAC_ARCH_X86)
    if(LUTMAC_ENABLE_AVX512)
        if(MSVC)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX512")
        else()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512bw -mavx512vl")
        endif()
        add_compile_definitions(LUTMAC_AVX512)
    elseif(LUTMAC_ENABLE_AVX2)
        if(MSVC)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
        else()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
        endif()
        add_compile_definitions(LUTMAC_AVX2)
    endif()
elseif(LUTMAC_ARCH_ARM)
    if(LUTMAC_ENABLE_NEON)
        if(NOT MSVC)
            # NEON is usually enabled by default on ARM64
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
        endif()
        add_compile_definitions(LUTMAC_NEON)
    endif()
endif()

# Release optimizations
if(NOT MSVC)
    # Always use O3 for better performance
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -DNDEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")
    
    # Apple Silicon optimizations
    if(APPLE AND LUTMAC_ARCH_ARM)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1")
    endif()
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party)

# Source files
set(LUTMAC_SOURCES
    src/kernels/lut_gemm_scalar.cpp
    src/kernels/lut_gemm_simd.cpp
    src/kernels/transform_simd.cpp
    src/kernels/lut_kernel.cpp
    src/layers/attention.cpp
    src/layers/ffn.cpp
    src/layers/norm.cpp
    src/layers/embedding.cpp
    src/model.cpp
    src/inference.cpp
    src/quantize.cpp
    src/format.cpp
)

# Main library
add_library(lutmac STATIC ${LUTMAC_SOURCES})
target_include_directories(lutmac PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

# CLI tools
if(LUTMAC_BUILD_TOOLS)
    add_executable(lutmac-quantize tools/lutmac-quantize.cpp)
    target_link_libraries(lutmac-quantize lutmac)
    
    add_executable(lutmac-inference tools/lutmac-inference.cpp)
    target_link_libraries(lutmac-inference lutmac)
    
    add_executable(lutmac-bench tools/lutmac-bench.cpp)
    target_link_libraries(lutmac-bench lutmac)

    add_executable(test_loader tools/test_loader.cpp)
    target_link_libraries(test_loader lutmac)
endif()

# Tests
if(LUTMAC_BUILD_TESTS)
    enable_testing()
    
    add_executable(test_quantization tests/test_quantization.cpp)
    target_link_libraries(test_quantization lutmac)
    add_test(NAME test_quantization COMMAND test_quantization)
    
    add_executable(test_lut_gemm tests/test_lut_gemm.cpp)
    target_link_libraries(test_lut_gemm PRIVATE lutmac)
    add_executable(test_tmac_correctness tests/test_tmac_correctness.cpp)
    target_link_libraries(test_tmac_correctness PRIVATE lutmac)
    add_test(NAME test_lut_gemm COMMAND test_lut_gemm)
    add_test(NAME test_tmac_correctness COMMAND test_tmac_correctness)
endif()

# Installation
install(TARGETS lutmac ARCHIVE DESTINATION lib)
install(DIRECTORY include/lutmac DESTINATION include)

message(STATUS "LutMac configuration:")
message(STATUS "  Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  AVX2: ${LUTMAC_ENABLE_AVX2}")
message(STATUS "  AVX-512: ${LUTMAC_ENABLE_AVX512}")
message(STATUS "  NEON: ${LUTMAC_ENABLE_NEON}")
