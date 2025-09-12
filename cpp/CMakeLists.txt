cmake_minimum_required(VERSION 3.18)
project(HaldClutApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(COMMON_FLAGS "-O3")

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" CPU_ARCH)
message(STATUS "Detected architecture: ${CPU_ARCH}")

if(CPU_ARCH MATCHES "x86_64|amd64|i[3-6]86")
    message(STATUS "Building for x86 with AVX2")
    add_executable(hald_clut main_generic.cpp)
    target_compile_options(hald_clut PRIVATE ${COMMON_FLAGS} -mavx2 -mfma -march=x86-64-v3)
    target_compile_definitions(hald_clut PRIVATE HALD_X86=1)

elseif(CPU_ARCH MATCHES "aarch64|arm64")
    message(STATUS "Building for ARM64 with NEON (AArch64)")
    add_executable(hald_clut main_generic.cpp)
    target_compile_options(hald_clut PRIVATE ${COMMON_FLAGS} -march=armv8.2-a)
    target_compile_definitions(hald_clut PRIVATE HALD_ARM=1)

    # Try SVE2 if available
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=armv9.2-a+sve2" HAS_SVE2)
    if(HAS_SVE2)
        message(STATUS "SVE2 supported, enabling it")
        target_compile_options(hald_clut PRIVATE -march=armv9.2-a+sve2)
        target_compile_definitions(hald_clut PRIVATE HALD_ARM_SVE2=1)
    endif()

elseif(CPU_ARCH MATCHES "^arm")
    message(STATUS "Building for 32-bit ARM with NEON")
    add_executable(hald_clut main_generic.cpp)
    target_compile_options(hald_clut PRIVATE ${COMMON_FLAGS} -mfpu=neon -mfloat-abi=hard -march=armv7-a)
    target_compile_definitions(hald_clut PRIVATE HALD_ARM=1)

else()
    message(WARNING "Unknown architecture, building generic fallback")
    add_executable(hald_clut main_generic.cpp)
    target_compile_options(hald_clut PRIVATE ${COMMON_FLAGS})
endif()

target_link_libraries(hald_clut PRIVATE Threads::Threads ${OpenCV_LIBS})

