# filmpi — HALD CLUT applier
#
# Builds on Linux x86-64 (AVX2/SSE4.1), Raspberry Pi & Jetson (NEON),
# and Android via the NDK toolchain file. The target architecture is
# autodetected; SIMD kernels are compiled per-architecture and selected
# at runtime, so a generic build still uses the best kernel the CPU has.
#
# Options:
#   FILMPI_SIMD=ON|OFF        build SIMD kernels (default ON)
#   FILMPI_NATIVE=ON|OFF      add -march=native for this machine (default OFF)
#   FILMPI_WITH_EXIV2=AUTO|ON|OFF  EXIF/IPTC/XMP support (default AUTO)
#   FILMPI_BUILD_CLI=ON|OFF   build the hald_clut executable (default ON)
#
# `make install` installs to ~/.local/bin unless CMAKE_INSTALL_PREFIX is set.

cmake_minimum_required(VERSION 3.18)
project(filmpi VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{HOME})
    set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH
        "Install path prefix" FORCE)
endif()

option(FILMPI_SIMD "Build SIMD kernels for the target architecture" ON)
option(FILMPI_NATIVE "Optimize for the build machine (-march=native)" OFF)
option(FILMPI_BUILD_CLI "Build the hald_clut command line tool" ON)
set(FILMPI_WITH_EXIV2 AUTO CACHE STRING
    "Copy image metadata via Exiv2 (AUTO/ON/OFF)")
set_property(CACHE FILMPI_WITH_EXIV2 PROPERTY STRINGS AUTO ON OFF)

# ---------------------------------------------------------------------------
# Target architecture detection
# ---------------------------------------------------------------------------
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _filmpi_arch)
set(FILMPI_ARCH_X86 FALSE)
set(FILMPI_ARCH_ARM64 FALSE)
set(FILMPI_ARCH_ARM32 FALSE)
if(_filmpi_arch MATCHES "^(x86_64|amd64|i[3-6]86|x86)$")
    set(FILMPI_ARCH_X86 TRUE)
elseif(_filmpi_arch MATCHES "^(aarch64|arm64)")
    set(FILMPI_ARCH_ARM64 TRUE)
elseif(_filmpi_arch MATCHES "^arm")
    set(FILMPI_ARCH_ARM32 TRUE)
endif()

include(CheckCXXCompilerFlag)

set(_kernel_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels)
set(FILMPI_KERNEL_SOURCES ${_kernel_dir}/kernel_scalar.cpp)
set(FILMPI_SIMD_DEFINITIONS "")
set(FILMPI_SIMD_SUMMARY "scalar")

if(FILMPI_SIMD AND NOT MSVC)
    if(FILMPI_ARCH_X86)
        check_cxx_compiler_flag("-msse4.1" FILMPI_COMPILER_HAS_SSE41)
        check_cxx_compiler_flag("-mavx2" FILMPI_COMPILER_HAS_AVX2)
        if(FILMPI_COMPILER_HAS_SSE41)
            list(APPEND FILMPI_KERNEL_SOURCES ${_kernel_dir}/kernel_sse41.cpp)
            set_source_files_properties(${_kernel_dir}/kernel_sse41.cpp
                PROPERTIES COMPILE_OPTIONS "-msse4.1")
            list(APPEND FILMPI_SIMD_DEFINITIONS FILMPI_HAVE_SSE41=1)
            string(APPEND FILMPI_SIMD_SUMMARY ", sse4.1")
        endif()
        if(FILMPI_COMPILER_HAS_AVX2)
            list(APPEND FILMPI_KERNEL_SOURCES ${_kernel_dir}/kernel_avx2.cpp)
            set_source_files_properties(${_kernel_dir}/kernel_avx2.cpp
                PROPERTIES COMPILE_OPTIONS "-mavx2")
            list(APPEND FILMPI_SIMD_DEFINITIONS FILMPI_HAVE_AVX2=1)
            string(APPEND FILMPI_SIMD_SUMMARY ", avx2")
        endif()
    elseif(FILMPI_ARCH_ARM64)
        # NEON (ASIMD) is mandatory on AArch64 — no extra flags needed.
        list(APPEND FILMPI_KERNEL_SOURCES ${_kernel_dir}/kernel_neon.cpp)
        list(APPEND FILMPI_SIMD_DEFINITIONS FILMPI_HAVE_NEON=1)
        string(APPEND FILMPI_SIMD_SUMMARY ", neon")
    elseif(FILMPI_ARCH_ARM32)
        check_cxx_compiler_flag("-mfpu=neon" FILMPI_COMPILER_HAS_NEON)
        if(FILMPI_COMPILER_HAS_NEON)
            list(APPEND FILMPI_KERNEL_SOURCES ${_kernel_dir}/kernel_neon.cpp)
            set_source_files_properties(${_kernel_dir}/kernel_neon.cpp
                PROPERTIES COMPILE_OPTIONS "-mfpu=neon")
            list(APPEND FILMPI_SIMD_DEFINITIONS FILMPI_HAVE_NEON=1)
            string(APPEND FILMPI_SIMD_SUMMARY ", neon (runtime checked)")
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)

set(FILMPI_HAS_EXIV2 FALSE)
if(NOT FILMPI_WITH_EXIV2 STREQUAL "OFF")
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(EXIV2 QUIET IMPORTED_TARGET exiv2)
    endif()
    if(EXIV2_FOUND)
        set(FILMPI_HAS_EXIV2 TRUE)
    elseif(FILMPI_WITH_EXIV2 STREQUAL "ON")
        message(FATAL_ERROR "FILMPI_WITH_EXIV2=ON but exiv2 was not found "
                            "(install libexiv2-dev or set FILMPI_WITH_EXIV2=OFF)")
    endif()
endif()

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
add_library(filmpi STATIC
    src/clut.cpp
    src/image.cpp
    src/metadata.cpp
    ${FILMPI_KERNEL_SOURCES}
)
target_include_directories(filmpi
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/external/stb
)
target_compile_definitions(filmpi PRIVATE ${FILMPI_SIMD_DEFINITIONS})
target_link_libraries(filmpi PUBLIC Threads::Threads)

if(NOT MSVC)
    target_compile_options(filmpi PRIVATE
        $<$<CONFIG:Release,RelWithDebInfo>:-O3 -funroll-loops>)
    if(FILMPI_NATIVE)
        target_compile_options(filmpi PRIVATE -march=native)
    endif()
endif()

if(FILMPI_HAS_EXIV2)
    target_compile_definitions(filmpi PRIVATE FILMPI_WITH_EXIV2=1)
    target_link_libraries(filmpi PRIVATE PkgConfig::EXIV2)
endif()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if(FILMPI_BUILD_CLI)
    add_executable(hald_clut cli/main.cpp)
    target_link_libraries(hald_clut PRIVATE filmpi)
    target_compile_definitions(hald_clut PRIVATE
        FILMPI_VERSION="${PROJECT_VERSION}")
    install(TARGETS hald_clut RUNTIME DESTINATION bin)
endif()

# ---------------------------------------------------------------------------
# Uninstall target
# ---------------------------------------------------------------------------
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY
    )
    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
        COMMENT "Uninstalling..."
    )
endif()

# ---------------------------------------------------------------------------
# Configuration summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "filmpi ${PROJECT_VERSION} configuration:")
message(STATUS "  target         : ${CMAKE_SYSTEM_NAME} / ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  build type     : ${CMAKE_BUILD_TYPE}")
message(STATUS "  SIMD kernels   : ${FILMPI_SIMD_SUMMARY}")
message(STATUS "  exiv2 metadata : ${FILMPI_HAS_EXIV2}")
message(STATUS "  install prefix : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
