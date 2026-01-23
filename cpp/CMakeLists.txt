cmake_minimum_required(VERSION 3.18)
project(HaldClutApp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -funroll-loops -march=native")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH "Install path prefix" FORCE)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/stb)

find_package(PkgConfig REQUIRED)
pkg_check_modules(EXIV2 REQUIRED exiv2)

add_executable(hald_clut main.cpp)
target_link_libraries(hald_clut ${EXIV2_LIBRARIES})
target_include_directories(hald_clut PRIVATE ${EXIV2_INCLUDE_DIRS})
target_compile_options(hald_clut PRIVATE ${EXIV2_CFLAGS_OTHER})

install(TARGETS hald_clut
    RUNTIME DESTINATION bin
)

if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY
    )

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
        COMMENT "Uninstalling..."
    )
endif()
