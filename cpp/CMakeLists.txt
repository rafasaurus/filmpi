cmake_minimum_required(VERSION 3.18)
project(HaldClutApp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -funroll-loops -march=native")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/stb)

add_executable(hald_clut main.cpp)
