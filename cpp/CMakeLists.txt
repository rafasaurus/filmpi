project(HaldClutApp)
cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm" AND WITH_NEON)
    message(STATUS "ARM architecture detected *using Neon instructions*")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -ftree-vectorizer-verbose=3 -mfpu=neon -mfloat-abi=hard -march=native")
    add_definitions(-DUSE_NEON)
else()
    message(STATUS "Non-ARM architecture detected")
endif()

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(hald_clut main.cpp)
add_executable(hald_clut_neon main_neon.cpp)

target_link_libraries(hald_clut Threads::Threads ${OpenCV_LIBS} )
target_link_libraries(hald_clut_neon Threads::Threads ${OpenCV_LIBS} )
