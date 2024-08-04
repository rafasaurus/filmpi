project(HaldClutApp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mavx2 -march=native")
# Find OpenCV
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(hald_clut main.cpp)
add_executable(hald_clut_avx2 main_avx2.cpp)

# Link OpenCV libraries
target_link_libraries(hald_clut ${OpenCV_LIBS})
target_link_libraries(hald_clut_avx2 ${OpenCV_LIBS})


