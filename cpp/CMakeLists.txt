project(HaldClutApp)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(hald_clut main.cpp)

# Link OpenCV libraries
target_link_libraries(hald_clut ${OpenCV_LIBS})


