cmake_minimum_required(VERSION 3.10)
project(DINOv3)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

# Set C++ standard to 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES "89")


# Find essential packages
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

# Include FetchContent for downloading dependencies
include(FetchContent)

# Fetch and configure a stable version of Abseil
FetchContent_Declare(
    absl
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG 20230802.1 # Using a specific LTS tag for stability
)
FetchContent_MakeAvailable(absl)

# Add the video scene detector executable
add_executable(video_scene_detector video_scene_detector.cc)

# Include necessary directories
target_include_directories(video_scene_detector PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# Link all required libraries to the executable
target_link_libraries(video_scene_detector PUBLIC
    # Link dependencies first
    Threads::Threads
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}

    # Then link the Abseil components needed by the source file
    absl::flags
    absl::flags_parse
    absl::strings
)

# Set RPATH for Torch libraries to ensure they are found at runtime
set_target_properties(video_scene_detector PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
