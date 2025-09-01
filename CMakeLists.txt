cmake_minimum_required(VERSION 3.10)
project(TrackNetV3)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable threading support
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Find LibTorch (PyTorch C++ API)
find_package(Torch REQUIRED)

# Include FetchContent for downloading dependencies
include(FetchContent)

# Fetch and configure Abseil
FetchContent_Declare(
    absl
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG master  # You might want to use a specific version tag
)
FetchContent_MakeAvailable(absl)

# Print OpenCV version and include dirs for debugging
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Torch_FOUND: ${Torch_FOUND}")
message(STATUS "Torch_INCLUDE_DIRS: ${Torch_INCLUDE_DIRS}")
message(STATUS "Torch_LIBRARIES: ${Torch_LIBRARIES}")

# Add executable for angle detection
add_executable(detect_angle detect_back_side_view_angle.cc)

# Include OpenCV headers
target_include_directories(detect_angle PUBLIC ${OpenCV_INCLUDE_DIRS})

# Link against OpenCV libraries, Abseil, and pthread
target_link_libraries(detect_angle PUBLIC 
    ${OpenCV_LIBS}
    absl::flags
    absl::flags_parse
    absl::strings
    Threads::Threads
)

# Add DINOv3 subdirectory
add_subdirectory(dinov3)
