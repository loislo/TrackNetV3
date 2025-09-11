# DINOv3 Subdirectory CMakeLists.txt
# This file builds all DINOv3-related executables

cmake_minimum_required(VERSION 3.10)
project(DINOv3)

# Set C++ standard to 17 (required by Abseil)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable threading support
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# Find OpenCV package with better error handling
find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND)
    # Try alternative OpenCV finding methods
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(OpenCV opencv4 QUIET)
        if(NOT OpenCV_FOUND)
            pkg_check_modules(OpenCV opencv QUIET)
        endif()
    endif()
    
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR 
            "OpenCV not found. Please install OpenCV development packages:\n"
            "  Ubuntu/Debian: sudo apt-get install libopencv-dev\n"
            "  CentOS/RHEL: sudo yum install opencv-devel\n"
            "  Fedora: sudo dnf install opencv-devel\n"
            "  macOS: brew install opencv\n"
            "Or set OpenCV_DIR to the OpenCV installation directory.")
    endif()
endif()

# Find LibTorch (PyTorch C++ API) with better error handling
find_package(Torch QUIET)
if(NOT Torch_FOUND)
    message(FATAL_ERROR 
        "LibTorch not found. Please ensure PyTorch is installed with C++ extensions:\n"
        "  pip install torch\n"
        "And set CMAKE_PREFIX_PATH to the torch cmake directory:\n"
        "  cmake -DCMAKE_PREFIX_PATH=\$(python -c 'import torch; print(torch.utils.cmake_prefix_path)') ..\n"
        "Or install LibTorch separately from https://pytorch.org/")
endif()

# Include FetchContent for downloading dependencies
include(FetchContent)

# Fetch and configure Abseil
FetchContent_Declare(
    absl
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG master  # You might want to use a specific version tag
)
FetchContent_MakeAvailable(absl)

# Set up OpenCV variables for consistent usage
if(OpenCV_FOUND)
    if(NOT OpenCV_INCLUDE_DIRS AND OpenCV_INCLUDE_DIRS)
        set(OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
    endif()
    if(NOT OpenCV_LIBS AND OpenCV_LIBRARIES)
        set(OpenCV_LIBS ${OpenCV_LIBRARIES})
    endif()
    # Handle pkg-config style variables
    if(OpenCV_CFLAGS_OTHER)
        set(OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
    endif()
    if(OpenCV_LDFLAGS)
        set(OpenCV_LIBS ${OpenCV_LDFLAGS})
    endif()
endif()

# Print debug information
message(STATUS "OpenCV_FOUND: ${OpenCV_FOUND}")
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "Torch_FOUND: ${Torch_FOUND}")
message(STATUS "Torch_INCLUDE_DIRS: ${Torch_INCLUDE_DIRS}")
message(STATUS "Torch_LIBRARIES: ${Torch_LIBRARIES}")

# Add simple test executable
add_executable(dinov3_simple_test dinov3_simple_test.cc)

# Include Torch headers
target_include_directories(dinov3_simple_test PUBLIC 
    ${TORCH_INCLUDE_DIRS}
)

# Link against Torch
target_link_libraries(dinov3_simple_test PUBLIC 
    ${TORCH_LIBRARIES}
)

# Set RPATH for Torch libraries
set_target_properties(dinov3_simple_test PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Add working DINOv3 executable
add_executable(dinov3_working dinov3_working.cc)

# Include Torch headers
target_include_directories(dinov3_working PUBLIC 
    ${TORCH_INCLUDE_DIRS}
)

# Link against Torch
target_link_libraries(dinov3_working PUBLIC 
    ${TORCH_LIBRARIES}
)

# Set RPATH for Torch libraries
set_target_properties(dinov3_working PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Add image processor executable
add_executable(dinov3_image_processor dinov3_image_processor.cc)

# Include OpenCV and Torch headers
target_include_directories(dinov3_image_processor PUBLIC 
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# Link against OpenCV, Torch, and pthread
target_link_libraries(dinov3_image_processor PUBLIC 
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    Threads::Threads
)

# Set RPATH for Torch libraries
set_target_properties(dinov3_image_processor PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Add video scene detector executable
add_executable(video_scene_detector video_scene_detector.cc)

# Include OpenCV and Torch headers
target_include_directories(video_scene_detector PUBLIC 
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# Link against OpenCV, Torch, Abseil, and pthread
target_link_libraries(video_scene_detector PUBLIC 
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    absl::flags
    absl::flags_parse
    absl::strings
    Threads::Threads
)

# Set RPATH for Torch libraries
set_target_properties(video_scene_detector PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Add scene classifier executable
add_executable(scene_classifier scene_classifier.cc)

# Include OpenCV and Torch headers
target_include_directories(scene_classifier PUBLIC 
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# Link against OpenCV, Torch, and pthread
target_link_libraries(scene_classifier PUBLIC 
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    Threads::Threads
)

# Set RPATH for Torch libraries
set_target_properties(scene_classifier PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)