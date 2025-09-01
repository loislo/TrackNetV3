# DINOv3 Subdirectory CMakeLists.txt
# This file builds all DINOv3-related executables

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
