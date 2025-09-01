# DINOv3 Subdirectory

This directory contains all DINOv3-related code for feature extraction, scene detection, and video analysis.

## Contents

### Python Scripts
- **`DYNOv3_trace.py`**: Traces DINOv3 model for C++ deployment
- **`requirements.txt`**: Python dependencies (in parent directory)

### C++ Executables
- **`dinov3_simple_test.cc`**: Basic LibTorch test without OpenCV
- **`dinov3_working.cc`**: GPU-accelerated DINOv3 inference test
- **`dinov3_image_processor.cc`**: Process real images with DINOv3
- **`video_scene_detector.cc`**: Main video scene detection tool
- **`scene_classifier.cc`**: Advanced scene classification (requires torch headers)
- **`simple_scene_classifier.cc`**: Standalone scene classifier demo

### Model Files
- **`dinov3_vits16_traced.pt`**: Traced DINOv3 model (25GB) for C++ deployment

### Documentation
- **`DINOv3_Scene_Detection_Guide.md`**: Comprehensive usage guide

## Building

The executables are built from the parent directory using the main CMakeLists.txt, which includes this subdirectory.

```bash
cd ..  # Go to parent directory
mkdir -p build
cd build
cmake ..
make video_scene_detector  # Build specific executable
make                        # Build all executables
```

## Usage

### Basic Scene Detection
```bash
cd dinov3
../build/video_scene_detector --video=../raw_video/1_1_9_5.mp4
```

### With Custom Parameters
```bash
cd dinov3
../build/video_scene_detector \
    --video=../raw_video/1_1_9_5.mp4 \
    --threshold=0.9 \
    --sample_interval=2 \
    --verbose \
    --output_dir=results
```

**Note**: The executable should be run from the `dinov3` directory since the model file (`dinov3_vits16_traced.pt`) is located there.

### Scene Classifier Demo
```bash
cd dinov3
g++ -std=c++17 simple_scene_classifier.cc -o simple_scene_classifier
./simple_scene_classifier
```

## Key Features

- **GPU Acceleration**: Uses MPS (Metal Performance Shaders) on Mac
- **Flexible Parameters**: Adjustable similarity threshold, sample interval, scene length
- **Professional CLI**: Abseil flags for robust command-line interface
- **Multiple Outputs**: Console results, detailed text files, CSV feature vectors
- **Scene Classification**: Identifies different camera angles and scene types

## Dependencies

- **LibTorch**: PyTorch C++ API
- **OpenCV**: Computer vision library
- **Abseil**: Google's C++ library (for flags)
- **C++17**: Modern C++ standard

## Architecture

The system works by:
1. **Loading** traced DINOv3 model
2. **Processing** video frames at regular intervals
3. **Extracting** 4096-dimensional feature vectors
4. **Comparing** consecutive frames using cosine similarity
5. **Detecting** scene changes when similarity drops below threshold
6. **Classifying** scenes based on feature characteristics
7. **Outputting** results in multiple formats

## Performance

- **GPU**: ~318ms per frame (MPS acceleration)
- **CPU**: ~7s per frame
- **Speedup**: ~22x with GPU acceleration
- **Memory**: Efficient batch processing for long videos

## Integration

This system integrates with TrackNetV3 by:
- Detecting different camera angles in videos
- Segmenting videos into meaningful scenes
- Providing feature vectors for further analysis
- Enabling scene-specific ball tracking

## Troubleshooting

- **Model not found**: Ensure `dinov3_vits16_traced.pt` is in this directory
- **Build errors**: Check that LibTorch and OpenCV are properly installed
- **Performance issues**: Verify MPS/CUDA availability
- **Memory problems**: Increase sample interval for long videos
