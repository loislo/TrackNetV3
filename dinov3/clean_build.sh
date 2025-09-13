#!/bin/bash

echo "Cleaning venv .dinov3..."
rm -rf .dinov3
echo "Cleaning build..."
rm -rf build
echo "Creating venv .dinov3..."
python3 -m venv .dinov3
echo "Activating venv .dinov3..."
source .dinov3/bin/activate
echo "Installing dependencies..."
pip install -r requirements.txt
echo "Creating build directory..."
mkdir -p build
cd build
echo "Running cmake..."
cmake -DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)') ..
echo "Running make..."
make -s -j 16 video_scene_detector
cd ..
echo "Done"