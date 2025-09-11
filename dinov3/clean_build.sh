rm -rf build
mkdir -p build
cd build
cmake -DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)') ..
make video_scene_detector
cd ..