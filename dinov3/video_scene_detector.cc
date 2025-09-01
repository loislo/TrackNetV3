#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/flags/usage.h>

// Define command-line flags
ABSL_FLAG(std::string, video, "", "Path to input video file (required)");
ABSL_FLAG(double, threshold, 0.85, "Similarity threshold for scene change detection (0.0-1.0)");
ABSL_FLAG(int, min_scene_length, 30, "Minimum frames per scene");
ABSL_FLAG(int, sample_interval, 5, "Sample every Nth frame for efficiency");
ABSL_FLAG(std::string, output_dir, "results", "Output directory for results");
ABSL_FLAG(bool, verbose, false, "Enable verbose output");
ABSL_FLAG(bool, save_features, true, "Save feature vectors to CSV file");
ABSL_FLAG(bool, help, false, "Show this help message");

// Structure to store scene information
struct SceneInfo {
    int start_frame;
    int end_frame;
    double start_time;
    double end_time;
    std::vector<float> representative_features;
    std::string scene_type;  // "court_view", "back_side", "other_angle", etc.
};

// Function to preprocess frame for DINOv3
torch::Tensor preprocess_frame(const cv::Mat& frame, const torch::Device& device) {
    // Resize frame to 224x224
    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(224, 224));
    
    // Convert BGR to RGB
    cv::Mat rgb_frame;
    cv::cvtColor(resized_frame, rgb_frame, cv::COLOR_BGR2RGB);
    
    // Convert to float and normalize to [0, 1]
    cv::Mat float_frame;
    rgb_frame.convertTo(float_frame, CV_32F, 1.0/255.0);
    
    // Convert OpenCV Mat to torch tensor
    auto tensor = torch::from_blob(float_frame.data, {224, 224, 3}, torch::kFloat32);
    
    // Permute dimensions from (H, W, C) to (C, H, W) and add batch dimension
    tensor = tensor.permute({2, 0, 1}).unsqueeze(0);
    
    // Move to device
    tensor = tensor.to(device);
    
    return tensor;
}

// Function to compute cosine similarity between two feature vectors
double cosine_similarity(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    if (vec1.size() != vec2.size()) {
        std::cerr << "Error: Feature vectors have different sizes!" << std::endl;
        return 0.0;
    }
    
    double dot_product = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;
    
    for (size_t i = 0; i < vec1.size(); ++i) {
        dot_product += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }
    
    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);
    
    if (norm1 == 0.0 || norm2 == 0.0) {
        return 0.0;
    }
    
    return dot_product / (norm1 * norm2);
}

// Function to classify scene type based on feature vector
std::string classify_scene_type(const std::vector<float>& features) {
    // This is a simple heuristic - you can make this more sophisticated
    // by training a classifier on labeled data
    
    // Calculate some basic statistics
    double mean = 0.0;
    double variance = 0.0;
    
    for (float f : features) {
        mean += f;
    }
    mean /= features.size();
    
    for (float f : features) {
        variance += (f - mean) * (f - mean);
    }
    variance /= features.size();
    
    // Simple classification based on feature statistics
    if (variance > 0.1) {
        return "complex_scene";  // High variance = complex scene
    } else if (mean > 0.05) {
        return "bright_scene";   // High mean = bright scene
    } else {
        return "simple_scene";   // Low variance and mean = simple scene
    }
}

// Function to print usage information
void print_usage() {
    std::cout << "DINOv3 Video Scene Detector" << std::endl;
    std::cout << "===========================" << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: video_scene_detector [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Required:" << std::endl;
    std::cout << "  --video=<path>          Path to input video file" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --threshold=<value>     Similarity threshold (0.0-1.0, default: 0.85)" << std::endl;
    std::cout << "  --min_scene_length=<N>  Minimum frames per scene (default: 30)" << std::endl;
    std::cout << "  --sample_interval=<N>   Sample every Nth frame (default: 5)" << std::endl;
    std::cout << "  --output_dir=<path>     Output directory (default: results)" << std::endl;
    std::cout << "  --save_features         Save feature vectors to CSV (default: true)" << std::endl;
    std::cout << "  --verbose               Enable verbose output" << std::endl;
    std::cout << "  --help                  Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  video_scene_detector --video=raw_video/1_1_9_5.mp4" << std::endl;
    std::cout << "  video_scene_detector --video=video.mp4 --threshold=0.9 --verbose" << std::endl;
    std::cout << "  video_scene_detector --video=video.mp4 --min_scene_length=60 --sample_interval=2" << std::endl;
}

// Function to validate command-line arguments
bool validate_arguments(const std::string& video_path, double similarity_threshold, 
                       int min_scene_length, int sample_interval) {
    // Validate required arguments
    if (video_path.empty()) {
        std::cerr << "Error: --video flag is required!" << std::endl;
        std::cerr << "Use --help for usage information." << std::endl;
        return false;
    }
    
    // Validate threshold range
    if (similarity_threshold < 0.0 || similarity_threshold > 1.0) {
        std::cerr << "Error: --threshold must be between 0.0 and 1.0!" << std::endl;
        return false;
    }
    
    // Validate other parameters
    if (min_scene_length <= 0) {
        std::cerr << "Error: --min_scene_length must be positive!" << std::endl;
        return false;
    }
    
    if (sample_interval <= 0) {
        std::cerr << "Error: --sample_interval must be positive!" << std::endl;
        return false;
    }
    
    return true;
}

// Function to print configuration information
void print_configuration(const std::string& video_path, double similarity_threshold,
                        int min_scene_length, int sample_interval, 
                        const std::string& output_dir, bool save_features) {
    std::cout << "DINOv3 Video Scene Detector" << std::endl;
    std::cout << "===========================" << std::endl;
    std::cout << "Video file: " << video_path << std::endl;
    std::cout << "Similarity threshold: " << similarity_threshold << std::endl;
    std::cout << "Minimum scene length: " << min_scene_length << " frames" << std::endl;
    std::cout << "Sample interval: " << sample_interval << std::endl;
    std::cout << "Output directory: " << output_dir << std::endl;
    std::cout << "Save features: " << (save_features ? "yes" : "no") << std::endl;
    std::cout << std::endl;
}

// Function to detect available device
torch::Device detect_device(bool verbose) {
    if (verbose) std::cout << "Checking device availability..." << std::endl;
    torch::Device device(torch::kCPU);
    
    if (torch::hasMPS()) {
        device = torch::Device(torch::kMPS);
        if (verbose) std::cout << "MPS (Metal Performance Shaders) is available! Using device: MPS" << std::endl;
    } else if (torch::hasCUDA()) {
        device = torch::Device(torch::kCUDA);
        if (verbose) std::cout << "CUDA is available! Using device: CUDA" << std::endl;
    } else {
        if (verbose) std::cout << "Using CPU device" << std::endl;
    }
    
    return device;
}

// Function to load DINOv3 model
torch::jit::script::Module load_model(const torch::Device& device, bool verbose) {
    if (verbose) std::cout << "Loading DINOv3 model..." << std::endl;
    
    // Try multiple possible model paths
    std::vector<std::string> possible_paths = {
        "dinov3/dinov3_vits16_traced.pt",    // From parent directory (most common)
        "dinov3_vits16_traced.pt",           // From dinov3 directory
        "../dinov3/dinov3_vits16_traced.pt"  // From other subdirectory
    };
    
    torch::jit::script::Module model;
    std::string successful_path;
    
    for (const auto& model_path : possible_paths) {
        if (verbose) std::cout << "Trying model path: " << model_path << std::endl;
        
        try {
            model = torch::jit::load(model_path);
            successful_path = model_path;
            break;
        } catch (const c10::Error& e) {
            if (verbose) std::cout << "Failed to load from: " << model_path << std::endl;
            continue;
        }
    }
    
    if (successful_path.empty()) {
        throw std::runtime_error("Could not load DINOv3 model from any of the attempted paths");
    }
    
    if (verbose) std::cout << "Successfully loaded model from: " << successful_path << std::endl;
    
    model.eval();
    model.to(device);
    
    if (verbose) std::cout << "Model loaded successfully!" << std::endl;
    return model;
}

// Function to get video properties
void get_video_properties(cv::VideoCapture& cap, bool verbose) {
    if (verbose) {
        int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
        double fps = cap.get(cv::CAP_PROP_FPS);
        int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        
        std::cout << "Video properties:" << std::endl;
        std::cout << "  Total frames: " << total_frames << std::endl;
        std::cout << "  FPS: " << fps << std::endl;
        std::cout << "  Resolution: " << width << "x" << height << std::endl;
    }
}

// Function to process video frames and extract features
std::pair<std::vector<std::vector<float>>, std::vector<double>> process_video_frames(
    cv::VideoCapture& cap, torch::jit::script::Module& model, 
    const torch::Device& device, int sample_interval, bool verbose) {
    
    std::vector<std::vector<float>> frame_features;
    std::vector<double> frame_timestamps;
    
    cv::Mat frame;
    int frame_count = 0;
    int processed_frames = 0;
    
    if (verbose) std::cout << "\nProcessing video frames..." << std::endl;
    
    torch::NoGradGuard no_grad;
    
    while (cap.read(frame)) {
        frame_count++;
        
        // Sample frames at regular intervals for efficiency
        if (frame_count % sample_interval != 0) {
            continue;
        }
        
        double fps = cap.get(cv::CAP_PROP_FPS);
        double timestamp = frame_count / fps;
        
        // Preprocess frame
        auto input_tensor = preprocess_frame(frame, device);
        
        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        auto output = model.forward(inputs);
        
        if (output.isTensor()) {
            auto output_tensor = output.toTensor();
            auto cpu_tensor = output_tensor.cpu();
            
            // Convert to std::vector<float>
            std::vector<float> features;
            features.reserve(cpu_tensor.size(1));
            for (int i = 0; i < cpu_tensor.size(1); ++i) {
                features.push_back(cpu_tensor[0][i].item<float>());
            }
            
            frame_features.push_back(features);
            frame_timestamps.push_back(timestamp);
            processed_frames++;
            
            if (verbose && processed_frames % 10 == 0) {
                std::cout << "Processed " << processed_frames << " frames..." << std::endl;
            }
        }
    }
    
    if (verbose) std::cout << "Finished processing " << processed_frames << " frames." << std::endl;
    
    return {frame_features, frame_timestamps};
}

// Function to detect scenes from feature vectors
std::vector<SceneInfo> detect_scenes(const std::vector<std::vector<float>>& frame_features,
                                    const std::vector<double>& frame_timestamps,
                                    double similarity_threshold, int min_scene_length,
                                    int sample_interval, bool verbose) {
    
    if (verbose) std::cout << "\nDetecting scenes..." << std::endl;
    
    if (frame_features.empty()) {
        std::cerr << "Error: No frames were processed!" << std::endl;
        return {};
    }
    
    std::vector<SceneInfo> scenes;
    
    // Start with the first frame as a new scene
    SceneInfo current_scene;
    current_scene.start_frame = 0;
    current_scene.start_time = frame_timestamps[0];
    current_scene.representative_features = frame_features[0];
    current_scene.scene_type = classify_scene_type(frame_features[0]);
    
    for (size_t i = 1; i < frame_features.size(); ++i) {
        double similarity = cosine_similarity(current_scene.representative_features, frame_features[i]);
        
        // Check if this is a scene change
        if (similarity < similarity_threshold) {
            // End current scene
            current_scene.end_frame = i - 1;
            current_scene.end_time = frame_timestamps[i - 1];
            
            // Only add scene if it's long enough
            if (current_scene.end_frame - current_scene.start_frame >= min_scene_length / sample_interval) {
                scenes.push_back(current_scene);
            }
            
            // Start new scene
            current_scene.start_frame = i;
            current_scene.start_time = frame_timestamps[i];
            current_scene.representative_features = frame_features[i];
            current_scene.scene_type = classify_scene_type(frame_features[i]);
        }
    }
    
    // Add the last scene
    current_scene.end_frame = frame_features.size() - 1;
    current_scene.end_time = frame_timestamps.back();
    if (current_scene.end_frame - current_scene.start_frame >= min_scene_length / sample_interval) {
        scenes.push_back(current_scene);
    }
    
    return scenes;
}

// Function to print scene detection results
void print_scene_results(const std::vector<SceneInfo>& scenes, int sample_interval) {
    std::cout << "\n=== SCENE DETECTION RESULTS ===" << std::endl;
    std::cout << "Found " << scenes.size() << " scenes:" << std::endl;
    std::cout << std::endl;
    
    for (size_t i = 0; i < scenes.size(); ++i) {
        const auto& scene = scenes[i];
        int scene_frames = (scene.end_frame - scene.start_frame + 1) * sample_interval;
        double scene_duration = scene.end_time - scene.start_time;
        
        std::cout << "Scene " << (i + 1) << ":" << std::endl;
        std::cout << "  Time: " << std::fixed << std::setprecision(2) 
                  << scene.start_time << "s - " << scene.end_time << "s (" 
                  << scene_duration << "s)" << std::endl;
        std::cout << "  Frames: " << scene.start_frame * sample_interval << " - " 
                  << scene.end_frame * sample_interval << " (" << scene_frames << " frames)" << std::endl;
        std::cout << "  Type: " << scene.scene_type << std::endl;
        std::cout << std::endl;
    }
}

// Function to save detailed results to file
void save_detailed_results(const std::vector<SceneInfo>& scenes, const std::string& video_path,
                          int total_frames, double fps, int width, int height,
                          int processed_frames, int sample_interval, double similarity_threshold,
                          int min_scene_length, const std::string& output_dir) {
    
    std::string output_file = output_dir + "/scene_detection_results.txt";
    std::ofstream file(output_file);
    if (file.is_open()) {
        file << "DINOv3 Video Scene Detection Results" << std::endl;
        file << "=====================================" << std::endl;
        file << "Video: " << video_path << std::endl;
        file << "Total frames: " << total_frames << std::endl;
        file << "FPS: " << fps << std::endl;
        file << "Resolution: " << width << "x" << height << std::endl;
        file << "Processed frames: " << processed_frames << std::endl;
        file << "Sample interval: " << sample_interval << std::endl;
        file << "Similarity threshold: " << similarity_threshold << std::endl;
        file << "Minimum scene length: " << min_scene_length << " frames" << std::endl;
        file << std::endl;
        
        file << "Detected Scenes:" << std::endl;
        file << "===============" << std::endl;
        
        for (size_t i = 0; i < scenes.size(); ++i) {
            const auto& scene = scenes[i];
            int scene_frames = (scene.end_frame - scene.start_frame + 1) * sample_interval;
            double scene_duration = scene.end_time - scene.start_time;
            
            file << "Scene " << (i + 1) << ":" << std::endl;
            file << "  Start time: " << std::fixed << std::setprecision(2) << scene.start_time << "s" << std::endl;
            file << "  End time: " << std::fixed << std::setprecision(2) << scene.end_time << "s" << std::endl;
            file << "  Duration: " << std::fixed << std::setprecision(2) << scene_duration << "s" << std::endl;
            file << "  Start frame: " << scene.start_frame * sample_interval << std::endl;
            file << "  End frame: " << scene.end_frame * sample_interval << std::endl;
            file << "  Frame count: " << scene_frames << std::endl;
            file << "  Scene type: " << scene.scene_type << std::endl;
            file << std::endl;
        }
        
        file.close();
        std::cout << "Detailed results saved to: " << output_file << std::endl;
    }
}

// Function to save feature vectors to CSV
void save_feature_vectors(const std::vector<std::vector<float>>& frame_features,
                         const std::vector<double>& frame_timestamps,
                         int sample_interval, const std::string& output_dir) {
    
    std::string features_file = output_dir + "/frame_features.csv";
    std::ofstream features_csv(features_file);
    if (features_csv.is_open()) {
        // Header
        features_csv << "frame,timestamp";
        for (int i = 0; i < 4096; ++i) {
            features_csv << ",feature_" << i;
        }
        features_csv << std::endl;
        
        // Data
        for (size_t i = 0; i < frame_features.size(); ++i) {
            features_csv << i * sample_interval << "," << frame_timestamps[i];
            for (float feature : frame_features[i]) {
                features_csv << "," << feature;
            }
            features_csv << std::endl;
        }
        
        features_csv.close();
        std::cout << "Feature vectors saved to: " << features_file << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Set program name for help messages
    absl::SetProgramUsageMessage("DINOv3 Video Scene Detector - Detect scene changes and camera angles in videos");
    
    // Parse command-line flags
    absl::ParseCommandLine(argc, argv);
    
    // Check for help flag
    if (absl::GetFlag(FLAGS_help)) {
        print_usage();
        return 0;
    }
    
    // Get flag values
    std::string video_path = absl::GetFlag(FLAGS_video);
    double similarity_threshold = absl::GetFlag(FLAGS_threshold);
    int min_scene_length = absl::GetFlag(FLAGS_min_scene_length);
    int sample_interval = absl::GetFlag(FLAGS_sample_interval);
    std::string output_dir = absl::GetFlag(FLAGS_output_dir);
    bool verbose = absl::GetFlag(FLAGS_verbose);
    bool save_features = absl::GetFlag(FLAGS_save_features);
    
    // Validate arguments
    if (!validate_arguments(video_path, similarity_threshold, min_scene_length, sample_interval)) {
        return -1;
    }
    
    // Print configuration
    if (verbose) {
        print_configuration(video_path, similarity_threshold, min_scene_length, 
                           sample_interval, output_dir, save_features);
    }
    
    try {
        // Detect available device
        torch::Device device = detect_device(verbose);
        
        // Load DINOv3 model
        torch::jit::script::Module model = load_model(device, verbose);
        
        // Open video file
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video file: " << video_path << std::endl;
            return -1;
        }
        
        // Get video properties
        get_video_properties(cap, verbose);
        
        // Process video frames and extract features
        auto [frame_features, frame_timestamps] = process_video_frames(
            cap, model, device, sample_interval, verbose);
        
        cap.release();
        
        // Detect scenes
        std::vector<SceneInfo> scenes = detect_scenes(frame_features, frame_timestamps,
                                                     similarity_threshold, min_scene_length,
                                                     sample_interval, verbose);
        
        // Print results
        print_scene_results(scenes, sample_interval);
        
        // Save detailed results
        double fps = cap.get(cv::CAP_PROP_FPS);
        int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
        int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        
        save_detailed_results(scenes, video_path, total_frames, fps, width, height,
                             frame_features.size(), sample_interval, similarity_threshold,
                             min_scene_length, output_dir);
        
        // Save feature vectors if requested
        if (save_features) {
            save_feature_vectors(frame_features, frame_timestamps, sample_interval, output_dir);
        }
        
        std::cout << "\nScene detection completed successfully!" << std::endl;
        
    } catch (const c10::Error& e) {
        std::cerr << "LibTorch error: " << e.msg() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Standard error: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return -1;
    }
    
    return 0;
}
