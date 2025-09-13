#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/flags/usage.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <torch/script.h>
#include <torch/torch.h>
#include <unistd.h>
#include <vector>

// Define command-line flags
ABSL_FLAG(std::string, video, "", "Path to input video file (required)");
ABSL_FLAG(
    std::string, model_version, "vith16plus",
    "DINOv3 model version (vitl16, vits16, vits16plus, vitb16, vith16plus)");
ABSL_FLAG(double, threshold, 0.85,
          "Similarity threshold for scene change detection (0.0-1.0)");
ABSL_FLAG(int, min_scene_length, 30, "Minimum frames per scene");
ABSL_FLAG(int, sample_interval, 5, "Sample every Nth frame for efficiency");
ABSL_FLAG(int, max_frames, 0,
          "Maximum number of frames to process (0 = no limit)");
ABSL_FLAG(std::string, output_dir, "results", "Output directory for results");
ABSL_FLAG(bool, verbose, false, "Enable verbose output");
ABSL_FLAG(bool, save_features, true, "Save feature vectors to CSV file");
ABSL_FLAG(bool, extract_scenes, false,
          "Extract each scene as a separate video file");
ABSL_FLAG(int, batch_size, 8, "Batch size for processing frames (1-32)");

// Structure to store scene information
struct SceneInfo {
  int start_frame;
  int end_frame;
  double start_time;
  double end_time;
  std::vector<float> representative_features;
  std::string scene_type; // "court_view", "back_side", "other_angle", etc.
};

// Function to preprocess frame for DINOv3
torch::Tensor preprocess_frame(const cv::Mat &frame,
                               const torch::Device &device) {
  // Resize frame to 224x224
  cv::Mat resized_frame;
  cv::resize(frame, resized_frame, cv::Size(224, 224));

  // Convert BGR to RGB
  cv::Mat rgb_frame;
  cv::cvtColor(resized_frame, rgb_frame, cv::COLOR_BGR2RGB);

  // Convert to float and normalize to [0, 1]
  cv::Mat float_frame;
  rgb_frame.convertTo(float_frame, CV_32F, 1.0 / 255.0);

  // Convert OpenCV Mat to torch tensor
  auto tensor =
      torch::from_blob(float_frame.data, {224, 224, 3}, torch::kFloat32);

  // Permute dimensions from (H, W, C) to (C, H, W) and add batch dimension
  tensor = tensor.permute({2, 0, 1}).unsqueeze(0);

  // Move to device
  tensor = tensor.to(device);

  return tensor;
}

// Function to preprocess multiple frames for batch processing
torch::Tensor preprocess_frames_batch(const std::vector<cv::Mat> &frames,
                                      const torch::Device &device) {
  if (frames.empty()) {
    throw std::runtime_error("Empty frame batch provided");
  }
  
  int batch_size = frames.size();
  
  // Create a tensor to hold the batch [batch_size, 3, 224, 224]
  auto batch_tensor = torch::zeros({batch_size, 3, 224, 224}, torch::kFloat32);
  
  for (int i = 0; i < batch_size; ++i) {
    // Resize frame to 224x224
    cv::Mat resized_frame;
    cv::resize(frames[i], resized_frame, cv::Size(224, 224));

    // Convert BGR to RGB
    cv::Mat rgb_frame;
    cv::cvtColor(resized_frame, rgb_frame, cv::COLOR_BGR2RGB);

    // Convert to float and normalize to [0, 1]
    cv::Mat float_frame;
    rgb_frame.convertTo(float_frame, CV_32F, 1.0 / 255.0);

    // Convert OpenCV Mat to torch tensor
    auto tensor =
        torch::from_blob(float_frame.data, {224, 224, 3}, torch::kFloat32);

    // Permute dimensions from (H, W, C) to (C, H, W)
    tensor = tensor.permute({2, 0, 1});
    
    // Copy to batch tensor
    batch_tensor[i] = tensor.clone();
  }

  // Move to device
  batch_tensor = batch_tensor.to(device);
  
  return batch_tensor;
}

// Function to compute cosine similarity between two feature vectors
double cosine_similarity(const std::vector<float> &vec1,
                         const std::vector<float> &vec2) {
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

// Structure to store cluster information
struct ClusterInfo {
  int cluster_id;
  std::vector<int> frame_indices;
  std::vector<float> centroid;
  std::string scene_type;
  double avg_similarity;
};

// Function to perform K-means clustering on feature vectors
std::vector<ClusterInfo>
cluster_features(const std::vector<std::vector<float>> &frame_features,
                 std::vector<int> &assignments, int k = 5,
                 int max_iterations = 100, bool verbose = false) {
  if (frame_features.empty())
    return {};

  int n_features = frame_features[0].size();
  int n_frames = frame_features.size();

  if (verbose) {
    std::cout << "Clustering " << n_frames << " frames into " << k
              << " clusters..." << std::endl;
  }

  // Initialize centroids randomly
  std::vector<std::vector<float>> centroids(
      k, std::vector<float>(n_features, 0.0));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, n_frames - 1);

  for (int i = 0; i < k; ++i) {
    int random_idx = dis(gen);
    centroids[i] = frame_features[random_idx];
  }

  assignments.resize(n_frames, 0);

  // K-means iterations
  for (int iter = 0; iter < max_iterations; ++iter) {
    bool changed = false;

    // Assign each frame to nearest centroid
    for (int i = 0; i < n_frames; ++i) {
      double best_similarity = -1.0;
      int best_cluster = 0;

      for (int c = 0; c < k; ++c) {
        double similarity = cosine_similarity(frame_features[i], centroids[c]);
        if (similarity > best_similarity) {
          best_similarity = similarity;
          best_cluster = c;
        }
      }

      if (assignments[i] != best_cluster) {
        assignments[i] = best_cluster;
        changed = true;
      }
    }

    if (!changed)
      break;

    // Update centroids
    for (int c = 0; c < k; ++c) {
      std::fill(centroids[c].begin(), centroids[c].end(), 0.0);
      int count = 0;

      for (int i = 0; i < n_frames; ++i) {
        if (assignments[i] == c) {
          for (int f = 0; f < n_features; ++f) {
            centroids[c][f] += frame_features[i][f];
          }
          count++;
        }
      }

      if (count > 0) {
        for (int f = 0; f < n_features; ++f) {
          centroids[c][f] /= count;
        }
      }
    }
  }

  // Create cluster info structures
  std::vector<ClusterInfo> clusters(k);
  for (int c = 0; c < k; ++c) {
    clusters[c].cluster_id = c;
    clusters[c].centroid = centroids[c];

    // Collect frame indices for this cluster
    for (int i = 0; i < n_frames; ++i) {
      if (assignments[i] == c) {
        clusters[c].frame_indices.push_back(i);
      }
    }

    // Calculate average similarity within cluster
    double total_similarity = 0.0;
    int similarity_count = 0;
    for (int idx : clusters[c].frame_indices) {
      double sim = cosine_similarity(frame_features[idx], centroids[c]);
      total_similarity += sim;
      similarity_count++;
    }
    clusters[c].avg_similarity =
        similarity_count > 0 ? total_similarity / similarity_count : 0.0;
  }

  // Assign scene types based on cluster size and characteristics
  for (size_t i = 0; i < clusters.size(); ++i) {
    if (i == 0) {
      clusters[i].scene_type =
          "main_court_view"; // Largest cluster = main court view
    } else if (i == 1) {
      clusters[i].scene_type =
          "secondary_angle"; // Second largest = secondary angle
    } else if (clusters[i].frame_indices.size() <
               n_frames * 0.05) { // Less than 5% of frames
      clusters[i].scene_type =
          "transition_scene"; // Small clusters = transitions/special shots
    } else {
      clusters[i].scene_type = "alternate_angle_" + std::to_string(i);
    }
  }

  if (verbose) {
    std::cout << "Clustering results:" << std::endl;
    for (size_t i = 0; i < clusters.size(); ++i) {
      double percentage =
          (double)clusters[i].frame_indices.size() / n_frames * 100.0;
      std::cout << "  Cluster " << i << " (" << clusters[i].scene_type
                << "): " << clusters[i].frame_indices.size() << " frames ("
                << std::fixed << std::setprecision(1) << percentage << "%), "
                << "avg similarity: " << std::setprecision(3)
                << clusters[i].avg_similarity << std::endl;
    }
  }

  return clusters;
}

// Function to classify scene type based on feature vector using clustering
std::string
classify_scene_type_with_clusters(const std::vector<float> &features,
                                  const std::vector<ClusterInfo> &clusters) {
  if (clusters.empty())
    return "unknown";

  double best_similarity = -1.0;
  std::string best_type = "unknown";

  for (const auto &cluster : clusters) {
    double similarity = cosine_similarity(features, cluster.centroid);
    if (similarity > best_similarity) {
      best_similarity = similarity;
      best_type = cluster.scene_type;
    }
  }

  return best_type;
}

// Legacy function for backward compatibility
std::string classify_scene_type(const std::vector<float> &features) {
  // This is now a fallback - the main classification should use clustering
  return "unclassified";
}

// Function to validate model version
bool validate_model_version(const std::string &model_version) {
  const std::vector<std::string> valid_models = {
      "vitl16", "vits16", "vits16plus", "vitb16", "vith16plus"};
  return std::find(valid_models.begin(), valid_models.end(), model_version) !=
         valid_models.end();
}

// Function to validate command-line arguments
bool validate_arguments(const std::string &video_path,
                        const std::string &model_version,
                        double similarity_threshold, int min_scene_length,
                        int sample_interval, int max_frames) {
  // Validate required arguments
  if (video_path.empty()) {
    std::cerr << "Error: --video flag is required!" << std::endl;
    std::cerr << "Use --help for usage information." << std::endl;
    return false;
  }

  // Validate model version
  if (!validate_model_version(model_version)) {
    std::cerr << "Error: Invalid model version '" << model_version << "'!"
              << std::endl;
    std::cerr << "Valid options: vitl16, vits16, vits16plus, vitb16, vith16plus"
              << std::endl;
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

  if (max_frames < 0) {
    std::cerr << "Error: --max_frames must be non-negative!" << std::endl;
    return false;
  }

  return true;
}

// Function to print configuration information
void print_configuration(const std::string &video_path,
                         const std::string &model_version,
                         double similarity_threshold, int min_scene_length,
                         int sample_interval, int max_frames,
                         const std::string &output_dir, bool save_features,
                         bool extract_scenes) {
  std::cout << "DINOv3 Video Scene Detector" << std::endl;
  std::cout << "===========================" << std::endl;
  std::cout << "Video file: " << video_path << std::endl;
  std::cout << "Model version: " << model_version << std::endl;
  std::cout << "Similarity threshold: " << similarity_threshold << std::endl;
  std::cout << "Minimum scene length: " << min_scene_length << " frames"
            << std::endl;
  std::cout << "Sample interval: " << sample_interval << std::endl;
  std::cout << "Max frames to process: "
            << (max_frames == 0 ? "unlimited" : std::to_string(max_frames))
            << std::endl;
  std::cout << "Output directory: " << output_dir << std::endl;
  std::cout << "Save features: " << (save_features ? "yes" : "no") << std::endl;
  std::cout << "Extract scenes: " << (extract_scenes ? "yes" : "no")
            << std::endl;
  std::cout << std::endl;
}

// Function to detect available device
torch::Device detect_device(bool verbose) {
  if (verbose)
    std::cout << "Checking device availability..." << std::endl;
  torch::Device device(torch::kCPU);

  if (torch::hasMPS()) {
    device = torch::Device(torch::kMPS);
    if (verbose)
      std::cout
          << "MPS (Metal Performance Shaders) is available! Using device: MPS"
          << std::endl;
  } else if (torch::hasCUDA()) {
    device = torch::Device(torch::kCUDA);
    if (verbose)
      std::cout << "CUDA is available! Using device: CUDA" << std::endl;
  } else {
    if (verbose)
      std::cout << "Using CPU device" << std::endl;
  }

  return device;
}

// Function to load DINOv3 model
torch::jit::script::Module load_model(const std::string &model_version,
                                      const torch::Device &device,
                                      bool verbose) {
  if (verbose)
    std::cout << "Loading DINOv3 model (" << model_version << ")..."
              << std::endl;

  // Print current working directory
  char cwd[1024];
  if (getcwd(cwd, sizeof(cwd)) != nullptr) {
    if (verbose)
      std::cout << "Current working directory: " << cwd << std::endl;
  }

  // Try multiple possible model paths for the specified model version
  std::string model_filename = "dinov3_" + model_version + "_traced.pt";
  std::vector<std::string> possible_paths = {
      "models/" + model_filename,   // From models subdirectory (preferred)
      "dinov3/" + model_filename,   // From parent directory
      model_filename,               // From dinov3 directory
      "../dinov3/" + model_filename // From other subdirectory
  };

  torch::jit::script::Module model;
  std::string successful_path;

  for (const auto &model_path : possible_paths) {
    if (verbose)
      std::cout << "Trying model path: " << model_path << std::endl;

    // Check if file exists
    std::ifstream file_check(model_path);
    if (!file_check.good()) {
      if (verbose)
        std::cout << "File does not exist: " << model_path << std::endl;
      file_check.close();
      continue;
    }
    file_check.close();

    // Get absolute path
    char abs_path[1024];
    if (realpath(model_path.c_str(), abs_path) != nullptr) {
      if (verbose)
        std::cout << "Absolute path: " << abs_path << std::endl;
    }

    try {
      model = torch::jit::load(model_path);
      successful_path = model_path;
      break;
    } catch (const c10::Error &e) {
      if (verbose)
        std::cout << "Failed to load from: " << model_path
                  << " - Error: " << e.what() << std::endl;
      continue;
    }
  }

  if (successful_path.empty()) {
    std::string error_msg = "Could not load DINOv3 model (" + model_version +
                            ") from any of the attempted paths.\n";
    error_msg += "Make sure you have traced the model using: python "
                 "DINOv3_trace.py --model_version=" +
                 model_version;
    throw std::runtime_error(error_msg);
  }

  if (verbose)
    std::cout << "Successfully loaded model from: " << successful_path
              << std::endl;

  model.eval();
  model.to(device);

  if (verbose)
    std::cout << "Model loaded successfully!" << std::endl;
  return model;
}

// Function to get video properties
void get_video_properties(cv::VideoCapture &cap, bool verbose) {
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
std::pair<std::vector<std::vector<float>>, std::vector<double>>
process_video_frames(cv::VideoCapture &cap, torch::jit::script::Module &model,
                     const torch::Device &device, int sample_interval,
                     int max_frames, int batch_size, bool verbose) {

  std::vector<std::vector<float>> frame_features;
  std::vector<double> frame_timestamps;

  // Get total frames for progress calculation
  int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
  double fps = cap.get(cv::CAP_PROP_FPS);

  // Apply max_frames limit if specified
  int effective_total_frames = total_frames;
  if (max_frames > 0 && max_frames < total_frames) {
    effective_total_frames = max_frames;
  }

  int estimated_processed_frames =
      (effective_total_frames + sample_interval - 1) / sample_interval;

  cv::Mat frame;
  int frame_count = 0;
  int processed_frames = 0;

  if (verbose) {
    std::cout << "\nProcessing video frames..." << std::endl;
    std::cout << "Total frames: " << total_frames;
    if (max_frames > 0 && max_frames < total_frames) {
      std::cout << " (limited to " << max_frames << ")";
    }
    std::cout << ", Sample interval: " << sample_interval << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Estimated frames to process: " << estimated_processed_frames
              << std::endl;
  }

  // Validate batch size
  if (batch_size < 1 || batch_size > 32) {
    std::cerr << "Warning: batch_size should be between 1 and 32. Using default value of 8." << std::endl;
    batch_size = 8;
  }

  torch::NoGradGuard no_grad;
  auto start_time = std::chrono::high_resolution_clock::now();

  // Batch processing variables
  std::vector<cv::Mat> frame_batch;
  std::vector<double> timestamp_batch;
  std::vector<int> frame_number_batch;

  while (cap.read(frame)) {
    frame_count++;

    // Check if we've reached the maximum frame limit
    if (max_frames > 0 && frame_count > max_frames) {
      break;
    }

    // Sample frames at regular intervals for efficiency
    if (frame_count % sample_interval != 0) {
      continue;
    }

    double timestamp = frame_count / fps;

    // Add frame to batch
    frame_batch.push_back(frame.clone());
    timestamp_batch.push_back(timestamp);
    frame_number_batch.push_back(frame_count);

    // Process batch when it's full or at the end of video
    if (frame_batch.size() >= batch_size || (max_frames > 0 && frame_count >= max_frames)) {
      // Preprocess batch
      auto batch_tensor = preprocess_frames_batch(frame_batch, device);

      // Run batch inference
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(batch_tensor);
      auto output = model.forward(inputs);

      if (output.isTensor()) {
        auto output_tensor = output.toTensor();
        auto cpu_tensor = output_tensor.cpu();

        // Process each result in the batch
        for (int i = 0; i < frame_batch.size(); ++i) {
          // Convert to std::vector<float>
          std::vector<float> features;
          features.reserve(cpu_tensor.size(1));
          for (int j = 0; j < cpu_tensor.size(1); ++j) {
            features.push_back(cpu_tensor[i][j].item<float>());
          }

          frame_features.push_back(features);
          frame_timestamps.push_back(timestamp_batch[i]);
          processed_frames++;

          // Enhanced progress reporting
          if (verbose) {
            // Calculate progress percentage
            double progress_percent =
                (double)processed_frames / estimated_processed_frames * 100.0;

            // Show detailed progress every batch or every 5 frames
            if (processed_frames <= 3 || processed_frames % 5 == 0 ||
                i == frame_batch.size() - 1) {
              auto current_time = std::chrono::high_resolution_clock::now();
              auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                  current_time - start_time);
              double avg_time_per_frame =
                  elapsed.count() / (double)processed_frames;
              int eta_ms = (estimated_processed_frames - processed_frames) *
                           avg_time_per_frame;

              // Create progress bar string
              std::string progress_bar =
                  std::string((int)(progress_percent / 5), '=') +
                  std::string(20 - (int)(progress_percent / 5), ' ');

              printf("\rProgress: [%s] %.1f%% (%d/%d) Frame: %d/%d Time: %.2fs "
                     "Avg: %.0fms/frame Batch: %zu ETA: %ds    ",
                     progress_bar.c_str(), progress_percent, processed_frames,
                     estimated_processed_frames, frame_number_batch[i],
                     effective_total_frames, timestamp_batch[i], avg_time_per_frame,
                     frame_batch.size(), eta_ms / 1000);
              fflush(stdout);
            }
          }
        }
      }

      // Clear batch for next iteration
      frame_batch.clear();
      timestamp_batch.clear();
      frame_number_batch.clear();
    }
  }

  // Process any remaining frames in the batch
  if (!frame_batch.empty()) {
    // Preprocess batch
    auto batch_tensor = preprocess_frames_batch(frame_batch, device);

    // Run batch inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(batch_tensor);
    auto output = model.forward(inputs);

    if (output.isTensor()) {
      auto output_tensor = output.toTensor();
      auto cpu_tensor = output_tensor.cpu();

      // Process each result in the batch
      for (int i = 0; i < frame_batch.size(); ++i) {
        // Convert to std::vector<float>
        std::vector<float> features;
        features.reserve(cpu_tensor.size(1));
        for (int j = 0; j < cpu_tensor.size(1); ++j) {
          features.push_back(cpu_tensor[i][j].item<float>());
        }

        frame_features.push_back(features);
        frame_timestamps.push_back(timestamp_batch[i]);
        processed_frames++;
      }
    }
  }

  if (verbose) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    printf("\n\nFinished processing %d frames in %.1f seconds.\n",
           processed_frames, total_elapsed.count() / 1000.0);
    printf("Average processing time: %.1f ms/frame\n",
           (double)total_elapsed.count() / processed_frames);
  }

  return {frame_features, frame_timestamps};
}

// Function to print scene detection results
void print_scene_results(const std::vector<SceneInfo> &scenes,
                         int sample_interval) {
  std::cout << "\n=== SCENE DETECTION RESULTS ===" << std::endl;
  std::cout << "Found " << scenes.size() << " scenes:" << std::endl;
  std::cout << std::endl;

  for (size_t i = 0; i < scenes.size(); ++i) {
    const auto &scene = scenes[i];
    int scene_frames = scene.end_frame - scene.start_frame + 1;
    double scene_duration = scene.end_time - scene.start_time;

    std::cout << "Scene " << (i + 1) << ":" << std::endl;
    std::cout << "  Time: " << std::fixed << std::setprecision(2)
              << scene.start_time << "s - " << scene.end_time << "s ("
              << scene_duration << "s)" << std::endl;
    std::cout << "  Frames: " << scene.start_frame << " - " << scene.end_frame
              << " (" << scene_frames << " frames)" << std::endl;
    std::cout << "  Type: " << scene.scene_type << std::endl;
    std::cout << std::endl;
  }
}

// Function to save detailed results to file
void save_detailed_results(const std::vector<SceneInfo> &scenes,
                           const std::string &video_path, int total_frames,
                           double fps, int width, int height,
                           int processed_frames, int sample_interval,
                           double similarity_threshold, int min_scene_length,
                           const std::string &output_dir) {

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
    file << "Minimum scene length: " << min_scene_length << " frames"
         << std::endl;
    file << std::endl;

    file << "Detected Scenes:" << std::endl;
    file << "===============" << std::endl;

    for (size_t i = 0; i < scenes.size(); ++i) {
      const auto &scene = scenes[i];
      int scene_frames = scene.end_frame - scene.start_frame + 1;
      double scene_duration = scene.end_time - scene.start_time;

      file << "Scene " << (i + 1) << ":" << std::endl;
      file << "  Start time: " << std::fixed << std::setprecision(2)
           << scene.start_time << "s" << std::endl;
      file << "  End time: " << std::fixed << std::setprecision(2)
           << scene.end_time << "s" << std::endl;
      file << "  Duration: " << std::fixed << std::setprecision(2)
           << scene_duration << "s" << std::endl;
      file << "  Start frame: " << scene.start_frame << std::endl;
      file << "  End frame: " << scene.end_frame << std::endl;
      file << "  Frame count: " << scene_frames << std::endl;
      file << "  Scene type: " << scene.scene_type << std::endl;
      file << std::endl;
    }

    file.close();
    std::cout << "Detailed results saved to: " << output_file << std::endl;
  }
}

// Function to save feature vectors to CSV
void save_feature_vectors(const std::vector<std::vector<float>> &frame_features,
                          const std::vector<int> &assignments,
                          const std::vector<double> &frame_timestamps,
                          int sample_interval, const std::string &output_dir) {

  std::string features_file = output_dir + "/frame_features.csv";
  std::ofstream features_csv(features_file);
  if (features_csv.is_open()) {
    // Header
    features_csv << "frame,timestamp,assignment";
    for (int i = 0; i < frame_features[0].size(); ++i) {
      features_csv << ",feature_" << i;
    }
    features_csv << std::endl;

    // Data
    for (size_t i = 0; i < frame_features.size(); ++i) {
      features_csv << i * sample_interval << "," << frame_timestamps[i];
      features_csv << "," << assignments[i];
      for (float feature : frame_features[i]) {
        features_csv << "," << feature;
      }
      features_csv << std::endl;
    }

    features_csv.close();
    std::cout << "Feature vectors saved to: " << features_file << std::endl;
  }
}

// Function to extract scene videos
void extract_scene_videos(const std::vector<SceneInfo> &scenes,
                          const std::string &video_path,
                          const std::string &output_dir, bool verbose) {
  if (scenes.empty()) {
    if (verbose)
      std::cout << "No scenes to extract." << std::endl;
    return;
  }

  if (verbose) {
    std::cout << "\nExtracting scene videos..." << std::endl;
    std::cout << "Input video: " << video_path << std::endl;
    std::cout << "Output directory: " << output_dir << std::endl;
  }

  // Open the original video
  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    std::cerr << "Error: Could not open video file for scene extraction: "
              << video_path << std::endl;
    return;
  }

  // Get video properties
  double fps = cap.get(cv::CAP_PROP_FPS);
  int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  int fourcc = cap.get(cv::CAP_PROP_FOURCC);

  if (verbose) {
    std::cout << "Video properties - FPS: " << fps << ", Resolution: " << width
              << "x" << height << std::endl;
  }

  // Extract each scene
  for (size_t scene_idx = 0; scene_idx < scenes.size(); ++scene_idx) {
    const auto &scene = scenes[scene_idx];

    // Create output filename
    std::string scene_filename = output_dir + "/scene_" +
                                 std::to_string(scene_idx + 1) + "_" +
                                 scene.scene_type + ".mp4";

    if (verbose) {
      printf("Extracting scene %zu: frames %d-%d (%.2fs-%.2fs) -> %s\n",
             scene_idx + 1, scene.start_frame, scene.end_frame,
             scene.start_time, scene.end_time, scene_filename.c_str());
    }

    // Create video writer with avc1 codec (H.264 compatible)
    cv::VideoWriter writer(scene_filename,
                           cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps,
                           cv::Size(width, height));

    if (!writer.isOpened()) {
      std::cerr << "Error: Could not create output video: " << scene_filename
                << std::endl;
      continue;
    }

    // Seek to start frame
    cap.set(cv::CAP_PROP_POS_FRAMES, scene.start_frame);

    cv::Mat frame;
    int frames_written = 0;
    int target_frames = scene.end_frame - scene.start_frame + 1;

    // Extract frames for this scene
    while (frames_written < target_frames && cap.read(frame)) {
      writer.write(frame);
      frames_written++;

      if (verbose && frames_written % 50 == 0) {
        printf("\r  Writing frame %d/%d", frames_written, target_frames);
        fflush(stdout);
      }
    }

    if (verbose) {
      printf("\r  Completed: %d frames written\n", frames_written);
    }

    writer.release();
  }

  cap.release();

  if (verbose) {
    std::cout << "Scene extraction completed! Extracted " << scenes.size()
              << " scene videos." << std::endl;
  }
}

// Function to extract cluster videos (one file per cluster)
void extract_cluster_videos(
    const std::vector<ClusterInfo> &clusters,
    const std::vector<std::vector<float>> &frame_features,
    const std::vector<int> &assignments, 
    torch::jit::script::Module& model, 
    const torch::Device& device,
    const std::string &video_path,
    const std::string &output_dir, int sample_interval, int max_frames,
    int batch_size, bool verbose) {

  if (verbose) {
    std::cout << "\nExtracting cluster videos (one file per camera angle)..."
              << std::endl;
    std::cout << "Input video: " << video_path << std::endl;
    std::cout << "Output directory: " << output_dir << std::endl;
  }

  // Open the input video
  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    std::cerr << "Error: Could not open video file for cluster extraction: "
              << video_path << std::endl;
    return;
  }

  // Get video properties
  double fps = cap.get(cv::CAP_PROP_FPS);
  int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);

  // Apply max_frames limit if specified
  int effective_total_frames = total_frames;
  if (max_frames > 0 && max_frames < total_frames) {
    effective_total_frames = max_frames;
  }

  if (verbose) {
    std::cout << "Video properties - FPS: " << std::fixed
              << std::setprecision(2) << fps << ", Resolution: " << width << "x"
              << height << std::endl;
    std::cout << "Processing frames: 0 to " << effective_total_frames
              << std::endl;
  }

  // Create video writers for each cluster
  std::vector<cv::VideoWriter> writers(clusters.size());
  std::vector<int> frames_written(clusters.size(), 0);

  for (size_t i = 0; i < clusters.size(); ++i) {
    std::string cluster_filename = output_dir + "/cluster_" +
                                   std::to_string(i) + "_" +
                                   clusters[i].scene_type + ".mp4";

    writers[i].open(cluster_filename,
                    cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps,
                    cv::Size(width, height));

    if (!writers[i].isOpened()) {
      std::cerr << "Error: Could not create output video: " << cluster_filename
                << std::endl;
      return;
    }

    if (verbose) {
      std::cout << "Created writer for cluster " << i << " ("
                << clusters[i].scene_type << "): " << cluster_filename
                << std::endl;
    }
  }

  // Process each frame and assign to appropriate cluster
  cv::Mat frame;
  int frame_count = 0;
  int current_cluster_id = assignments[0]; // Default to first cluster

  // For progress reporting every 2 seconds
  auto start_time = std::chrono::high_resolution_clock::now();
  auto last_progress_time = start_time;

  while (cap.read(frame) && frame_count < effective_total_frames) {
    int feature_index = frame_count / sample_interval;
    
    // Determine current cluster ID for this frame
    if (frame_count % sample_interval == 0 && feature_index < assignments.size()) {
      // This frame has features - use the assignment
      current_cluster_id = assignments[feature_index];
    }
    // For frames without features, current_cluster_id remains the same as last known
    
    // Check if we need to process intermediate frames
    int next_feature_index = (frame_count + sample_interval) / sample_interval;
    int next_cluster_id = current_cluster_id; // Default to current
    if (next_feature_index < assignments.size()) {
      next_cluster_id = assignments[next_feature_index];
    }

    if (frame_count % sample_interval != 0 && current_cluster_id != next_cluster_id) {
      // Calculate features for the next N frames that have no features
      // calculated and attribute them to the clusters

      // N is the number of frames to look ahead for feature calculation
      const int N = sample_interval - 1; // You can adjust N as needed

      // Only process if this is a transition point between clusters
      // Calculate features for the next N frames (or until end of video)
      std::vector<std::vector<float>> new_features;
      std::vector<int> new_assignments;
      std::vector<cv::Mat> new_frames;
      new_frames.push_back(frame);

      // Read and process next N - 1 frames
      for (int n = 0; n < N - 1; ++n) {
        cv::Mat next_frame;
        if (!cap.read(next_frame)) {
          break;
        }
        new_frames.push_back(next_frame);
      }
      // Process intermediate frames in batches
      torch::NoGradGuard no_grad;
      
      // Process frames in batches
      for (size_t start_idx = 0; start_idx < new_frames.size(); start_idx += batch_size) {
        size_t end_idx = std::min(start_idx + batch_size, new_frames.size());
        std::vector<cv::Mat> batch_frames(new_frames.begin() + start_idx, new_frames.begin() + end_idx);
        
        // Preprocess batch and extract features
        auto batch_tensor = preprocess_frames_batch(batch_frames, device);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(batch_tensor);
        auto output = model.forward(inputs);

        if (output.isTensor()) {
          auto output_tensor = output.toTensor();
          auto cpu_tensor = output_tensor.cpu();

          // Process each result in the batch
          for (size_t i = 0; i < batch_frames.size(); ++i) {
            std::vector<float> features;
            features.reserve(cpu_tensor.size(1));
            for (int j = 0; j < cpu_tensor.size(1); ++j) {
              features.push_back(cpu_tensor[i][j].item<float>());
            }
            new_features.push_back(features);

            // Assign to nearest cluster
            double best_similarity = -1.0;
            int best_cluster = 0;
            for (size_t c = 0; c < clusters.size(); ++c) {
              double similarity = cosine_similarity(features, clusters[c].centroid);
              if (similarity > best_similarity) {
                best_similarity = similarity;
                best_cluster = c;
              }
            }
            new_assignments.push_back(best_cluster);
          }
        } else {
          // If feature extraction fails, assign to previous cluster for all frames in batch
          for (size_t i = 0; i < batch_frames.size(); ++i) {
            new_assignments.push_back(current_cluster_id);
          }
        }
      }

      // Write the new frames to their assigned cluster videos
      for (size_t n = 0; n < new_frames.size(); ++n) {
        int assigned_cluster = new_assignments[n];
        writers[assigned_cluster].write(new_frames[n]);
        frames_written[assigned_cluster]++;
        frame_count++;
      }
      continue;
    }
    // Write frame to the appropriate cluster video
    writers[current_cluster_id].write(frame);
    frames_written[current_cluster_id]++;

    frame_count++;

    // Progress reporting every 10 seconds
    if (verbose) {
      auto current_time = std::chrono::high_resolution_clock::now();
      auto elapsed_since_last =
          std::chrono::duration_cast<std::chrono::seconds>(current_time -
                                                           last_progress_time);

      if (elapsed_since_last.count() >= 2 ||
          frame_count == effective_total_frames) {
        auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            current_time - start_time);
        double progress_percent =
            (double)frame_count / effective_total_frames * 100.0;
        double avg_fps = frame_count / (total_elapsed.count() +
                                        1); // +1 to avoid division by zero

        printf(
            "\rProcessed frame %d/%d (%.1f%%) - %.1f fps - %d seconds elapsed",
            frame_count, effective_total_frames, progress_percent, avg_fps,
            (int)total_elapsed.count());
        fflush(stdout);

        last_progress_time = current_time;
      }
    }
  }

  // Close all writers
  for (size_t i = 0; i < writers.size(); ++i) {
    writers[i].release();
  }

  cap.release();

  if (verbose) {
    printf("\nCluster extraction completed!\n");
    for (size_t i = 0; i < clusters.size(); ++i) {
      std::cout << "Cluster " << i << " (" << clusters[i].scene_type
                << "): " << frames_written[i] << " frames written" << std::endl;
    }
  }
}

int main(int argc, char *argv[]) {
  // Set program name for help messages
  absl::SetProgramUsageMessage("DINOv3 Video Scene Detector - Detect scene "
                               "changes and camera angles in videos");

  // Parse command-line flags
  absl::ParseCommandLine(argc, argv);

  // Get flag values
  std::string video_path = absl::GetFlag(FLAGS_video);
  std::string model_version = absl::GetFlag(FLAGS_model_version);
  double similarity_threshold = absl::GetFlag(FLAGS_threshold);
  int min_scene_length = absl::GetFlag(FLAGS_min_scene_length);
  int sample_interval = absl::GetFlag(FLAGS_sample_interval);
  int max_frames = absl::GetFlag(FLAGS_max_frames);
  std::string output_dir = absl::GetFlag(FLAGS_output_dir);
  bool verbose = absl::GetFlag(FLAGS_verbose);
  bool save_features = absl::GetFlag(FLAGS_save_features);
  bool extract_scenes = absl::GetFlag(FLAGS_extract_scenes);

  // Validate arguments
  if (!validate_arguments(video_path, model_version, similarity_threshold,
                          min_scene_length, sample_interval, max_frames)) {
    return -1;
  }

  // Print configuration
  if (verbose) {
    print_configuration(video_path, model_version, similarity_threshold,
                        min_scene_length, sample_interval, max_frames,
                        output_dir, save_features, extract_scenes);
  }

  try {
    // Detect available device
    torch::Device device = detect_device(verbose);

    // Load DINOv3 model
    torch::jit::script::Module model =
        load_model(model_version, device, verbose);

    // Open video file
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
      std::cerr << "Error: Could not open video file: " << video_path
                << std::endl;
      return -1;
    }

    // Get video properties
    get_video_properties(cap, verbose);

    // Process video frames and extract features
    int batch_size = absl::GetFlag(FLAGS_batch_size);
    auto [frame_features, frame_timestamps] = process_video_frames(
        cap, model, device, sample_interval, max_frames, batch_size, verbose);

    cap.release();

    std::vector<int> assignments;

    // Perform clustering on all features to identify camera angles
    std::vector<ClusterInfo> clusters =
        cluster_features(frame_features, assignments, 5, 100, verbose);

    // Save feature vectors if requested
    if (save_features) {
      save_feature_vectors(frame_features, assignments, frame_timestamps,
                           sample_interval, output_dir);
    }

     // Extract cluster videos if requested (one file per camera angle)
     if (extract_scenes) {
       extract_cluster_videos(clusters, frame_features, assignments, model, device,
                              video_path, output_dir, sample_interval, max_frames, batch_size, verbose);
     }

    std::cout << "\nScene detection completed successfully!" << std::endl;

  } catch (const c10::Error &e) {
    std::cerr << "LibTorch error: " << e.msg() << std::endl;
    return -1;
  } catch (const std::exception &e) {
    std::cerr << "Standard error: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cerr << "Unknown error occurred" << std::endl;
    return -1;
  }

  return 0;
}
