#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <filesystem>

torch::Tensor preprocess_image(const cv::Mat& image, const torch::Device& device) {
    // Resize image to 224x224
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(224, 224));
    
    // Convert BGR to RGB
    cv::Mat rgb_image;
    cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);
    
    // Convert to float and normalize to [0, 1]
    cv::Mat float_image;
    rgb_image.convertTo(float_image, CV_32F, 1.0/255.0);
    
    // Convert OpenCV Mat to torch tensor
    auto tensor = torch::from_blob(float_image.data, {224, 224, 3}, torch::kFloat32);
    
    // Permute dimensions from (H, W, C) to (C, H, W) and add batch dimension
    tensor = tensor.permute({2, 0, 1}).unsqueeze(0);
    
    // Move to device
    tensor = tensor.to(device);
    
    return tensor;
}

int main() {
    try {
        // Check for MPS availability
        std::cout << "Checking device availability..." << std::endl;
        torch::Device device(torch::kCPU);
        
        if (torch::hasMPS()) {
            device = torch::Device(torch::kMPS);
            std::cout << "MPS (Metal Performance Shaders) is available! Using device: MPS" << std::endl;
        } else if (torch::hasCUDA()) {
            device = torch::Device(torch::kCUDA);
            std::cout << "CUDA is available! Using device: CUDA" << std::endl;
        } else {
            std::cout << "Using CPU device" << std::endl;
        }

        // Load the traced model
        std::cout << "Loading traced DINOv3 model..." << std::endl;
        torch::jit::script::Module module;
        module = torch::jit::load("dinov3_vits16_traced.pt");
        module.eval();
        module.to(device);
        std::cout << "Model loaded successfully and moved to device!" << std::endl;

        // Create a test image if no image is provided
        std::cout << "Creating test image..." << std::endl;
        cv::Mat test_image = cv::Mat::zeros(480, 640, CV_8UC3);
        
        // Draw some shapes for testing
        cv::rectangle(test_image, cv::Point(100, 100), cv::Point(300, 300), cv::Scalar(255, 0, 0), -1);
        cv::circle(test_image, cv::Point(400, 200), 80, cv::Scalar(0, 255, 0), -1);
        cv::line(test_image, cv::Point(50, 400), cv::Point(550, 400), cv::Scalar(0, 0, 255), 5);
        
        std::cout << "Test image created with size: " << test_image.size() << std::endl;

        // Preprocess the image
        std::cout << "Preprocessing image..." << std::endl;
        auto input_tensor = preprocess_image(test_image, device);
        std::cout << "Input tensor shape: " << input_tensor.sizes() << " on device: " << input_tensor.device() << std::endl;

        // Run inference
        std::cout << "Running inference..." << std::endl;
        torch::NoGradGuard no_grad;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        auto output = module.forward(inputs);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        if (output.isTensor()) {
            auto output_tensor = output.toTensor();
            std::cout << "Output tensor shape: " << output_tensor.sizes() << std::endl;
            std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
            
            // Print statistics
            float min_val = output_tensor.min().item<float>();
            float max_val = output_tensor.max().item<float>();
            float mean_val = output_tensor.mean().item<float>();
            float std_val = output_tensor.std().item<float>();
            
            std::cout << "Feature vector statistics:" << std::endl;
            std::cout << "  Min: " << min_val << std::endl;
            std::cout << "  Max: " << max_val << std::endl;
            std::cout << "  Mean: " << mean_val << std::endl;
            std::cout << "  Std: " << std_val << std::endl;
            
            // Print first few values
            std::cout << "First 10 feature values:" << std::endl;
            auto output_accessor = output_tensor.accessor<float, 2>();
            for (int i = 0; i < 10 && i < output_tensor.size(1); ++i) {
                std::cout << "  [" << i << "]: " << output_accessor[0][i] << std::endl;
            }
            
            // Save feature vector to file
            std::cout << "Saving feature vector to file..." << std::endl;
            auto cpu_tensor = output_tensor.cpu();
            std::ofstream file("feature_vector.txt");
            if (file.is_open()) {
                file << "DINOv3 Feature Vector (4096 dimensions)" << std::endl;
                file << "Min: " << min_val << ", Max: " << max_val << ", Mean: " << mean_val << ", Std: " << std_val << std::endl;
                file << "Values:" << std::endl;
                for (int i = 0; i < cpu_tensor.size(1); ++i) {
                    file << cpu_tensor[0][i].item<float>();
                    if (i < cpu_tensor.size(1) - 1) file << ", ";
                }
                file.close();
                std::cout << "Feature vector saved to feature_vector.txt" << std::endl;
            }
        }

        // Performance benchmark with image processing
        std::cout << "\nRunning performance benchmark with image processing..." << std::endl;
        int num_runs = 10;
        
        // Warm up
        for (int i = 0; i < 3; ++i) {
            std::vector<torch::jit::IValue> warmup_inputs;
            warmup_inputs.push_back(input_tensor);
            module.forward(warmup_inputs);
        }
        
        // Benchmark
        start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_runs; ++i) {
            std::vector<torch::jit::IValue> benchmark_inputs;
            benchmark_inputs.push_back(input_tensor);
            module.forward(benchmark_inputs);
        }
        end_time = std::chrono::high_resolution_clock::now();
        
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto avg_duration = total_duration.count() / num_runs;
        
        std::cout << "Average inference time over " << num_runs << " runs: " << avg_duration << " ms" << std::endl;
        std::cout << "Throughput: " << (1000.0 / avg_duration) << " images/second" << std::endl;

        std::cout << "\nDINOv3 image processing completed successfully!" << std::endl;
        std::cout << "The model can now process real images and extract feature vectors." << std::endl;
        
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
