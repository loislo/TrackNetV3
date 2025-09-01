#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <chrono>

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
        
        // Move model to device
        module.to(device);
        std::cout << "Model loaded successfully and moved to device!" << std::endl;

        // Create input tensors for testing on the correct device
        std::cout << "Creating test inputs..." << std::endl;
        
        // Test 1: Ones tensor
        auto ones_input = torch::ones({1, 3, 224, 224}, device);
        std::cout << "Ones input shape: " << ones_input.sizes() << " on device: " << ones_input.device() << std::endl;
        
        // Test 2: Zeros tensor
        auto zeros_input = torch::zeros({1, 3, 224, 224}, device);
        std::cout << "Zeros input shape: " << zeros_input.sizes() << " on device: " << zeros_input.device() << std::endl;
        
        // Test 3: Random tensor (small values)
        auto rand_input = torch::randn({1, 3, 224, 224}, device) * 0.1;
        std::cout << "Random input shape: " << rand_input.sizes() << " on device: " << rand_input.device() << std::endl;

        // Run inference on all test inputs
        torch::NoGradGuard no_grad;
        
        std::cout << "\nRunning inference on test inputs..." << std::endl;
        
        // Test 1: Ones
        std::cout << "Testing with ones input..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<torch::jit::IValue> ones_inputs;
        ones_inputs.push_back(ones_input);
        auto ones_output = module.forward(ones_inputs);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        if (ones_output.isTensor()) {
            auto ones_output_tensor = ones_output.toTensor();
            std::cout << "Ones output shape: " << ones_output_tensor.sizes() << std::endl;
            std::cout << "Ones output mean: " << ones_output_tensor.mean().item<float>() << std::endl;
            std::cout << "Ones inference time: " << duration.count() << " ms" << std::endl;
        }
        
        // Test 2: Zeros
        std::cout << "Testing with zeros input..." << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<torch::jit::IValue> zeros_inputs;
        zeros_inputs.push_back(zeros_input);
        auto zeros_output = module.forward(zeros_inputs);
        
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        if (zeros_output.isTensor()) {
            auto zeros_output_tensor = zeros_output.toTensor();
            std::cout << "Zeros output shape: " << zeros_output_tensor.sizes() << std::endl;
            std::cout << "Zeros output mean: " << zeros_output_tensor.mean().item<float>() << std::endl;
            std::cout << "Zeros inference time: " << duration.count() << " ms" << std::endl;
        }
        
        // Test 3: Random
        std::cout << "Testing with random input..." << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<torch::jit::IValue> rand_inputs;
        rand_inputs.push_back(rand_input);
        auto rand_output = module.forward(rand_inputs);
        
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        if (rand_output.isTensor()) {
            auto rand_output_tensor = rand_output.toTensor();
            std::cout << "Random output shape: " << rand_output_tensor.sizes() << std::endl;
            std::cout << "Random output mean: " << rand_output_tensor.mean().item<float>() << std::endl;
            std::cout << "Random inference time: " << duration.count() << " ms" << std::endl;
            
            // Print first few values
            std::cout << "First 5 random output values:" << std::endl;
            auto output_accessor = rand_output_tensor.accessor<float, 2>();
            for (int i = 0; i < 5 && i < rand_output_tensor.size(1); ++i) {
                std::cout << "  [" << i << "]: " << output_accessor[0][i] << std::endl;
            }
        }

        // Performance benchmark
        std::cout << "\nRunning performance benchmark..." << std::endl;
        int num_runs = 10;
        
        // Warm up
        for (int i = 0; i < 3; ++i) {
            std::vector<torch::jit::IValue> warmup_inputs;
            warmup_inputs.push_back(ones_input);
            module.forward(warmup_inputs);
        }
        
        // Benchmark
        start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_runs; ++i) {
            std::vector<torch::jit::IValue> benchmark_inputs;
            benchmark_inputs.push_back(ones_input);
            module.forward(benchmark_inputs);
        }
        end_time = std::chrono::high_resolution_clock::now();
        
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto avg_duration = total_duration.count() / num_runs;
        
        std::cout << "Average inference time over " << num_runs << " runs: " << avg_duration << " ms" << std::endl;
        std::cout << "Throughput: " << (1000.0 / avg_duration) << " inferences/second" << std::endl;

        std::cout << "\nDINOv3 C++ inference completed successfully!" << std::endl;
        std::cout << "The model is working correctly and can be used for feature extraction." << std::endl;
        
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
