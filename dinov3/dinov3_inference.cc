#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    try {
        // Load the traced model
        std::cout << "Loading traced DINOv3 model..." << std::endl;
        torch::jit::script::Module module;
        module = torch::jit::load("dinov3_vits16_traced.pt");
        module.eval();
        std::cout << "Model loaded successfully!" << std::endl;

        // Create a dummy input tensor using ones instead of randn for stability
        // Shape: (batch_size, channels, height, width) = (1, 3, 224, 224)
        auto input_tensor = torch::ones({1, 3, 224, 224});
        std::cout << "Input tensor shape: " << input_tensor.sizes() << std::endl;

        // Run inference with proper error handling
        std::cout << "Running inference..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        torch::NoGradGuard no_grad;
        
        // Create input vector
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        // Run forward pass
        auto output = module.forward(inputs);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Inference completed in " << duration.count() << " ms" << std::endl;
        
        // Check if output is a tensor
        if (output.isTensor()) {
            auto output_tensor = output.toTensor();
            std::cout << "Output tensor shape: " << output_tensor.sizes() << std::endl;
            
            // Print some statistics about the output
            float min_val = output_tensor.min().item<float>();
            float max_val = output_tensor.max().item<float>();
            float mean_val = output_tensor.mean().item<float>();
            
            std::cout << "Output statistics:" << std::endl;
            std::cout << "  Min: " << min_val << std::endl;
            std::cout << "  Max: " << max_val << std::endl;
            std::cout << "  Mean: " << mean_val << std::endl;
            
            // Print first few values
            std::cout << "First 10 output values:" << std::endl;
            auto output_accessor = output_tensor.accessor<float, 2>();
            for (int i = 0; i < 10 && i < output_tensor.size(1); ++i) {
                std::cout << "  [" << i << "]: " << output_accessor[0][i] << std::endl;
            }
        } else {
            std::cout << "Output is not a tensor. Output type: " << output.type()->str() << std::endl;
        }

        // Now try with OpenCV image processing
        std::cout << "\nProcessing OpenCV image..." << std::endl;
        
        try {
            // Create a simple test image
            cv::Mat test_image = cv::Mat::zeros(224, 224, CV_8UC3);
            cv::rectangle(test_image, cv::Point(50, 50), cv::Point(174, 174), cv::Scalar(255, 255, 255), -1);
            
            // Convert OpenCV Mat to torch tensor
            cv::Mat float_image;
            test_image.convertTo(float_image, CV_32F, 1.0/255.0);
            
            // Create tensor from OpenCV data
            auto image_tensor = torch::from_blob(float_image.data, {224, 224, 3}, torch::kFloat32);
            image_tensor = image_tensor.permute({2, 0, 1}).unsqueeze(0); // (1, 3, 224, 224)
            
            std::cout << "OpenCV image tensor shape: " << image_tensor.sizes() << std::endl;
            
            // Run inference on the image
            std::vector<torch::jit::IValue> image_inputs;
            image_inputs.push_back(image_tensor);
            
            auto image_output = module.forward(image_inputs);
            
            if (image_output.isTensor()) {
                auto image_output_tensor = image_output.toTensor();
                std::cout << "OpenCV image output shape: " << image_output_tensor.sizes() << std::endl;
                std::cout << "OpenCV image output mean: " << image_output_tensor.mean().item<float>() << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cout << "OpenCV processing failed: " << e.what() << std::endl;
        }

        std::cout << "DINOv3 C++ inference completed successfully!" << std::endl;
        
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.msg() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return -1;
    }

    return 0;
}
