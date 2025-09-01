#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>

int main() {
    try {
        std::cout << "Testing LibTorch installation..." << std::endl;
        
        // Test basic tensor operations
        auto tensor = torch::randn({2, 3});
        std::cout << "Created tensor with shape: " << tensor.sizes() << std::endl;
        std::cout << "Tensor sum: " << tensor.sum().item<float>() << std::endl;
        
        // Test loading the model file
        std::cout << "Attempting to load traced model..." << std::endl;
        torch::jit::script::Module module;
        module = torch::jit::load("dinov3_vits16_traced.pt");
        std::cout << "Model loaded successfully!" << std::endl;
        
        // Test model properties
        std::cout << "Model has " << module.parameters().size() << " parameters" << std::endl;
        
        // Test with a very simple input
        auto simple_input = torch::ones({1, 3, 224, 224});
        std::cout << "Created simple input tensor" << std::endl;
        
        // Try to run inference
        std::cout << "Running inference..." << std::endl;
        torch::NoGradGuard no_grad;
        
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(simple_input);
        
        auto output = module.forward(inputs);
        std::cout << "Inference completed!" << std::endl;
        
        if (output.isTensor()) {
            auto output_tensor = output.toTensor();
            std::cout << "Output shape: " << output_tensor.sizes() << std::endl;
        }
        
        std::cout << "Test completed successfully!" << std::endl;
        
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
