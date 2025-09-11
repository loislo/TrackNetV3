from transformers import AutoImageProcessor, AutoModel
import torch
import sys

model_version = "vith16plus"
# Create a wrapper class to handle the dictionary output
class DINOv3Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # Get the model output
        output = self.model(x)
        # Return the pooled output (main feature vector)
        return output.pooler_output

# Check for MPS availability
print("Checking device availability...")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"MPS (Metal Performance Shaders) is available! Using device: {device}")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available! Using device: {device}")
else:
    device = torch.device("cpu")
    print(f"Using CPU device: {device}")

# 1. Load a Pretrained DINOv3 Model
# DINOv3 models are available through torch.hub from the official repository.
# Here we load the smallest variant, ViT-Small with patch size 16.
# For a full list of available models, see the DINOv3 GitHub repository.
try:
    model_name = f"facebook/dinov3-{model_version}-pretrain-lvd1689m"
    dinov3 = AutoModel.from_pretrained(model_name)
    print(f"Successfully loaded DINOv3 {model_name} model.")
except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr)
    sys.exit(1)

# 2. Set the Model to Evaluation Mode and move to device
# This is a crucial step that disables layers like Dropout, which behave
# differently during training and inference.
dinov3.eval()
dinov3 = dinov3.to(device)
print(f"Model set to evaluation mode and moved to device: {device}")

# 3. Wrap the model to handle dictionary output
wrapped_model = DINOv3Wrapper(dinov3)
wrapped_model.eval()
wrapped_model = wrapped_model.to(device)
print("Model wrapped to handle dictionary output.")

# 4. Create a Dummy Input Tensor on the correct device
# The tensor's properties (shape, dtype, device) must match what the C++
# application will provide. These properties will be baked into the traced graph.
# Shape: (batch_size, num_channels, height, width)
# Dtype: torch.float32
# Device: Same as model
dummy_input = torch.randn(1, 3, 224, 224, device=device)
print(f"Created a dummy input tensor of shape: {dummy_input.shape} on device: {dummy_input.device}")

# 5. Test the wrapped model first
print("Testing wrapped model...")
with torch.no_grad():
    test_output = wrapped_model(dummy_input)
    print(f"Test output shape: {test_output.shape}")
    print(f"Test output device: {test_output.device}")

# 6. Trace the Model
# Use torch.jit.trace to execute the model with the dummy input and
# record the computational graph.
print("Tracing the model... This may take a moment.")
try:
    traced_model = torch.jit.trace(wrapped_model, dummy_input, strict=False)
    print("Model successfully traced.")
except Exception as e:
    print(f"Error during tracing: {e}", file=sys.stderr)
    sys.exit(1)

# 7. Save the Traced Model to a File
# The resulting ScriptModule is saved to a '.pt' file, which can be
# loaded directly by LibTorch in C++.
output_path = "dinov3.pt"
traced_model.save(output_path)
print(f"Traced model saved to: {output_path}")

# 8. Test the traced model
print("Testing traced model...")
with torch.no_grad():
    traced_output = traced_model(dummy_input)
    print(f"Traced model output shape: {traced_output.shape}")
    print(f"Traced model output device: {traced_output.device}")
    print("Tracing successful!")

# 9. Performance comparison
print("\nPerformance comparison:")
print("Testing inference speed...")

# Warm up
for _ in range(3):
    with torch.no_grad():
        _ = wrapped_model(dummy_input)

# Benchmark
import time
num_runs = 10

# Original model
start_time = time.time()
for _ in range(num_runs):
    with torch.no_grad():
        _ = wrapped_model(dummy_input)
original_time = (time.time() - start_time) / num_runs

# Traced model
start_time = time.time()
for _ in range(num_runs):
    with torch.no_grad():
        _ = traced_model(dummy_input)
traced_time = (time.time() - start_time) / num_runs

print(f"Original model average inference time: {original_time:.4f} seconds")
print(f"Traced model average inference time: {traced_time:.4f} seconds")
print(f"Speedup: {original_time/traced_time:.2f}x")