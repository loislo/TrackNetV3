import torch
import sys

# 1. Load a Pretrained DINOv3 Model
# DINOv3 models are available through torch.hub from the official repository.
# Here we load the smallest variant, ViT-Small with patch size 16.
# For a full list of available models, see the DINOv3 GitHub repository.
try:
    dinov3_vits16 = torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16')
    print("Successfully loaded DINOv3 ViT-S/16 model.")
except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr)
    sys.exit(1)

# 2. Set the Model to Evaluation Mode
# This is a crucial step that disables layers like Dropout, which behave
# differently during training and inference.
dinov3_vits16.eval()
print("Model set to evaluation mode.")

# 3. Create a Dummy Input Tensor
# The tensor's properties (shape, dtype, device) must match what the C++
# application will provide. These properties will be baked into the traced graph.
# Shape: (batch_size, num_channels, height, width)
# Dtype: torch.float32
# Device: CPU for this example
dummy_input = torch.randn(1, 3, 224, 224)
print(f"Created a dummy input tensor of shape: {dummy_input.shape}")

# 4. Trace the Model
# Use torch.jit.trace to execute the model with the dummy input and
# record the computational graph.
print("Tracing the model... This may take a moment.")
try:
    traced_model = torch.jit.trace(dinov3_vits16, dummy_input)
    print("Model successfully traced.")
except Exception as e:
    print(f"Error during tracing: {e}", file=sys.stderr)
    sys.exit(1)

# 5. Save the Traced Model to a File
# The resulting ScriptModule is saved to a '.pt' file, which can be
# loaded directly by LibTorch in C++.
output_path = "dinov3_vits16_traced.pt"
traced_model.save(output_path)
print(f"Traced model saved to: {output_path}")
