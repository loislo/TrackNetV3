# Load model directly
from transformers import AutoModel
import argparse

# Define available model versions
AVAILABLE_MODELS = {
    'vitl16': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
    'vits16': 'facebook/dinov3-vits16-pretrain-lvd1689m', 
    'vits16plus': 'facebook/dinov3-vits16plus-pretrain-lvd1689m',
    'vitb16': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
    'vith16plus': 'facebook/dinov3-vith16plus-pretrain-lvd1689m'
}

# Parse command line arguments
parser = argparse.ArgumentParser(description='Load DINOv3 model')
parser.add_argument('--model_version', 
                    choices=list(AVAILABLE_MODELS.keys()),
                    default='vits16',
                    help='DINOv3 model version to use (default: vits16)')
args = parser.parse_args()

model_version = args.model_version
model_name = AVAILABLE_MODELS[model_version]

print(f"Loading DINOv3 model: {model_version} ({model_name})")

model = AutoModel.from_pretrained(model_name)

print(f"Successfully loaded {model_version} model.")
 