import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
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
parser = argparse.ArgumentParser(description='Run DINOv3 inference on an image')
parser.add_argument('--model_version', 
                    choices=list(AVAILABLE_MODELS.keys()),
                    default='vits16',
                    help='DINOv3 model version to use (default: vits16)')
parser.add_argument('--image_url',
                    default="http://images.cocodataset.org/val2017/000000039769.jpg",
                    help='URL of image to process (default: COCO example image)')
args = parser.parse_args()

model_version = args.model_version
model_name = AVAILABLE_MODELS[model_version]
url = args.image_url

print(f"Loading DINOv3 model: {model_version} ({model_name})")
print(f"Processing image from: {url}")

image = load_image(url)

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(
    model_name, 
    device_map="auto", 
)

inputs = processor(images=image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs)

pooled_output = outputs.pooler_output
print("Pooled output shape:", pooled_output.shape)
