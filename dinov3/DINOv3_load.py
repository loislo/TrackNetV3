# Load model directly
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vit7b16-pretrain-lvd1689m")
model = AutoModel.from_pretrained("facebook/dinov3-vit7b16-pretrain-lvd1689m")
 