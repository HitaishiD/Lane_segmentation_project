# Import required libraries
import torch
from PIL import Image

# Import custom classes
from framework.processor import Processor
from framework.inference import Inference
from framework.color_map import color_map
from framework.model import DeepLabV3Plus

# Test Inference class
NUM_CLASSES = 13
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "/home/hitaishi/lane-segmentation/Lane_segmentation_project/experiments/final_model.pth"
model = DeepLabV3Plus(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location = DEVICE))
model.to(DEVICE)
 
inference = Inference(model, color_map, DEVICE)
image_path = "/media/hitaishi/9961-4D68/data_semantics/training/image_2/000041_10.png"
 
output_mask = inference.predict_from_image_path(image_path)

original_image = Image.open(image_path)
original_size = original_image.size

output_mask_im = Image.fromarray(output_mask)
output_mask_im = output_mask_im.resize(original_size)
output_mask_im.save('output_mask.png')
