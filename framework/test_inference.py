# Import required libraries
import torch

# Import custom classes
from framework.processor import Processor
from framework.inference import Inference
from framework.color_map import color_map
from framework.model import DeepLabV3Plus

# Test Inference class
NUM_CLASSES = 13
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "/home/ubuntu/computer-vision/computer-vision/experiments/modelweights"
model = DeepLabV3Plus(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location = DEVICE))
model.to(DEVICE)
 
inference = Inference(model, color_map, DEVICE)
image_path = "/home/ubuntu/computer-vision/computer-vision/testing/image_2/000160_10.png"
inference.predict_from_image_path(image_path)
 
test = inference.predict_from_image_path(image_path)
print(type(test))