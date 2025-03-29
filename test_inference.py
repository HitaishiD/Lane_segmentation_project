import torch

from processor import Processor
from inference import Inference
from color_map import color_map
from model import DeepLabV3Plus

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Test Model class
NUM_CLASSES = 13


# Load the model
MODEL_PATH = "/home/ubuntu/computer-vision/computer-vision/experiments/modelweights"
model = DeepLabV3Plus(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location = DEVICE))
model.to(DEVICE)
 
inference = Inference(model, color_map, DEVICE)
image_path = "/home/ubuntu/computer-vision/computer-vision/testing/image_2/000160_10.png"
inference.predict_from_image_path(image_path)
 
test = inference.predict_from_image_path(image_path)
print(type(test))