import torch
from tqdm import tqdm
import os
from PIL import Image
import numpy as np

from inference import Inference
from color_map import color_map
from model import DeepLabV3Plus

# Inputs
MODEL_PATH = r"C:\Users\dhoow\Desktop\computer-vision-march\final_model.pth"
image_dir = r"C:\Users\dhoow\Desktop\Computer Vision\lane_segmentation_project\modern\dummy_kitti_dataset\train_images"
output_dir = r"C:\Users\dhoow\Desktop\Computer Vision\lane_segmentation_project\modern\dummy_kitti_dataset\modern_prediction"

# Inputs that do not change
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 13

# Load the model
model = DeepLabV3Plus(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location = DEVICE))
model.to(DEVICE)

# Create an instance of the inference class
inference = Inference(model, color_map, DEVICE)

# Make an inference
images = os.listdir(image_dir)

for image in tqdm(images, desc="Processing Images", unit="image"):
    image_path = os.path.join(image_dir, image)
    output_array = inference.predict_from_image_path(image_path)
    output = Image.fromarray(output_array.astype(np.uint8))
    save_path = os.path.join(output_dir,image)
    output.save(save_path)