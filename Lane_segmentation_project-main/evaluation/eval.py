import torch
from models import deeplabv3plus
from utils import config
from iou import compute_iou
from training.train import test_loader
import os
import numpy as np
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(config.CHECKPOINT_DIR, f"{config.MODEL_NAME}_best.pth")
NUM_CLASSES = config.NUM_CLASSES
BATCH_SIZE = config.BATCH_SIZE
IMAGE_SIZE = config.IMAGE_SIZE

model = deeplabv3plus(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def evaluate_model(model, test_loader, device):
    total_iou = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for image, mask in test_loader:
            image, mask = image.to(device), mask.to(device)
            pred = model(image)
            
            total_dice += compute_iou(pred , mask, class_id = 1)
            num_samples += 1

    mean_iou = total_iou / num_samples
    print(f" Mean IoU: {mean_iou:.4f}")

if __name__ == "__main__":
    evaluate_model(model, test_loader, DEVICE)