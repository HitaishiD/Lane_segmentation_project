import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import logging
import time
import pandas as pd

# Import your custom classes and models
from dataset import ToTensorWithoutNormalization
from dataset import KITTIdataset
from model import DeepLabV3Plus
from processor import Processor




class Inference:
    def __init__(self, model, color_map, device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.color_map = color_map
        
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_from_image_path(self, image_path):
        ''' Make prediction from image file'''

        image = Image.open(image_path).convert('RGB')
        input_tensor = self.image_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        predicted_rgb_mask = Processor.convert_mask_to_rgb(predicted_mask, self.color_map)

        return predicted_rgb_mask
    
    def predict_from_test_loader(self, test_loader):
        ''' Make prediction for an image from test_loader'''

        data_iter = iter(test_loader)
        image, _ = next(data_iter)
        image = image[0].unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
            predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        predicted_rgb_mask = Processor.convert_mask_to_rgb(predicted_mask, self.color_map)

        return predicted_rgb_mask