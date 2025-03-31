# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import segmentation
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

# Define processor class for masks
class Processor:
    def convert_rgb_to_mask(self, mask_folder, output_mask_folder, color_map):
        ''' convert a folder of rgb masks to single channel mask with class indices'''
        os.makedirs(output_mask_folder, exist_ok=True)

        lookup_table = np.zeros((256, 256, 256), dtype=np.uint8)
        for (r, g, b), label in color_map.items():
            lookup_table[r, g, b] = label

        mask_files = [f for f in os.listdir(mask_folder) if f.endswith("png")]

        for mask_filename in tqdm(mask_files, desc="Processing Masks", unit="img"):

            mask_path = os.path.join(mask_folder, mask_filename)
            mask_image = cv2.imread(mask_path)
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)

            integer_mask = lookup_table[mask_image[...,0], mask_image[...,1], mask_image[...,2]]
    
            output_mask_path = os.path.join(output_mask_folder, mask_filename)
            cv2.imwrite(output_mask_path, integer_mask)     


    def convert_mask_to_rgb(mask, color_map):
        ''' convert an 2D mask with class indices to an rgb mask '''
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()  

        height, width = mask.shape
        rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

        for class_rgb, class_idx in color_map.items():
            rgb_mask[mask == class_idx] = class_rgb  

        return rgb_mask           