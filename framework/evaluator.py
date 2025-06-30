# Import required libraries
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

# Import custom classes
from framework.processor import Processor

# Define the class which performs evaluation of the model by computing IoU
class Evaluator:
    def compute_iou(self, pred_mask_array, true_mask_array, class_id, color_map):
        
        """
        Computes the Intersection over Union (IoU) for a specific class between 
        the predicted and true segmentation masks.

        Parameters:
            pred_mask_array (numpy.ndarray): The predicted mask 3D array (H, W, 3) 
            true_mask_array (numpy.ndarray): The ground truth mask image 3D array (H, W, 3) 
            class_id (int): The class ID for which to calculate the IoU.
            color_map (dict): The dictionary mapping RGB colors to Class ID.                        

        Returns:
            float: The Intersection over Union (IoU) score for the specified class.
                If there is no overlap (union is zero), returns 0."""
        
        assert pred_mask_array.shape == true_mask_array.shape, "Mask shapes should be the same"
        assert class_id in color_map.values(), "class_ID should be defined in color to class mapping"

        class_rgb = np.array(list(key for key, value in color_map.items() if value == class_id)[0])

        # Reshape the arrays into 2D arrays with 3 columns (RGB)
        reshaped_pred_mask_array = pred_mask_array.reshape(-1, 3)
        reshaped_true_mask_array = true_mask_array.reshape(-1, 3)

        # Create binary masks (1 for the target class, 0 for others)
        pred_lane = np.array([1 if np.all(pix == class_rgb) else 0 for pix in reshaped_pred_mask_array])
        true_lane = np.array([1 if np.all(pix == class_rgb) else 0 for pix in reshaped_true_mask_array])

        # Compute intersection and union
        intersection = np.logical_and(pred_lane, true_lane).sum()
        union = np.logical_or(pred_lane, true_lane).sum()

        # Compute IoU 
        iou = intersection / union if union > 0 else 0
        return iou
    
    def compute_mean_iou(self,model, test_loader, class_id, color_map, device):
        """
        Computes the mean IoU across all images in the test dataset for one class.

        Args:
            model (torch.nn.Module): Trained segmentation model.
            test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
            class_id (int): Class index for which the IoU needs to be computed
            color_map (dict): Mapping from class indices to RGB values.
            device (torch.device): Device (CPU/GPU).

        Returns:
            float: mean iou score."""
        
        model.eval()  
        all_iou = [] 

        with torch.no_grad():  
            for images, true_masks in test_loader:
                images = images.to(device)  
                true_masks = true_masks.to(device)  

                # Get predictions from model
                outputs = model(images)  # (B, num_classes, H, W)
                predicted_masks = torch.argmax(outputs, dim=1)  #  (B, H, W)

                for i in range(images.shape[0]):  
                    # Convert predicted & true masks to RGB
                    true_mask_rgb = Processor.convert_mask_to_rgb(true_masks[i].squeeze(0), color_map)
                    pred_mask_rgb = Processor.convert_mask_to_rgb(predicted_masks[i], color_map)
        
                    iou = self.compute_iou(pred_mask_rgb, true_mask_rgb, class_id, color_map)
                    all_iou.append(iou)

        # Compute final mean IoU per class
        mean_iou_score = sum(all_iou)/len(all_iou)

        return mean_iou_score
        
