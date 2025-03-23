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
import argparse
import optuna


# Import your custom classes and models
from dataset import ToTensorWithoutNormalization
from dataset import KITTIdataset
from model import DeepLabV3Plus
from trainer import train
from color_map import color_map

def objective(trial):
    """
    Objective function for Optuna.
    """
    # Define hyperparameter search space
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    epochs = trial.suggest_int("epochs", 10, 50)
    
    CHECKPOINT_DIR = "experiments"
    EXPERIMENT_NAME = f"bs{batch_size}_lr{learning_rate}_epochs{epochs}"

    NUM_CLASSES = 13
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    image_dir = '/home/ubuntu/computer-vision/computer-vision/training/image_2'
    mask_dir = '/home/ubuntu/computer-vision/computer-vision/preprocessed_masks'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        ToTensorWithoutNormalization()
    ])

    # Create dataset and split into train, validation, and test sets
    dataset = KITTIdataset(image_dir=image_dir, mask_dir=mask_dir,
                           transform=transform, mask_transform=mask_transform)

    num_img = len(dataset)
   

    train_size = int(0.6 * num_img)
    val_size = int(0.2 * num_img)
    test_size = num_img - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Model setup
    model = DeepLabV3Plus(NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=10, 
                                                gamma=0.1)

    experiment_folder = os.path.join(CHECKPOINT_DIR, EXPERIMENT_NAME)
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)

    class_id = 1
    
    val_loss =  train(model, 
          train_loader, 
          val_loader,
          criterion,
          optimizer,
          scheduler,
          DEVICE,
          epochs,
          NUM_CLASSES,
          experiment_folder,  
          EXPERIMENT_NAME,
          class_id,
          color_map)

    return val_loss

# Run the optimization
study = optuna.create_study(direction="minimize", study_name="study2",
                            storage="sqlite:///optuna_study2.db")
study.optimize(objective, n_trials=20)  

# Print best parameters
print("Best parameters:", study.best_params)
    
