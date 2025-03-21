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


# Import your custom classes and models
from dataset import ToTensorWithoutNormalization
from dataset import KITTIdataset
from model import DeepLabV3Plus



def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, NUM_CLASSES, experiment_folder, EXPERIMENT_NAME):
    train_losses = []
    val_losses = []

    plt.ion()
    fig, ax = plt.subplots()

    logging.info("Starting training...")

    epoch_progress = tqdm(range(epochs), desc="Training Progress")  

    for epoch in epoch_progress:
        print(f"\nEpoch {epoch + 1}/{epochs}")
        logging.info(f"Epoch {epoch + 1}/{epochs}")

        # Train Phase
        model.train()
        running_train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.squeeze(1).long().to(device)

            masks = torch.clamp(masks, 0, NUM_CLASSES - 1)

            if (masks < 0).any() or (masks >= NUM_CLASSES).any():
                print(f"Invalid mask values found: {masks}")

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Train Loss: {avg_train_loss:.4f}")
        logging.info(f"Train Loss: {avg_train_loss:.4f}")

        # Validation Phase
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.squeeze(1).long().to(device)

                masks = torch.clamp(masks, 0, NUM_CLASSES - 1)

                preds = model(images)
                loss = criterion(preds, masks)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Val Loss: {avg_val_loss:.4f}")
        logging.info(f"Val Loss: {avg_val_loss:.4f}")

        epoch_progress.set_postfix({"Train Loss": f"{avg_train_loss:.4f}", "Val Loss": f"{avg_val_loss:.4f}"})

        # Update plot
        ax.clear()
        ax.plot(train_losses, label="Train Loss")
        ax.plot(val_losses, label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.draw()
        plt.pause(0.1)

        # Learning rate scheduling
        scheduler.step()

        # Save model checkpoint after each epoch
        checkpoint_path = os.path.join(experiment_folder, f"epoch_{epoch+1}.pth")  # Fixed path here
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved for epoch {epoch+1}!")

    # Final model checkpoint
    final_checkpoint_path = os.path.join(experiment_folder, "final_model.pth")  # Save the final model
    torch.save(model.state_dict(), final_checkpoint_path)
    print("Final model checkpoint saved!")
    logging.info(f"Final model checkpoint saved at {final_checkpoint_path}")

    # Save the losses to a CSV file
    loss_df = pd.DataFrame({
        'Epoch': range(1, epochs + 1),
        'Training Loss': train_losses,
        'Validation Loss': val_losses
    })

    loss_df.to_csv(os.path.join(experiment_folder, "training_losses.csv"), index=False)

    print("Training Completed!")
    logging.info("Training Completed!")
    plt.ioff()
    plt.show()

    return val_losses[-1]



def parse_args():
    parser = argparse.ArgumentParser(description="Training parameters for model")

    # Adding the arguments
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs to train the model')
    # Parse arguments
    args = parser.parse_args()

    return args




def main():
    '''
    1. Dataset including transforms
    2. Model 
    3. Criterion
    4. Optimizer
    5. Scheduler
    6. Device
    7. Epochs 
    '''

    ##########################################
    ###       PARAMETERS 
    ##########################################
    # Get parsed arguments
    args = parse_args()

    # Now you can use the parsed arguments in your code
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    epochs = args.epochs
    CHECKPOINT_DIR = "experiments"
    
    # Create experiment name based on hyperparameters
    EXPERIMENT_NAME = f"bs{BATCH_SIZE}_lr{LEARNING_RATE}_epochs{epochs}"

    ### Params that won't change
    NUM_CLASSES = 13
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


    # Dataset and Transforms setup
    image_dir = '/home/ubuntu/computer-vision/computer-vision/training/image_2'
    mask_dir = '/home/ubuntu/computer-vision/computer-vision/preprocessed_mask'

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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model setup
    model = DeepLabV3Plus(NUM_CLASSES)
    model = model.to(DEVICE)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=10, 
                                                gamma=0.1)

    # Create the experiment folder if it doesn't exist
    experiment_folder = os.path.join(CHECKPOINT_DIR, EXPERIMENT_NAME)
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)

    # Train the model
    train(model, 
          train_loader, 
          val_loader,
          criterion,
          optimizer,
          scheduler,
          DEVICE,
          epochs,
          NUM_CLASSES,
          experiment_folder,   # Pass the experiment folder here
          EXPERIMENT_NAME)



if __name__ == "__main__":
    main()
