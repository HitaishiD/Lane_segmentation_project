import torch
import torch.nn as nn
import torch.optim as optim
import os

from data.data_loader import KITTIdataset
from torch.utils.data import DataLoader, random_split
from models import deeplabv3plus
import utils.config as config


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs):
    best_val_loss = float("inf")

    for epoch in range(epochs):
        print(f"\n Epoch {epoch + 1}/{epochs}")

        # Train Phase
        model.train()
        running_train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        print(f"Train Loss: {avg_train_loss:.4f}")

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        val_losses = []

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                preds = model(images)
                loss = criterion(preds, masks)
                running_val_loss += loss.item()

                val_losses.append(criterion(preds, masks))  

        avg_val_loss = running_val_loss / len(val_loader)
        avg_dice = sum(val_losses) / len(val_losses)
        print(f"Val Loss: {avg_val_loss:.4f} | Dice Score: {avg_dice:.4f}")

        # Learning rate scheduling
        scheduler.step()

        # Save best model checkpoint ---- 
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{config.MODEL_NAME}_best.pth"))
            print("Model checkpoint saved!")

    print("Training Completed!")


# -----------------------------------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = config.EPOCHS
BATCH_SIZE = config.BATCH_SIZE
NUM_CLASSES = config.NUM_CLASSES
LEARNING_RATE = config.LEARNING_RATE
IMAGE_SIZE = config.IMAGE_SIZE
CHECKPOINT_DIR = config.CHECKPOINT_DIR

dataset = KITTIdataset(image_dir='\\a', mask_dir='\\b',transform=config.transform, mask_transform=config.mask_transform)

num_img = len(dataset)
train_size = int(0.6*num_img)
val_size = int(0.2*num_img)
test_size = num_img - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = deeplabv3plus.DeepLabV3Plus(num_classes=13).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

train(model, train_loader, val_loader, criterion, optimizer, scheduler, DEVICE, EPOCHS)