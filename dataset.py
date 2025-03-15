from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import torch


class KITTIdataset(Dataset):
    def __init__(self,image_dir, mask_dir, transform=None, mask_transform=None, image_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_size = image_size

        self.image_names = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.mask_names = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        assert len(self.image_names) == len(self.mask_names), "Mismatch between image and mask count!"

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])

        image = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.image_size:
            image = image.resize(self.image_size)
            mask = mask.resize(self.image_size)

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


class ToTensorWithoutNormalization:
    def __call__(self, sample):
        mask = np.array(sample)
        mask_tensor = torch.tensor(mask, dtype=torch.long)
        return mask_tensor