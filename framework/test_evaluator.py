# Import required libraries
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

# Import custom classes
from framework.evaluator import Evaluator
from framework.color_map import color_map
from framework.model import DeepLabV3Plus
from framework.dataset import ToTensorWithoutNormalization
from framework.dataset import KITTIdataset
from framework.processor import Processor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 13
BATCH_SIZE = 2

model = DeepLabV3Plus(NUM_CLASSES)
MODEL_PATH = "/home/ubuntu/computer-vision/computer-vision/experiments/modelweights"
model.load_state_dict(torch.load(MODEL_PATH, map_location = DEVICE))
model.to(DEVICE)

# Test Evaluator class
evaluator = Evaluator()
image_dir = '/home/ubuntu/computer-vision/computer-vision/training/image_2'
mask_dir = '/home/ubuntu/computer-vision/computer-vision/preprocessed_mask'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    ToTensorWithoutNormalization()
])

dataset = KITTIdataset(image_dir=image_dir, mask_dir=mask_dir,
                       transform=transform, mask_transform=mask_transform)

num_img = len(dataset)
train_size = int(0.6 * num_img)
val_size = int(0.2 * num_img)
test_size = num_img - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
mean_iou = evaluator.compute_mean_iou(model, test_loader,1,color_map, DEVICE)
print(mean_iou)