from torchvision import transforms

EPOCHS = 50
BATCH_SIZE = 8
NUM_CLASSES = 13
LEARNING_RATE = 1e-3
IMAGE_SIZE = (256, 256)
MODEL_NAME = "deeplabv3"
CHECKPOINT_DIR = "./experiments/"

transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

mask_transform = transforms.Compose([
    transforms.ToTensor()
])