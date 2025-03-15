from dataset import ToTensorWithoutNormalization
from dataset import KITTIdataset
from torchvision import transforms



# Test Dataset class and Transforms class

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
