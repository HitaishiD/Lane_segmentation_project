import torch
from torchvision import transforms
import torchvision.models.segmentation as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from color_map import color_map

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.deeplabv3_resnet101(weights=None, aux_loss=True)
model.classifier[4] = torch.nn.Conv2d(256, 13, kernel_size=(1, 1), stride=(1, 1))
model.load_state_dict(torch.load("C:\\Users\\dhoow\\Downloads\\deeplabv3_epoch10.pth", map_location = device))
model.eval()

image_path = "train_000037_10.png"
mask_path = "true_000037_10.png"

image = Image.open(image_path).convert("RGB")  
mask = Image.open(mask_path).convert("RGB") 

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor (scales pixel values to [0,1])
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet values
])

image_tensor = transform(image)

with torch.no_grad():
    output = model(image_tensor.unsqueeze(0).to(device))['out'][0]
    predicted_mask = torch.argmax(output, dim=0).cpu().numpy()


def class_to_rgb(mask, colormap):
    mask = np.asarray(mask, dtype=np.uint8)
    
    # Create an empty RGB image
    rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for color, class_id in colormap.items():
        rgb_image[mask == class_id] = color
    
    return rgb_image


plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
plt.imshow(image)

plt.subplot(1,3,2)
plt.imshow(mask)

plt.subplot(1,3,3)
rgb_mask = class_to_rgb(predicted_mask, color_map)
plt.imshow(rgb_mask)
plt.show()


predicted_mask_img = Image.fromarray(rgb_mask)
predicted_mask_img.save("pred_000037_10.png")

# from iou import compute_iou_for_rgb_class
# print(compute_iou_for_rgb_class(rgb_mask, mask,(128, 64, 128) ))