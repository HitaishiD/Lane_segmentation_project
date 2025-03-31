# Import required libraries
from torchvision.models import segmentation
import torch.nn as nn

# Define model class
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        self.model = segmentation.deeplabv3_resnet101(weights='DEFAULT')
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)['out']