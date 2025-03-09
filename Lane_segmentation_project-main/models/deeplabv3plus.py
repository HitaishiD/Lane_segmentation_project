import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes = 13):
        super(DeepLabV3Plus, self).__init__()
        self.model = segmentation.deeplabv3_resnet101(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)['out']