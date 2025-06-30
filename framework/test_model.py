# Import required libraries
import sys
import os

# Import custom class
from framework.model import DeepLabV3Plus

# Test Model class
NUM_CLASSES = 13

model = DeepLabV3Plus(NUM_CLASSES)

print(model)