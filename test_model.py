import sys
import os

# Add the directory containing model.py to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import DeepLabV3Plus

# Test Model class
NUM_CLASSES = 13

model = DeepLabV3Plus(NUM_CLASSES)

print(model)