# Import required libraries
import sys
import os

# Import custom classes
from framework.processor import Processor
from framework.color_map import color_map

# Test Processor class
mask_folder = '/home/ubuntu/computer-vision/computer-vision/training/semantic_rgb'
output_mask_folder = '/home/ubuntu/computer-vision/computer-vision/testing_preprocessed_masks'

preprocessor = Processor()
preprocessor.convert_rgb_to_mask(mask_folder, output_mask_folder, color_map)