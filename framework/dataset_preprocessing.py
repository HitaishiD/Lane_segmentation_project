# Import custom classes
from framework.processor import Processor
from framework.color_map import color_map

# Define folders containing input masks and output masks
mask_folder = '/home/ubuntu/computer-vision/computer-vision/training/semantic_rgb'
output_mask_folder = '/home/ubuntu/computer-vision/computer-vision/preprocessed_masks'

# Preprocess input masks according to color map
preprocessor = Processor()
preprocessor.convert_rgb_to_mask(mask_folder, output_mask_folder, color_map)