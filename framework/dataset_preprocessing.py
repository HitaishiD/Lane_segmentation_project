# Import custom classes
from framework.processor import Processor
from framework.color_map import color_map

# Define folders containing input masks and output masks
mask_folder = '/root/Lane_segmentation_project/dataset/training/semantic_rgb'
output_mask_folder = '/root/Lane_segmentation_project/dataset/training/preprocessed_masks'

# Preprocess input masks according to color map
preprocessor = Processor()
preprocessor.convert_rgb_to_mask(mask_folder, output_mask_folder, color_map)
