from processor import Processor
from color_map import color_map

# Test Preprocessor class

mask_folder = '/home/ubuntu/computer-vision/computer-vision/training/semantic_rgb'
output_mask_folder = '/home/ubuntu/computer-vision/computer-vision/testing_preprocessed_masks'

preprocessor = Processor()
preprocessor.convert_rgb_to_mask(mask_folder, output_mask_folder, color_map)