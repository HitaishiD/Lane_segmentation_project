from evaluate import Evaluator
from color_map import color_map

from PIL import Image
import numpy as np 

mask_path = r"C:\Users\dhoow\Desktop\Computer Vision\lane_segmentation_project\modern\dummy_kitti_dataset\train_masks\000009_10.png"
mask_array = np.array(Image.open(mask_path).convert("RGB"))

evaluate = Evaluator()
print(evaluate.compute_iou(mask_array, mask_array, 1, color_map)) 
