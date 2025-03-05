import numpy as np
from color_map import color_map

# def compute_iou(pred_mask, true_mask, class_id = 1):
#     """
#     Computes the Intersection over Union (IoU) for a specific class in a single image.

#     Args:
#     - pred_mask (numpy array): Predicted segmentation mask
#     - true_mask (numpy array): Ground truth segmentation mask
#     - class_id (int): The class ID to compute IoU for. The default value is 1 (road)

#     Returns:
#     - IoU score (float)
#     """

#     assert pred_mask.shape == true_mask.shape, "Mask shapes should be the same"

#     assert class_id in color_map.values(), "class_ID should be defined in color to class mapping"

#     # Create binary masks for the given class - Extract the class required
#     pred_class = (pred_mask == class_id)
#     true_class = (true_mask == class_id)

#     # Compute intersection and union
#     intersection = np.logical_and(pred_class, true_class).sum()
#     union = np.logical_or(pred_class, true_class).sum()

#     if union == 0: #If the required class does not exist in either of the prediction or true mask
#         return 0.0

#     return intersection / union



# class_rgb = np.array([128, 64, 128], dtype=np.uint8)
# print(class_rgb)
# # Function to compute IoU for a specific RGB class
# def compute_iou_for_rgb_class(predicted_rgb_mask, true_rgb_mask, class_rgb):

#     class_rgb = np.asarray(class_rgb, dtype=np.uint8).reshape(1, 1, 3)
#     print(class_rgb)

#     # Convert predicted and true RGB masks to boolean arrays where RGB matches the lane class
#     pred_lane = np.all(predicted_rgb_mask == class_rgb, axis=-1)
#     true_lane = np.all(true_rgb_mask == class_rgb, axis=-1)
   
    
#     # Calculate intersection and union for the lane class
#     intersection = np.sum(pred_lane & true_lane)  # True Positives
#     union = np.sum(pred_lane | true_lane)        # Union of predicted and true lane
    
#     # Compute IoU (avoid division by zero)
#     if union == 0:
#         return 0  # If no lane exists in the union, return NaN (to handle edge cases)
#     else:
#         return intersection / union

# Example usage:
# predicted_rgb_mask = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)  # Replace with actual predicted RGB mask
# true_rgb_mask = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)      # Replace with actual true RGB mask

import numpy as np

# The RGB value of the class you want to compute IoU for (e.g., Lane)
class_rgb = np.array([128, 64, 128])  # Store as a tuple


from PIL import Image
# Compute IoU for the lane class
pred_mask = Image.open("pred_000037_10.png").convert("RGB")
true_mask = Image.open("true_000037_10.png").convert("RGB") 

pred_mask_array = np.array(pred_mask)
reshaped_pred_mask_array = pred_mask_array.reshape(-1,3)

pred_lane = np.array([1 if np.all(pix == class_rgb) else 0 for pix in reshaped_pred_mask_array])


# pred_lane = np.all(pred_mask == class_rgb, axis=-1).astype(np.uint8)
# print(pred_lane)
# pred_pix = list(pred_mask.getdata())
# pred_lane = []
# for item in pred_pix:
#     if item == class_rgb:
#         pred_lane.append(1)
#     else:
#         pred_lane.append(0)

# true_pix = list(true_mask.getdata())
# true_lane = []
# for item in true_pix:
#     if item == class_rgb:
#         true_lane.append(1)
#     else:
#         true_lane.append(0)

# intersection = np.logical_and(pred_lane, true_lane).sum()
# union = np.logical_or(pred_lane, true_lane).sum()
# print(intersection/union)

# print(pred_pix)
# iou_lane = compute_iou_for_rgb_class(pred_mask, true_mask, class_rgb)

# print("IoU for the lane class:", iou_lane)
