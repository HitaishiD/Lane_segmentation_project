import numpy as np
from PIL import Image
from color_map import color_map

# class_rgb = np.array([128, 64, 128])

def compute_iou(pred_mask_array , true_mask_array, class_id = 1):

    """
    Computes the Intersection over Union (IoU) for a specific class between 
    the predicted and true segmentation masks.

    Parameters:
        pred_mask_array (numpy.ndarray): The predicted mask 3D array (H, W, 3) 
        true_mask_array (numpy.ndarray): The ground truth mask image 3D array (H, W, 3) 
        class_id (int, optional): The class ID for which to calculate the IoU.
                                  Default is 1 (corresponding to lane)

    Returns:
        float: The Intersection over Union (IoU) score for the specified class.
               If there is no overlap (union is zero), returns 0."""

    assert pred_mask_array.shape == true_mask_array.shape, "Mask shapes should be the same"
    assert class_id in color_map.values(), "class_ID should be defined in color to class mapping"

    class_rgb = np.array(list(key for key, value in color_map.items() if value == class_id)[0])

    # Reshape the arrays into 2D arrays with 3 columns (RGB)
    reshaped_pred_mask_array = pred_mask_array.reshape(-1, 3)
    reshaped_true_mask_array = true_mask_array.reshape(-1, 3)

    # Create binary masks (1 for the target class, 0 for others)
    pred_lane = np.array([1 if np.all(pix == class_rgb) else 0 for pix in reshaped_pred_mask_array])
    true_lane = np.array([1 if np.all(pix == class_rgb) else 0 for pix in reshaped_true_mask_array])

    # Compute intersection and union
    intersection = np.logical_and(pred_lane, true_lane).sum()
    union = np.logical_or(pred_lane, true_lane).sum()

    # Compute IoU 
    iou = intersection / union if union > 0 else 0
    return iou

# Example 
# pred_mask = Image.open("pred_000037_10.png").convert("RGB")
# true_mask = Image.open("true_000037_10.png").convert("RGB") 

# pred_mask_array = np.array(pred_mask)
# true_mask_array = np.array(true_mask)
# # print(compute_iou(true_mask_array , true_mask_array))

# print(pred_mask_array.shape)

# black_rgb_array = np.zeros((375, 1242, 3), dtype=np.uint8)
# print(compute_iou(black_rgb_array , true_mask_array))