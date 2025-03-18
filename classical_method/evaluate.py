import numpy as np

class Evaluator:
    def compute_iou(self, pred_mask_array, true_mask_array, class_id, color_map):
        
        """
        Computes the Intersection over Union (IoU) for a specific class between 
        the predicted and true segmentation masks.

        Parameters:
            pred_mask_array (numpy.ndarray): The predicted mask 3D array (H, W, 3) 
            true_mask_array (numpy.ndarray): The ground truth mask image 3D array (H, W, 3) 
            class_id (int): The class ID for which to calculate the IoU.
            color_map (dict): The dictionary mapping RGB colors to Class ID.                        

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
    

    
