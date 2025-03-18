from inference import Inference
from evaluate import Evaluator
from color_map import color_map

from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import time

def predict_evaluate_one_image(image_path, true_mask_path, class_id, color_map):
    """
    Predicts lane markings for a single image and evaluates the prediction using IoU.

    Parameters:
    image_path (str): Path to the input image.
    true_mask_path (str): Path to the ground truth mask.
    class_id (int): Class ID for lane markings in the mask.
    color_map (dict): Mapping of class IDs to colors in the mask.

    Returns:
    float: Intersection over Union (IoU) score for the prediction.
    """
    inference = Inference()
    pred_mask = inference.predict(image_path)
    
    # Load the ground truth mask and convert it to a NumPy array
    true_mask = Image.open(true_mask_path).convert("RGB")
    true_mask_array = np.array(true_mask)

    # Compute IoU using the evaluator
    evaluator = Evaluator()
    iou = evaluator.compute_iou(pred_mask, true_mask_array, class_id, color_map)
    
    return iou

def predict_evaluate_dataset(images_dir, true_masks_dir, class_id, color_map):
    """
    Predicts lane markings and evaluates IoU for an entire dataset of images.

    Parameters:
    images_dir (str): Directory containing input images.
    true_masks_dir (str): Directory containing ground truth masks.
    class_id (int): Class ID for lane markings in the mask.
    color_map (dict): Mapping of class IDs to colors in the mask.

    Returns:
    tuple: (mean IoU, mean processing time per image)
    """
    images = [f for f in os.listdir(images_dir) if f.endswith(".png")]
    
    # Initialize lists to store IoU and processing time for each image 
    ious = []
    times = []

    # Iterate through images and compute IoU
    for image in tqdm(images, desc= "Processing Images", unit = "image"):
        image_path = os.path.join(images_dir, image)
        # Assuming masks have the same name as the image:
        true_mask_path = os.path.join(true_masks_dir, image) 

        start = time.time()
        iou = predict_evaluate_one_image(image_path, true_mask_path, class_id, color_map)
        end = time.time()
        processing_time = end - start

        times.append(processing_time)
        ious.append(iou)

    # Compute mean IoU and mean processing time
    mean_iou = sum(ious)/len(ious)
    mean_time = sum(times)/len(times)
    return mean_iou, mean_time

if __name__ == "__main__":
    """
    Main execution block to evaluate lane detection performance on an entire dataset.
    
    """
    images_dir = r"E:\data_semantics\training\image_2"
    true_masks_dir = r"E:\data_semantics\training\semantic_rgb"
    class_id = 1
    color_map = color_map
    mean_iou, mean_time = predict_evaluate_dataset(images_dir, true_masks_dir, class_id, color_map)
    print("Mean IoU: " + str(round(mean_iou,5)))
    print("Mean processing time: " + str(round(mean_time,5)) + " s")


