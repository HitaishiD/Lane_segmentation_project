from inference import Inference
from evaluate import Evaluator
from color_map import color_map

from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import time

def predict_evaluate_one_image(image_path, true_mask_path, class_id, color_map):
    inference = Inference()
    pred_mask = inference.predict(image_path)
    
    true_mask = Image.open(true_mask_path).convert("RGB")
    true_mask_array = np.array(true_mask)

    evaluator = Evaluator()
    iou = evaluator.compute_iou(pred_mask, true_mask_array, class_id, color_map)
    
    return iou

def predict_evaluate_dataset(images_dir, true_masks_dir, class_id, color_map):
    images = [f for f in os.listdir(images_dir) if f.endswith(".png")]
    ious = []
    times = []

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


    mean_iou = sum(ious)/len(ious)
    mean_time = sum(times)/len(times)
    return mean_iou, mean_time

if __name__ == "__main__":
    images_dir = r"E:\data_semantics\training\image_2"
    true_masks_dir = r"E:\data_semantics\training\semantic_rgb"
    class_id = 1
    color_map = color_map
    mean_iou, mean_time = predict_evaluate_dataset(images_dir, true_masks_dir, class_id, color_map)
    print("Mean IoU: " + str(round(mean_iou,5)))
    print("Mean processing time: " + str(round(mean_time,5)) + " s")


