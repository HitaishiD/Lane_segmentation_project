from main import predict_evaluate_one_image
from color_map import color_map

image_path = r"C:\Users\dhoow\Desktop\Computer Vision\lane_segmentation_project\modern\dummy_kitti_dataset\train_images\000005_10.png"
true_mask_path = r"C:\Users\dhoow\Desktop\Computer Vision\lane_segmentation_project\modern\dummy_kitti_dataset\train_masks\000005_10.png"
class_id = 1
color_map = color_map

print(predict_evaluate_one_image(image_path, true_mask_path, class_id, color_map))
