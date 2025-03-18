from inference import Inference

image_path = r"C:\Users\dhoow\Desktop\Computer Vision\lane_segmentation_project\modern\dummy_kitti_dataset\train_images\000005_10.png"

inference = Inference()
prediction = inference.predict(image_path)
print(prediction)