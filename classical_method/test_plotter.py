from plotter import Plotter

image_path = r"C:\Users\dhoow\Desktop\Computer Vision\lane_segmentation_project\modern\dummy_kitti_dataset\train_images\000007_10.png"


plotter = Plotter()

# ************ Method 1 ************
# plotter.plot_from_path(image_path)


# ************ Method 2 ************
from inference import Inference
inference = Inference()
prediction = inference.predict(image_path)
plotter.plot_array(prediction)