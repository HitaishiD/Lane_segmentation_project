from classical_method import ClassicalMethod
import matplotlib.pyplot as plt

image_path = r"C:\Users\dhoow\Desktop\Computer Vision\lane_segmentation_project\train_000037_10.png"
classical_method = ClassicalMethod(image_path)
output = classical_method.process()

plt.imshow(output)
plt.show()