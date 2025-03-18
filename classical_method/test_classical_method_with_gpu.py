from classical_method_with_gpu import ClassicalMethod
import matplotlib.pyplot as plt

image_path = r"/home/ubuntu/computer-vision/training/image_2/000002_10.png"
classical_method = ClassicalMethod(image_path)
output = classical_method.process()

plt.imshow(output)
plt.show()