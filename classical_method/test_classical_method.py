from classical_method import ClassicalMethod
import matplotlib.pyplot as plt

image_path = r"E:\data_semantics\training\image_2\000105_10.png"

classical_method = ClassicalMethod(image_path)
output = classical_method.process()

print(classical_method.lines)
plt.imshow(output)
plt.axis('off')
plt.show()