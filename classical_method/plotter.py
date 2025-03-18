from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

class Plotter:
    def plot_from_path(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        plt.imshow(image_array)
        plt.axis('off')
        plt.show()

    def plot_array(self, array):
        plt.imshow(array)
        plt.axis('off')
        plt.show()
