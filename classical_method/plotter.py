from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

class Plotter:
    def plot_from_path(self, image_path):
        """
        Plots an image from its file path
        
        Parameters:
        image_path (str): The file path of the image which is plotted

        Returns:
        None

        """
        # Open an image using the PIL library and convert it to RGB
        image = Image.open(image_path).convert("RGB")

        # Convert PIL image into numpy.ndarray
        image_array = np.array(image)

        # Plot and show image
        plt.imshow(image_array)
        plt.axis('off')
        plt.show()

    def plot_array(self, array):
        """
        Plots an image from its numpy.ndarray
        
        Parameters:
        numpy.ndarray: The numpy.ndarray of the image which is plotted

        Returns:
        None
        """
        # Plot and show image
        plt.imshow(array)
        plt.axis('off')
        plt.show()
