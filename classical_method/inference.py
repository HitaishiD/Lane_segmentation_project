from classical_method import ClassicalMethod

class Inference:
    def predict(self, image_path):
        """
        Predicts the lane segment in a given image

        Parameters:
        image_path (str): The file path of the image for which lane detection is performed

        Returns:
        numpy.ndarray: The processed image with the detected lane segment

         """
        
        # Create an instance of the ClassicalMethod class
        classical_method = ClassicalMethod(image_path)

        # Call the process method to perform lane detection and get the output
        output = classical_method.process()
        
        return output

