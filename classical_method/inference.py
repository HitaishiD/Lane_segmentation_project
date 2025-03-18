from classical_method import ClassicalMethod

class Inference:
    def predict(self, image_path):
        classical_method = ClassicalMethod(image_path)
        output = classical_method.process()
        return output

