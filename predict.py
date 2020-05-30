from glob import glob
import argparse
import io
import os

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from networks import MnistNetwork


class DigitPredictor:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(model_path)
        self.model = MnistNetwork().to(self.device)
        self.model.load_state_dict(checkpoint)

    def _create_image_object(self, image_bytes):
        """Create a PIL.Image object that contains the image bytes."""
        f = io.BytesIO()
        f.write(image_bytes)
        image = Image.open(f)
        return image

    def _to_jpeg(self, image_object, mode='RGB'):
        """Convert the given PIL.Image object's format into `JPEG`."""
        f = io.BytesIO()
        image_object.convert(mode).save(f, format='JPEG')
        image = Image.open(f)
        return image

    def image_bytes_to_tensor(self, image_bytes):
        """Preprocesses image bytes.

        Returns:
            - a torch.Tensor of which shape is [1, 1, 28, 28].
        """
        image = self._create_image_object(image_bytes)
        image = self._to_jpeg(image, mode='L')  # 'L' for grayscale
        image = image.resize(size=(28, 28))
        image = np.array(image, dtype=np.uint8)
        tensor = torchvision.transforms.ToTensor()(image)
        tensor = torchvision.transforms.Normalize(mean=(0.1307,), std=(0.3081,))(tensor)
        tensor = tensor.unsqueeze(0)
        return tensor

    def predict(self, image_bytes):
        """Predicts a digit the image represents."""
        tensor = self.image_bytes_to_tensor(image_bytes)
        output = self.model(tensor.to(self.device))
        prediction = output.argmax().item()
        return prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="path to MNIST model")
    parser.add_argument('--image', type=str, required=True, help="path to an image")
    args = parser.parse_args()

    predictor = DigitPredictor(args.model_path)
    prediction = predictor.predict(open(args.image, 'rb').read())
    print('Prediction: %s' % prediction)


if __name__ == "__main__":
    main()
