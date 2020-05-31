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


def create_image_object(image_bytes):
    """Create a PIL.Image object that contains the image bytes."""
    f = io.BytesIO()
    f.write(image_bytes)
    image = Image.open(f)
    return image


def to_jpeg(image_object, mode='RGB'):
    """Convert the given PIL.Image object's format into `JPEG`."""
    f = io.BytesIO()
    image_object.convert(mode).save(f, format='JPEG')
    image = Image.open(f)
    return image


def transform(image):
    """Transforms the image to torch.Tensor."""
    tensor = torchvision.transforms.ToTensor()(image)
    tensor = torchvision.transforms.Normalize(mean=(0.1307,), std=(0.3081,))(tensor)
    return tensor


def preprocess(image_bytes):
    """Preprocesses image bytes so that they can be input into the model.

    Returns:
    - a torch.Tensor of which shape is [1, 1, 28, 28].
    """
    image = create_image_object(image_bytes)
    image = to_jpeg(image, mode='L')  # 'L' for grayscale
    image = image.resize(size=(28, 28))
    image = np.array(image, dtype=np.uint8)
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    return tensor


def predict(model_path, image_bytes):
    """Predicts a digit the image represents."""
    tensor = preprocess(image_bytes)

    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_path)
    model = MnistNetwork().to(device)
    model.load_state_dict(checkpoint)

    output = model(tensor.to(device))
    prediction = output.argmax().item()
    return prediction


def main(model_path, image_path):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    prediction = predict(model_path, image_bytes)
    print('prediction: %s' % prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="path to MNIST model")
    parser.add_argument('--image', type=str, required=True, help="path to an image")
    args = parser.parse_args()
    main(args.model_path, args.image)
