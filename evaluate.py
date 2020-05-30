import argparse

import torch

from dataset import get_mnist_data_loader
from networks import MnistNetwork
from train import test_step


def evaluate(model_path, batch_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_path)
    model = MnistNetwork().to(device)
    model.load_state_dict(checkpoint)
    test_loader = get_mnist_data_loader(train=False, batch_size=batch_size)

    test_loss, test_accuracy = test_step(test_loader, model, device)
    print("Test loss: %.6f, Test accuracy: %.2f%%" % (test_loss, test_accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help="path to MNIST model")
    parser.add_argument('--batch_size', type=int, default=50)
    args = vars(parser.parse_args())
    evaluate(**args)
