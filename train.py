import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from dataset import get_mnist_data_loader
from networks import MnistNetwork


def train_step(train_loader, model, optimizer, device, log_interval):
    model.train()
    for batch_index, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()

        if batch_index % log_interval == 0:
            print("[%5s/%5s] %5.2f%% | Loss: %.6f"
                  % (batch_index*len(data), len(train_loader.dataset),
                     100.0*batch_index / len(train_loader), loss.item()))


def test_step(test_loader, model, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, label, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = (100.0 * correct) / len(test_loader.dataset)
    return test_loss, test_accuracy


def train(train_batch_size, test_batch_size, epochs, lr, output_model_name, log_interval=100):
    train_loader = get_mnist_data_loader(train=True, batch_size=train_batch_size)
    test_loader = get_mnist_data_loader(train=False, batch_size=test_batch_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MnistNetwork().to(device)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=lr
    )

    for epoch in range(1, epochs+1):
        print('\nEpoch %s' % epoch)
        train_step(train_loader, model, optimizer, device, log_interval)
        test_loss, test_accuracy = test_step(test_loader, model, device)
        print("Test loss: %.6f, Test accuracy: %.2f%%" % (test_loss, test_accuracy))

    dirname = os.path.dirname(output_model_name)
    if dirname and not os.path.isdir(dirname):
        os.makedirs(dirname)
    torch.save(model.state_dict(), output_model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--output_model_name', type=str, default='models/mnist_cnn.pt')
    args = vars(parser.parse_args())
    train(**args)
