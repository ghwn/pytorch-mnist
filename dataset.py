import torch
import torchvision


def get_mnist_data_loader(train, batch_size, root='./datasets'):
    gpu_options = {
        "num_workers": 1,
        "pin_memory": True
    } if torch.cuda.is_available() else {}

    data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
            ])
        ),
        batch_size=batch_size,
        shuffle=True,
        **gpu_options
    )
    return data_loader
