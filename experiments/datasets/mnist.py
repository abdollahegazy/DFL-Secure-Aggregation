"""Load MNIST as raw GPU tensors for vmap-based training.

Returns (x_train, y_train, x_test, y_test). Pixels normalized to [-1, 1].
"""
import torch
from torchvision import datasets


def load_mnist(data_dir: str, device: torch.device, download: bool = True):
    train_ds = datasets.MNIST(data_dir, train=True, download=download)
    test_ds = datasets.MNIST(data_dir, train=False, download=download)

    x_train = ((train_ds.data.float() / 255.0) - 0.5) / 0.5
    x_test = ((test_ds.data.float() / 255.0) - 0.5) / 0.5
    x_train = x_train.unsqueeze(1).to(device)          # (60000, 1, 28, 28)
    x_test = x_test.unsqueeze(1).to(device)            # (10000, 1, 28, 28)
    y_train = train_ds.targets.to(device)              # (60000,)
    y_test = test_ds.targets.to(device)                # (10000,)
    return x_train, y_train, x_test, y_test
