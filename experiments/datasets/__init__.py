"""Dataset registry: name -> loader(data_dir, device) -> (x_train, y_train, x_test, y_test).

AUGMENTS is the per-dataset training-time augmentation function (or None).
Signature: augment_fn(x, y) -> (x', y'), where x is (N, B, *) on GPU.
"""
from .mnist import load_mnist
from .cifar10 import load_cifar10, cifar10_augment

DATASETS = {
    "mnist": load_mnist,
    "cifar10": load_cifar10,
}

AUGMENTS = {
    "mnist": None,
    "cifar10": cifar10_augment,
}

__all__ = ["load_mnist", "load_cifar10", "cifar10_augment", "DATASETS", "AUGMENTS"]
