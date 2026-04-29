from .mnist import load_mnist
from .cifar10 import load_cifar10

DATASET_LOADERS = {
    "mnist": load_mnist,
    "cifar10": load_cifar10,
}

def load_dataset_by_name(name: str, **kwargs):
    try:
        return DATASET_LOADERS[name](**kwargs)
    except KeyError:
        raise ValueError(f"Dataset not supported: {name}")
