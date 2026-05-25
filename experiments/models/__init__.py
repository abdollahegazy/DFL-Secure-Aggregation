"""Per-dataset model registry. Keyed by dataset name so the sweep can pick the right net."""
from .mnist_net import MNISTNet
from .resnet20 import ResNet20

MODELS = {
    "mnist": MNISTNet,
    "cifar10": ResNet20,
}

__all__ = ["MNISTNet", "ResNet20", "MODELS"]
