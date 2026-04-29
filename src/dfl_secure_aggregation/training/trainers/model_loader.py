from .digit_classifier import DigitClassifier
from .cifar10 import CIFAR10Trainer

MODEL_REGISTRY = {
    "mnist": DigitClassifier,
    "cifar10": CIFAR10Trainer,
}

def load_model_by_name(model_name: str):
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError:
        raise ValueError(f"Model not supported: {model_name}")
