from torchvision import datasets, transforms
from .databundle import DataBundle

def load_cifar10(data_dir:str,
                 transform: transforms.Compose | None = None,
                download=True) -> DataBundle:
    
    train = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=transform,
        download=download,
    )

    test = datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=transform,
        download=download,
    )   

    return DataBundle(train=train, test=test)