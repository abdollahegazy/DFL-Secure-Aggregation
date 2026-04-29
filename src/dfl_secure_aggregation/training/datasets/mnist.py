from torchvision import datasets, transforms
from .databundle import DataBundle


def load_mnist(
    data_dir: str,
    download: bool = True,
    transform: transforms.Compose | None =
    transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
) -> DataBundle:

    train = datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=download,
    )

    test = datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transform,
        download=download,
    )

    return DataBundle(train=train, test=test)
