import torch
import torch.nn.functional as F
import numpy as np
from torchvision import datasets

_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
_STD = torch.tensor([0.2470, 0.2435, 0.2616])


def load_cifar10(data_dir: str, device: torch.device, download: bool = True):
    train_ds = datasets.CIFAR10(data_dir, train=True, download=download)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=download)

    def to_tensor(ds) -> torch.Tensor:
        x = (
            torch.from_numpy(np.array(ds.data))
            .permute(0, 3, 1, 2)
            .to(device=device, dtype=torch.float32)
            / 255.0
        )
        mean = _MEAN.to(device).view(1, 3, 1, 1)
        std = _STD.to(device).view(1, 3, 1, 1)
        return x.sub_(mean).div_(std)

    x_train = to_tensor(train_ds)
    x_test = to_tensor(test_ds)
    y_train = torch.tensor(train_ds.targets, dtype=torch.long, device=device)
    y_test = torch.tensor(test_ds.targets, dtype=torch.long, device=device)

    return x_train, y_train, x_test, y_test

@torch.compile(mode="default")
def cifar10_augment(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample random horizontal flip + pad-4 reflect-crop, applied to a
    stacked (N, B, 3, 32, 32) batch. Returns (x', y) with the same shape.

    Each of the N*B samples gets its own coin flip and crop offset, which is
    what the centralized CIFAR-10 recipe uses to get ResNet-20 from ~80% to ~88%.
    """
    N, B, C, H, W = x.shape
    NB = N * B
    dev = x.device

    # 1. Random horizontal flip per (N, B).
    flip = torch.rand(N, B, 1, 1, 1, device=dev) < 0.5
    x = torch.where(flip, x.flip(-1), x)

    # 2. Reflect-pad 4 on each side -> (NB, 3, 40, 40).
    x_flat = x.reshape(NB, C, H, W)
    x_pad = F.pad(x_flat, (4, 4, 4, 4), mode="reflect")

    # 3. Per-sample crop offsets, gathered via advanced indexing.
    h_off = torch.randint(0, 9, (NB,), device=dev)
    w_off = torch.randint(0, 9, (NB,), device=dev)
    sample = torch.arange(NB, device=dev).view(NB, 1, 1, 1)
    chan = torch.arange(C, device=dev).view(1, C, 1, 1)
    row = h_off.view(NB, 1, 1, 1) + torch.arange(H, device=dev).view(1, 1, H, 1)
    col = w_off.view(NB, 1, 1, 1) + torch.arange(W, device=dev).view(1, 1, 1, W)
    x_cropped = x_pad[sample, chan, row, col]                                # (NB, 3, 32, 32)

    return x_cropped.view(N, B, C, H, W), y
