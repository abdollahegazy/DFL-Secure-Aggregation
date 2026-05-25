"""ResNet-20 for CIFAR-10 with GroupNorm (vmap-safe; BatchNorm running stats break vmap+FL)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c: int) -> nn.GroupNorm:
    # 8 groups when possible; degrades gracefully for tiny channel counts.
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class BasicBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.gn1 = _gn(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.gn2 = _gn(out_c)
        if stride == 1 and in_c == out_c:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                _gn(out_c),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return F.relu(h + self.shortcut(x))


class ResNet20(nn.Module):
    """3 stages of 3 BasicBlocks each: 16 -> 32 -> 64 channels."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.gn1 = _gn(16)
        self.stage1 = nn.Sequential(*[BasicBlock(16, 16) for _ in range(3)])
        self.stage2 = nn.Sequential(
            BasicBlock(16, 32, stride=2),
            BasicBlock(32, 32),
            BasicBlock(32, 32),
        )
        self.stage3 = nn.Sequential(
            BasicBlock(32, 64, stride=2),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)
