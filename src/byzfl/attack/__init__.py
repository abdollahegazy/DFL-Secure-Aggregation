"""
Byzantine attacks.

Each attack mutates the malicious rows of the bank's params in place:
    (bank: NodeBank, topology: Topology, **kwargs) -> None

The mutation happens under torch.no_grad() and preserves tensor identity so
the optimizer state inside the bank survives across rounds.
"""
from .signflip import signflip
from .noise import add_noise
from .random_noise import set_random

ATTACKS = {
    "signflip": signflip,
    "add_noise": add_noise,
    "set_random": set_random,
}

__all__ = [
    "signflip",
    "add_noise",
    "set_random",
    "ATTACKS",
]
