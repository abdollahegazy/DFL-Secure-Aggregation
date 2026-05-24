"""
Byzantine-robust aggregators.

Each aggregator is a pure function with signature:
    (params: dict[str, Tensor], topology: Topology, **kwargs) -> dict[str, Tensor]

It reads node parameters as stacked tensors (N, *param_shape), applies its
per-node combining rule (encoded in topology.candidate_mask()), and returns a
fresh dict of the same shape. Caller is expected to pipe the result back into
the bank via bank.load_params(...).
"""
from .fedavg import fedavg
from .geomed import geomed
from .krum import krum
from .median import median
from .trimmedmean import trimmedmean

STRATEGIES = {
    "fedavg": fedavg,
    "geomed": geomed,
    "krum": krum,
    "median": median,
    "trimmedmean": trimmedmean,
}

__all__ = [
    "fedavg",
    "geomed",
    "krum",
    "median",
    "trimmedmean",
    "STRATEGIES",
]
