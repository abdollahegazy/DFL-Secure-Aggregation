"""
byzfl: vectorized decentralized federated learning with Byzantine-robust aggregators.

Public API surfaced here is everything you'd need for a typical experiment driver:
    from byzfl import NodeBank, Topology, run_simulation
    from byzfl import NodeDataLoader, sliding_window_partition
    from byzfl import STRATEGIES, ATTACKS

Submodules:
    byzfl.aggregation  - fedavg, krum, geomed, trimmedmean, median (+ STRATEGIES registry)
    byzfl.attack       - signflip, noise (+ ATTACKS registry)
    byzfl.network      - Topology, graph generators
"""
from .nodebank import NodeBank
from .simulator import run_simulation
from .data import NodeDataLoader, sliding_window_partition
from .network import Topology
from .aggregation import STRATEGIES
from .attack import ATTACKS

__all__ = [
    "NodeBank",
    "Topology",
    "run_simulation",
    "NodeDataLoader",
    "sliding_window_partition",
    "STRATEGIES",
    "ATTACKS",
]
