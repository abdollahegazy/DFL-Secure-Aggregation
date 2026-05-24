"""
Network topology: tensor-native runtime representation + graph generators.

generate.py builds graphs (NetworkX under the hood, never imported in the hot path).
topology.py is the runtime dataclass that aggregators and attacks consume.
"""
from .generate import small_world_graph, scale_free_graph, random_graph, save
from .topology import Topology

__all__ = [
    "Topology",
    "random_graph",
    "small_world_graph",
    "scale_free_graph",
    "save",
]
