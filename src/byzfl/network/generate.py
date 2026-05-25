import math
import random
from pathlib import Path

import networkx as nx
import torch

from .placement import apply_placement


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def save(topology: dict, path: str | Path) -> None:
    torch.save(topology, path)


def _to_edge_index(g: nx.Graph) -> torch.Tensor:
    """Undirected NetworkX graph → (2, 2E) edge_index with both directions."""
    edges = list(g.edges)
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    both = edges + [(v, u) for u, v in edges]
    return torch.tensor(both, dtype=torch.long).t().contiguous()


def _malicious_mask(n: int, malicious_nodes: list[int]) -> torch.Tensor:
    mask = torch.zeros(n, dtype=torch.bool)
    if malicious_nodes:
        mask[list(malicious_nodes)] = True
    return mask


def _pack(g: nx.Graph, network_type: str, malicious_nodes: list[int]) -> dict:
    n = g.number_of_nodes()
    return dict(
        edge_index=_to_edge_index(g),
        n=n,
        malicious_mask=_malicious_mask(n, malicious_nodes),
        network_type=network_type,
    )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def random_graph(
    num_nodes: int,
    edge_density: float,
    *,
    malicious_proportion: float = 0.0,
    placement: str = "random",
    seed: int | None = None,
) -> dict:
    """Random graph where every node targets degree = ceil(edge_density * N)."""
    rng = random.Random(seed)
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))

    target_deg = math.ceil(edge_density * num_nodes)
    for u in range(num_nodes):
        existing = set(g.neighbors(u))
        needed = target_deg - len(existing)
        candidates = [v for v in range(num_nodes) if v != u and v not in existing]
        for v in rng.sample(candidates, min(needed, len(candidates))):
            g.add_edge(u, v)

    mal_nodes = apply_placement(
        g,
        num_nodes=num_nodes,
        topology_kind="random",
        placement=placement,
        malicious_proportion=malicious_proportion,
        seed=seed if seed is not None else 0,
    )
    return _pack(g, "random", mal_nodes)


def small_world_graph(
    num_nodes: int,
    k: int,
    beta: float,
    *,
    malicious_proportion: float = 0.0,
    placement: str = "random",
    seed: int | None = None,
) -> dict:
    """Watts-Strogatz small-world graph."""
    g = nx.watts_strogatz_graph(num_nodes, k, beta, seed=seed)
    mal_nodes = apply_placement(
        g,
        num_nodes=num_nodes,
        topology_kind="small_world",
        placement=placement,
        malicious_proportion=malicious_proportion,
        seed=seed if seed is not None else 0,
        small_world_k=k,
    )
    return _pack(g, "small_world", mal_nodes)


def scale_free_graph(
    num_nodes: int,
    m: int,
    *,
    malicious_proportion: float = 0.0,
    placement: str = "random",
    seed: int | None = None,
) -> dict:
    """Barabási-Albert scale-free graph."""
    g = nx.barabasi_albert_graph(num_nodes, m, seed=seed)
    mal_nodes = apply_placement(
        g,
        num_nodes=num_nodes,
        topology_kind="scale_free",
        placement=placement,
        malicious_proportion=malicious_proportion,
        seed=seed if seed is not None else 0,
    )
    return _pack(g, "scale_free", mal_nodes)
