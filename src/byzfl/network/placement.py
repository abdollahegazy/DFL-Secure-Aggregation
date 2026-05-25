"""Malicious-node placement strategies, applied on top of a built graph.

Each `_pick_*` function takes the already-built NetworkX graph and returns the
list of malicious node ids. `apply_placement` is the dispatcher that
generators use to pick a strategy based on (topology_kind, placement) and
slot the result into the packed Topology dict.
"""
import random

import networkx as nx


def _pick_random(num_nodes: int, count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return rng.sample(range(num_nodes), count)


def _pick_high_degree(g: nx.Graph, count: int) -> list[int]:
    return sorted(g.nodes, key=lambda u: g.degree(u), reverse=True)[:count]


def _pick_small_world_strategic(g: nx.Graph, count: int, num_nodes: int, k: int) -> list[int]:
    """Endpoints of rewired long-range edges, padded with high-degree if needed.

    In a Watts-Strogatz graph any edge with |u - v| > k must have been rewired
    (the base ring only connects within k hops). Targeting one endpoint of each
    rewired edge is the legacy paper's strategic placement.
    """
    malicious: list[int] = []
    seen: set[int] = set()
    rewired: set[tuple[int, int]] = set()

    for u in g.nodes:
        for v in g.neighbors(u):
            edge = tuple(sorted((u, v)))
            if edge in rewired:
                continue
            if abs(u - v) > k:
                rewired.add(edge)
                if u not in seen:
                    malicious.append(u)
                    seen.add(u)
                break
        if len(malicious) >= count:
            return malicious[:count]

    for u in _pick_high_degree(g, num_nodes):
        if u not in seen:
            malicious.append(u)
            seen.add(u)
        if len(malicious) >= count:
            break
    return malicious[:count]


def apply_placement(
    g: nx.Graph,
    *,
    num_nodes: int,
    topology_kind: str,
    placement: str,
    malicious_proportion: float,
    seed: int,
    small_world_k: int | None = None,
) -> list[int]:
    """Pick malicious nodes for a built graph.

    placement="random": uniform sample of `num_nodes * malicious_proportion`.
    placement="strategic": small-world uses rewired-edge endpoints, others use
        highest-degree nodes.
    """
    count = int(num_nodes * malicious_proportion)
    if count == 0:
        return []
    if placement == "random":
        return _pick_random(num_nodes, count, seed)
    if placement != "strategic":
        raise ValueError(f"Unsupported placement: {placement}")
    if topology_kind == "small_world":
        if small_world_k is None:
            raise ValueError("small_world_k required for strategic placement on small-world")
        return _pick_small_world_strategic(g, count, num_nodes, small_world_k)
    return _pick_high_degree(g, count)
