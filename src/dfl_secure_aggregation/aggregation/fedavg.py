import torch
from torch import Tensor


from ..network import Topology


def fedavg(params: dict[str, Tensor], topology: Topology) -> dict[str, Tensor]:
    """Per-node weighted average over each node's candidate set.

    Benign node i:    neighbors = neighbors(i) U {i}
    Malicious node i: neighbors = benign neighbors(i)  (no self, no other malicious)

    Implemented as one matmul per parameter: new[k] = W @ params[k].flatten(1).
    """
    W = _mixing_matrix(topology)
    return {k: (W @ v.flatten(1)).reshape_as(v) for k, v in params.items()}


def _mixing_matrix(topology: Topology) -> Tensor:
    candidates = topology.candidate_masks()  # (N, N) bool
    row_sums = candidates.sum(dim=1, keepdim=True).float()
    return candidates.float() / row_sums

