import torch
from torch import Tensor
import sys

import os
sys.path.append(os.path.abspath(".."))
from ..network.topology import Topology


def fedavg(params: dict[str, Tensor], topology: Topology) -> dict[str, Tensor]:
    """Per-node weighted average over each node's candidate set.

    Benign node i:    neighbors = neighbors(i) U {i}
    Malicious node i: neighbors = benign neighbors(i)  (no self, no other malicious)

    Implemented as one matmul per parameter: new[k] = W @ params[k].flatten(1).
    """
    W = _mixing_matrix(topology)
    return {k: (W @ v.flatten(1)).reshape_as(v) for k, v in params.items()}


def _mixing_matrix(topology: Topology) -> Tensor:
    """(N, N) row-stochastic mixing matrix encoding the malicious-aggregation rule.

    Row i:
      - benign:    1/|cand_i| over neighbors(i) U {i}
      - malicious: 1/|cand_i| over benign neighbors(i)
      - empty cand (malicious with zero benign neighbors): self-loop, preserve own params
    """
    n = topology.n
    device = topology.device
    is_mal_row = topology.malicious_mask.unsqueeze(1)  # (N, 1)

    cand = torch.where(
        is_mal_row,
        topology.benign_view_adjacency(),  # malicious rows
        topology.with_self_loops(),  # benign rows
    )

    # Fallback self-loop for any row with no candidates.
    no_cand = ~cand.any(dim=1)
    if no_cand.any():
        eye = torch.eye(n, dtype=torch.bool, device=device)
        cand = cand | (no_cand.unsqueeze(1) & eye)

    row_sums = cand.sum(dim=1, keepdim=True).float()
    return cand.float() / row_sums

