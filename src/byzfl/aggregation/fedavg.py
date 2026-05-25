import torch
from torch import Tensor


from ..network import Topology

@torch.no_grad()
def fedavg(params: dict[str, Tensor], topology: Topology) -> dict[str, Tensor]:
    """Per-node weighted average over each node's candidate set.

    Benign node i:    neighbors = neighbors(i) U {i}
    Malicious node i: neighbors = benign neighbors(i)  (no self, no other malicious)

    Implemented as one matmul per parameter: new[k] = W @ params[k].flatten(1).
    """
    W = topology.mixing_matrix
    return {k: (W @ v.flatten(1)).reshape_as(v) for k, v in params.items()}

