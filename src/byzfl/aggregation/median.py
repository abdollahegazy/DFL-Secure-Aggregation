import torch


from torch import Tensor
from typing import Dict
from ..network import Topology


@torch.no_grad()
def median(
    params: Dict[str, Tensor],  # param_name -> (N, *param_shape)
    topology: Topology,
) -> Dict[str, Tensor]:
    """Per-node coordinate-wise median over each node's candidate set."""
    n = topology.n
    cand_indices = topology._cand_indices
    new_params = {k: v.clone() for k, v in params.items()}
    for i in range(n):
        for k, v in params.items():
            cands = v[cand_indices[i]]
            m = cands.shape[0]
            if m % 2 == 1:
                new_params[k][i] = cands.median(dim=0).values
            else:
                sorted_cands, _ = cands.sort(dim=0)
                new_params[k][i] = (sorted_cands[m // 2 - 1] + sorted_cands[m // 2]) / 2
    return new_params