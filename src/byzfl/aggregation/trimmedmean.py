import torch

from torch import Tensor
from typing import Dict

from ..network import Topology

def _trimmedmean_along_candidates(
    candidates: Tensor,  #(M,*param_shape)
    beta: float
):
    m = candidates.shape[0]
    num_trim = int(m * beta)
    if 2*num_trim >= m:
        return candidates.mean(dim=0)
    sorted_candidates, _ = torch.sort(candidates, dim=0)
    trimmed_candidates = sorted_candidates[num_trim : m - num_trim]
    return trimmed_candidates.mean(dim=0)

@torch.no_grad()
def trimmedmean(
    params: Dict[str, Tensor],  # param_name -> (N, *param_shape)
    topology: Topology,
    beta: float = 0.2,
) -> Dict[str, Tensor]:
    """Per-node trimmed mean over each node's candidate set."""
    n = topology.n
    cand_indices = topology._cand_indices
    new_params = {k: v.clone() for k, v in params.items()}
    for i in range(n):
        for k, v in params.items():
            new_params[k][i] = _trimmedmean_along_candidates(v[cand_indices[i]], beta)
    return new_params