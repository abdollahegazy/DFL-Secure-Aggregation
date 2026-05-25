import torch

from typing import Dict
from ..network import Topology

def weighted_average(x: torch.Tensor, w: torch.Tensor):
    # first dim of x is node dim, same shape as w then broadcast to param shape
    return torch.einsum("i,i...->...", w, x) / w.sum()

def _geomed_along_candidates(
    candidates: torch.Tensor,  #(M,*param_shape)
    maxiter:int,
    eps:float,
    ftol:float,
):
    """
    Weiszfeld alg for geometric median.
    """
    m = candidates.shape[0]
    weights = torch.ones(m, device=candidates.device)
    median = weighted_average(candidates, weights)
    def objective(median):
        dists = (candidates - median).reshape(m, -1).norm(dim=1) # (M,)
        return (weights*dists).sum() / weights.sum()
    prev = objective(median)

    for _ in range(maxiter):
        dists = (candidates - median).reshape(m, -1).norm(dim=1).clamp(min=eps) # (M,)
        new_weights = weights / dists
        median = weighted_average(candidates, new_weights)
        curr = objective(median)
        if abs(prev - curr) <= ftol*curr:
            break
        prev = curr
    return median


@torch.no_grad()
def geomed(
    params: Dict[str, torch.Tensor],
    topology: Topology,
    maxiter: int = 100,
    eps: float = 1e-6,
    ftol: float = 1e-10,  # kept for back-compat; unused in vectorized path
):
    cand_padded, cand_mask = topology.cand_padded            # (N, M_max), (N, M_max)
    N, M_max = cand_padded.shape
    mask_f = cand_mask.float()                               # (N, M_max)
    valid_counts = mask_f.sum(dim=1, keepdim=True)           # (N, 1)

    new_params = {}
    for k, v in params.items():
        param_shape = v.shape[1:]
        P = v[0].numel()
        cands = v[cand_padded].reshape(N, M_max, P)          # (N, M_max, P)

        # Initial median = mean over valid entries.
        median = (mask_f.unsqueeze(-1) * cands).sum(dim=1) / valid_counts  # (N, P)

        prev = torch.full((N,), float("inf"), device=v.device)
        for _ in range(maxiter):
            diff = cands - median.unsqueeze(1)               # (N, M_max, P)
            dists = diff.norm(dim=2).clamp(min=eps)          # (N, M_max)
            w = mask_f / dists                               # (N, M_max), invalid -> 0
            median = (w.unsqueeze(-1) * cands).sum(dim=1) / w.sum(dim=1, keepdim=True)
            curr = (mask_f * dists * w).sum(dim=1) / w.sum(dim=1)  # (N,)

            rel = torch.abs(prev - curr) / curr.clamp_min(eps)
            if rel.max() <= ftol:
                break

            prev = curr

        new_params[k] = median.reshape((N,) + param_shape)
    return new_params


# @torch.no_grad()
# def geomed(
#     params: Dict[str, torch.Tensor],  # param_name -> (N, *param_shape)
#     topology: Topology,
#     maxiter: int = 100,
#     eps: float = 1e-6,
#     ftol: float = 1e-10,
# ):
#     cand_indices = topology._cand_indices

#     new_params = {k: v.clone() for k, v in params.items()}
#     for i in range(topology.n):
#         for k, v in params.items():
#             new_params[k][i] = _geomed_along_candidates(v[cand_indices[i]], maxiter, eps, ftol)
#     return new_params



# this is a simplified version from claude but kinda confusing 
# to read but i did validate they produce same output using a small test case
# also were using geomed per layer, but some methods might do it for whole param vec?
# def geomed(params, topology, maxiter=30, eps=1e-6):
#     mask = topology.candidate_mask().float()  # (N, N)
#     denom_init = mask.sum(dim=1, keepdim=True).clamp_min(1.0)

#     out = {}
#     for k, v in params.items():
#         flat = v.flatten(1)  # (N, P_layer)
#         median = (mask @ flat) / denom_init  # (N, P_layer)

#         for _ in range(maxiter):
#             f_sq = (flat * flat).sum(dim=1)
#             m_sq = (median * median).sum(dim=1)
#             dists = (
#                 (m_sq[:, None] + f_sq[None, :] - 2 * median @ flat.T)
#                 .clamp_min(eps * eps)
#                 .sqrt()
#             )
#             w = mask / dists
#             median = (w @ flat) / w.sum(dim=1, keepdim=True).clamp_min(eps)

#         out[k] = median.reshape_as(v)
#     return out