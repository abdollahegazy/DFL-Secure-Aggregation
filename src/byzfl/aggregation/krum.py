import torch

from typing import Dict
from ..network import Topology




@torch.no_grad()
def krum(
    params: Dict[str, torch.Tensor],  # param_name -> (N, *param_shape)
    topology: Topology,
    f:int = 0, # number of Byzantine count among candidates
    m_select: int | None = None, # number of candidates to select, default 1 for original krum but multi-krum variants select more
) -> Dict[str, torch.Tensor]:
    """
    For each node i, score every candidate c in i's neighborhood by the sum of
    c's (m - f - 2) smallest squared L2 distances to *other* candidates in i's
    neighborhood, where m = |candidates(i)| and f is the assumed upper bound on
    Byzantine neighbors. The candidate with minimum score is selected, and its
    full parameters replace node i's parameters.

    Multi-Krum (m_select>1): average the m_select lowest-scoring candidates.
    """

    cand_mask = topology.candidate_mask()  # (N, N) bool
    cand_indices = [cand_mask[i].nonzero(as_tuple=False).squeeze(1) for i in range(topology.n)]

    new_params = {k: v.clone() for k, v in params.items()}


    for i in range(topology.n):
        cand_idx = cand_indices[i]
        m = cand_idx.numel()

        if m == 1:
            chosen = cand_idx[0]
            for k,v in params.items():
                new_params[k][i] = v[chosen]

            continue

        # (M, P_total) where M = num candidates and P_total = total num params across all layers
        flat = torch.cat(
            [params[k][cand_idx].flatten(1) for k in params], 
            dim=1
        )

        #faster than cdist bc avoid doing sqrt that we dont need
        # its jsut doing ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x@y
        sq_norms = (flat * flat).sum(dim=1)
        D = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (flat @ flat.T)
        D = D.clamp_min(0.0)  # numerical floor; sqrt isn't needed
        D.fill_diagonal_(float("inf"))  # ignore self-distance

        # krum score: sum of m-f-2 closest distances
        k_smallest = max(1,min(m-1, m-f-2))  # handle edge cases where m is small
        scores = D.topk(k_smallest, dim=1, largest=False).values.sum(dim=1)  # (M,)

        if m_select is None:
            n_winners = 1 # normal krum
        else:
            n_winners = min(m_select, m-f) 
        n_winners = max(1, n_winners)  # handle edge case where m-f <= 0
        winners_local = scores.topk(n_winners, largest=False).indices  # (n_winners,)
        winners_global = cand_idx[winners_local]  # (n_winners,)

        for k, v in params.items():
            new_params[k][i] = v[winners_global].mean(dim=0)
    return new_params
