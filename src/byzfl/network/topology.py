from dataclasses import dataclass
import torch

from torch import Tensor
from functools import cached_property

@dataclass
class Topology:
    n:int

    # edge matrix (2,2E) directed
    edges: Tensor

    # (N,) bool mask of which nodes are malicious
    malicious_mask: Tensor
    device: torch.device

    @classmethod
    def from_pt(cls, path, device: torch.device) -> "Topology":
        return cls.from_dict(torch.load(path, weights_only=True), device)

    @classmethod
    def from_dict(cls, data: dict, device: torch.device) -> "Topology":
        """Build from a generator output dict (edge_index, n, malicious_mask, ...)."""
        return cls(
            n=data["n"],
            edges=data["edge_index"].to(device),
            malicious_mask=data["malicious_mask"].to(device),
            device=device,
        )

    def adjacency(self) -> Tensor:
        """(N, N) bool, symmetric. from directed edge matrix"""
        adj = torch.zeros(self.n, self.n, dtype=torch.bool, device=self.device)
        if self.edges.numel() > 0:
            adj[self.edges[0], self.edges[1]] = True
        return adj

    def with_self_loops(self) -> Tensor:
        """(N, N) bool, adjacency() | I.  This is so nodes can aggregate with themselves."""
        return self.adjacency() | torch.eye(self.n, dtype=torch.bool, device=self.device)

    def benign_view_adjacency(self) -> Tensor:
        """
        Adjacency with edges to malicious nodes zeroed out. 
        """
        return self.adjacency() & ~self.malicious_mask.unsqueeze(0)

    @cached_property
    def _candidate_mask(self) -> Tensor:
        return self.candidate_mask()

    def candidate_mask(self) -> Tensor:
        """
        (N, N) bool. Row i: nodes that node i can aggregate from.
        Encodes the malicious-aggregation rule and empty-set fallback.

        Row i:
        - benign:    1/|cand_i| over neighbors(i) U {i}
        - malicious: 1/|cand_i| over benign neighbors(i)
        - empty cand (malicious with zero benign neighbors): self-loop, preserve own params
        """
        is_mal_row = self.malicious_mask.unsqueeze(1)  # (N, 1)

        cand = torch.where(
            is_mal_row,
            self.benign_view_adjacency(),  # malicious rows
            self.with_self_loops(),  # benign rows
        )

        # Fallback self-loop for any row with no candidates.
        no_cand = ~cand.any(dim=1)
        if no_cand.any():
            eye = torch.eye(self.n, dtype=torch.bool, device=self.device)
            cand = cand | (no_cand.unsqueeze(1) & eye)
        return cand
    
    @cached_property
    def _cand_indices(self) -> list[Tensor]:
        """List of length N. Row i: indices of nodes in node i's candidate set."""
        return [self._candidate_mask[i].nonzero(as_tuple=False).squeeze(1) for i in range(self.n)]
    

    @cached_property
    def mixing_matrix(self) -> Tensor:
        candidates = self._candidate_mask
        row_sums = candidates.sum(dim=1, keepdim=True).float()
        return candidates.float() / row_sums
    

    @cached_property
    def cand_padded(self) -> tuple[Tensor, Tensor]:
        """Vectorized candidate gather. Returns (indices, mask) of shape (N, M_max).
        indices[i, :|cand_i|] are valid node ids; rest are 0 (safe, masked out).
        """
        from torch.nn.utils.rnn import pad_sequence
        padded = pad_sequence(self._cand_indices, batch_first=True, padding_value=0)
        sizes = torch.tensor([c.numel() for c in self._cand_indices], device=self.device)
        arange = torch.arange(padded.size(1), device=self.device).unsqueeze(0)
        mask = arange < sizes.unsqueeze(1)
        return padded, mask