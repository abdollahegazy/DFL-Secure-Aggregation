from dataclasses import dataclass
import torch

@dataclass
class Topology:
    n:int

    # edge matrix (2,2E) directed
    edges: torch.Tensor

    # (N,) bool mask of which nodes are malicious
    malicious_mask: torch.Tensor
    device: torch.device

    @classmethod
    def from_pt(cls, path, device: torch.device) -> "Topology":
        data = torch.load(path, weights_only=True)
        return cls(
            n=data["n"],
            edges=data["edge_index"].to(device),
            malicious_mask=data["malicious_mask"].to(device),
            device=device,
        )

    def adjacency(self) -> torch.Tensor:
        """(N, N) bool, symmetric. from directed edge matrix"""
        adj = torch.zeros(self.n, self.n, dtype=torch.bool, device=self.device)
        if self.edges.numel() > 0:
            adj[self.edges[0], self.edges[1]] = True
        return adj

    def with_self_loops(self) -> torch.Tensor:
        """(N, N) bool, adjacency() | I.  This is so nodes can aggregate with themselves."""
        return self.adjacency() | torch.eye(self.n, dtype=torch.bool, device=self.device)

    def benign_view_adjacency(self) -> torch.Tensor:
        """
        Adjacency with edges to malicious nodes zeroed out. 
        """
        return self.adjacency() & ~self.malicious_mask.unsqueeze(0)


    def candidate_masks(self) -> torch.Tensor:
        """(N, N) row-stochastic mixing matrix encoding the malicious-aggregation rule.
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