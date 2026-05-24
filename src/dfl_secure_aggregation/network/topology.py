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