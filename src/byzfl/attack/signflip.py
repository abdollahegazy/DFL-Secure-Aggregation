import torch

from ..network import Topology
from ..nodebank import NodeBank

@torch.no_grad()
def signflip(bank: NodeBank, topology: Topology):
    """Flip the sign of the gradient."""
    for v in bank.params.values():
        v[topology.malicious_mask] = torch.neg(v[topology.malicious_mask])