import torch

from ..network import Topology
from ..nodebank import NodeBank

def signflip(bank: NodeBank, topology: Topology):
    """Flip the sign of the gradient."""
    with torch.no_grad():
        for v in bank.params.values():
            v[topology.malicious_mask] = torch.neg(v[topology.malicious_mask])
