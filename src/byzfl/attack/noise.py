import torch

from ..network import Topology
from ..nodebank import NodeBank

@torch.no_grad()
def add_noise(
    bank: NodeBank,
    topology: Topology,
    strength: float = 1.0,
):
    """
    this ADDS random noise to the parameters 
    of malicious nodes
    """
    for v in bank.params.values():
        v[topology.malicious_mask] += torch.randn_like(v[topology.malicious_mask]) * strength
    
    