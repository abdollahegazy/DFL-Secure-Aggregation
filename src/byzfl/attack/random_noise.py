import torch

from ..network import Topology
from ..nodebank import NodeBank

@torch.no_grad()
def set_random(
    bank: NodeBank,
    topology: Topology,
):
    """
    This replaces parameters with random weight
    """
    for v in bank.params.values():
        v[topology.malicious_mask] = torch.randn_like(v[topology.malicious_mask])

