"""
Vectorized DFL.

NodeBank owns N copies of a model's parameters as stacked tensors and runs
forward/backward/step over all N simultaneously via torch.func.vmap.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, stack_module_state, vmap
 
from typing import Sequence

class NodeBank:
    """
    N copies of a model as stacked parameter tensors.

    Each model must be identical beises its weights in shape and stuff.
    Also, each model batch_size must be the same for forward to work.


    This allows us to do forward/backward/step for all N models simultaneously via vmap, 
    which is much faster than what we prev had of N individual models and loops over them.
    """

    def __init__(
        self,
        models: Sequence[nn.Module],
        device: torch.device,
        optimizer_cls: type[torch.optim.Optimizer],
        optimizer_kwargs = {'lr': 1e-3, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 1e-2},
        
    ):
        self.n = len(models)
        self.device = device

        for m in models:
            m.to(device)

        # Template is a structural skeleton only; meta device means no storage.
        self.template = copy.deepcopy(models[0]).to("meta")
        self.params, self.buffers = stack_module_state(models)

        self.optimizer = optimizer_cls(
            self.params.values(), **optimizer_kwargs
        )

        self._forward_fn = vmap(self._functional_call, in_dims=(0, 0, 0))
        self.compiled_forward = torch.compile(self._forward_fn,mode='default')
        
        self._forward_shared_fn = vmap(self._functional_call, in_dims=(0, 0, None))
        self.compiled_forward_shared = torch.compile(self._forward_shared_fn,mode='default')
        



    def _functional_call(self, params, buffers, x):
        return functional_call(self.template, (params, buffers), (x,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # slice across N for params, buffers, and inputs
        return self.compiled_forward(self.params, self.buffers, x)
        # return vmap(self._functional_call, in_dims=(0, 0, 0))(self.params, self.buffers, x)


    # def _train_step(self, params, buffers, x, y):
    #     logits = self._forward_fn(params, buffers, x)
    #     per_node_loss = vmap(F.cross_entropy)(logits, y)
    #     self.optimizer.zero_grad(set_to_none=True)
    #     per_node_loss.sum().backward()
    #     self.optimizer.step()
    #     return per_node_loss.detach()
    
    def forward_shared(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, *input). Returns (N, B, num_classes). All N nodes see the same x.
        Use for evaluation where the test set is shared, not per-node."""
        return self.compiled_forward_shared(self.params, self.buffers, x)
        # return vmap(self._functional_call, in_dims=(0, 0, None))(self.params, self.buffers, x)

    def load_params(self,new_params: dict[str, torch.Tensor]):

        with torch.no_grad():
            for k,v in new_params.items():
                self.params[k].copy_(v)
                
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """One train step: vmap forward, per-node CE loss,  sum, backward, AdamW step.
        Returns per-node loss (N,)."""
        logits = self.forward(x)  # (N, B, ?)
        per_node_loss = vmap(F.cross_entropy)(logits, y)  # (N,)
        self.optimizer.zero_grad(set_to_none=True)
        per_node_loss.sum().backward()
        self.optimizer.step()
        return per_node_loss.detach()

