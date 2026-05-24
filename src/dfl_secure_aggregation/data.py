from collections.abc import Iterator

import torch
from torch import Tensor


def sliding_window_partition(
    n_dataset_samples: int,
    n_nodes: int,
    samples_per_node: int,
) -> list[Tensor]:
    """Legacy partition: node i gets indices [i*S, (i+1)*S) mod M.

    Deterministic; each node holds `samples_per_node` consecutive indices, wrapping
    around the dataset if N * S > M. Matches the existing simulator's per-node
    assignment so legacy results stay comparable.
    """
    return [
        (torch.arange(samples_per_node) + i * samples_per_node) % n_dataset_samples
        for i in range(n_nodes)
    ]


class NodeDataLoader:
    """Yields per-node minibatches stacked as (N, B, *).

    Each call to __iter__ independently shuffles each node's index pool using the
    internal generator, then yields stacked (N, B, *) minibatches until the smallest
    node's pool is exhausted.

    Reproducibility: pass `seed` to fix the shuffle sequence. The generator advances
    across __iter__ calls (so successive epochs see different shuffles), but the full
    sequence is fully reproducible across runs with the same seed.
    """

    def __init__(
        self,
        x: Tensor,                       # (M, *input_shape), on target device
        y: Tensor,                       # (M,), on target device
        node_indices: list[Tensor],      # length N; each (samples_i,) of dataset indices
        batch_size: int,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        device = x.device
        self.x = x
        self.y = y
        self.node_indices = [idx.to(device) for idx in node_indices]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = torch.Generator(device=device)
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        if self.shuffle:
            pools = [
                idx[torch.randperm(idx.size(0), generator=self.generator, device=idx.device)]
                for idx in self.node_indices
            ]
        else:
            pools = self.node_indices

        n_batches = min(p.size(0) for p in pools) // self.batch_size
        B = self.batch_size
        for b in range(n_batches):
            batch_idx = torch.stack([p[b * B:(b + 1) * B] for p in pools])  # (N, B)
            yield self.x[batch_idx], self.y[batch_idx]                       # (N, B, *), (N, B)

    def __len__(self) -> int:
        return min(idx.size(0) for idx in self.node_indices) // self.batch_size
