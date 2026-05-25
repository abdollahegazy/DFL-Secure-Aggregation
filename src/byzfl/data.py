from collections.abc import Callable, Iterator

import torch
from torch import Tensor


from typing import Tuple

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
    def __init__(
        self,
        x: Tensor,
        y: Tensor,
        node_indices: list[Tensor],
        batch_size: int,
        shuffle: bool = True,
        seed: int | None = None,
        augment_fn: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]] | None = None,
    ):
        self.device = x.device
        self.x = x
        self.y = y

        # NEW: Stack indices into a single 2D tensor (N, Samples_per_node)
        self.node_indices = torch.stack(node_indices).to(self.device)
        self.N, self.S = self.node_indices.shape

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment_fn = augment_fn
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        if self.shuffle:
            # NEW: 100x faster vectorized shuffle across all N nodes at once
            noise = torch.rand(
                (self.N, self.S), generator=self.generator, device=self.device
            )
            shuffle_idx = noise.argsort(dim=1)
            pools = torch.gather(self.node_indices, 1, shuffle_idx)
        else:
            pools = self.node_indices

        n_batches = self.S // self.batch_size
        B = self.batch_size

        for b in range(n_batches):
            # NEW: Instant 2D slice, no Python list comprehensions or torch.stack
            batch_idx = pools[:, b * B : (b + 1) * B]

            x_batch = self.x[batch_idx]
            y_batch = self.y[batch_idx]

            if self.augment_fn is not None:
                x_batch, y_batch = self.augment_fn(x_batch, y_batch)

            yield x_batch, y_batch

    def __len__(self) -> int:
        return self.S // self.batch_size