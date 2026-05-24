"""
Round loop for vectorized DFL.

Pure function: composes NodeBank + Topology + aggregator + attack + data loader.
Returns per-round metrics. Persistence (writing to disk) is the caller's job.
"""
from collections.abc import Callable, Iterable

import torch

from .nodebank import NodeBank
from .network import Topology


def run_simulation(
    bank: NodeBank,
    topology: Topology,
    train_minibatch_iter: Iterable[tuple[torch.Tensor, torch.Tensor]],  # yields (x, y) of shape (N, B, *) and (N, B)
    test_x: torch.Tensor,                                                # (B_test, *input_shape), shared across nodes
    test_y: torch.Tensor,                                                # (B_test,)
    aggregate_fn: Callable[[dict[str, torch.Tensor], Topology], dict[str, torch.Tensor]],
    num_rounds: int,
    steps_per_round: int,
    eval_every: int,
    log_metrics: bool = True,
    attack_fn: Callable[[NodeBank, Topology], None] | None = None,
    eval_batch_size: int = 512,
) -> list[dict]:
    
    """Run the DFL simulation.
    Each round:
      1. Pull `steps_per_round` minibatches from train_minibatch_iter, call bank.train_step on each.
         Iterator is cycled — if it runs out before steps_per_round, it restarts (re-shuffles
         if it's a NodeDataLoader).
      2. If attack_fn is set, call attack_fn(bank, topology) — mutates bank.params in place.
      3. Call aggregate_fn(bank.params, topology) and load_params the result.
      4. Every eval_every rounds (and on the final round), evaluate against test_minibatch_iter.

    Returns list of dicts, one per evaluated round, with keys:
      - round (int)
      - per_node_acc (Tensor, (N,))
      - mean_benign_acc (float)
      - train_loss_per_node (Tensor, (N,))
    """
    history: list[dict] = []
    train_iter = _cycle(train_minibatch_iter)

    for r in range(num_rounds):
        # 1. Train
        losses: list[torch.Tensor] = []
        for _ in range(steps_per_round):
            x, y = next(train_iter)
            losses.append(bank.train_step(x, y))
        train_loss = torch.stack(losses).mean(dim=0)  # (N,)

        # 2. Attack (in-place mutation of bank.params, malicious rows only)
        if attack_fn is not None:
            attack_fn(bank, topology)

        # 3. Aggregate
        bank.load_params(aggregate_fn(bank.params, topology))

        # 4. Evaluate
        if r % eval_every == 0 or r == num_rounds - 1:
            metrics = _evaluate(bank, test_x, test_y, topology.malicious_mask, eval_batch_size)
            metrics["round"] = r
            metrics["train_loss_per_node"] = train_loss.detach().cpu()
            history.append(metrics)
            if log_metrics:
                print(
                    f"[round {r:3d}] mean_benign_acc={metrics['mean_benign_acc']:.4f}  "
                    f"train_loss(mean)={train_loss.mean().item():.4f}"
                )

    return history


def _cycle(iterable: Iterable):
    """Indefinitely re-iterate over `iterable`. Each pass calls iter() afresh, so a
    NodeDataLoader will reshuffle each epoch."""
    while True:
        yielded = False
        for item in iterable:
            yielded = True
            yield item
        if not yielded:
            raise RuntimeError("train_minibatch_iter produced no batches")


def _evaluate(
    bank: NodeBank,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    malicious_mask: torch.Tensor | None = None,
    batch_size: int = 512,
) -> dict:
    """Per-node test accuracy + benign-mean. Chunks test_x along B for memory."""
    correct = torch.zeros(bank.n, device=bank.device)
    total = 0
    with torch.no_grad():
        for start in range(0, test_x.shape[0], batch_size):
            xc = test_x[start:start + batch_size]
            yc = test_y[start:start + batch_size]
            preds = bank.forward_shared(xc).argmax(dim=-1)    # (N, B), broadcasts x to all N nodes via vmap
            correct += (preds == yc.unsqueeze(0)).sum(dim=1).float()
            total += xc.shape[0]
    per_node_acc = correct / max(total, 1)
    if malicious_mask is not None:
        benign = per_node_acc[~malicious_mask]
        mean_benign = benign.mean().item() if benign.numel() > 0 else float("nan")
    else:
        mean_benign = per_node_acc.mean().item()
    return {
        "per_node_acc": per_node_acc.cpu(),
        "mean_benign_acc": mean_benign,
    }
