"""Mass sweep harness for byzfl: runs all 880 configs of reviewer_fixes_v1 in-process.

Topologies, datasets, and model architectures are all cached in memory; each
config builds a fresh NodeBank, runs the simulation, and writes one JSON
result file. Skips configs whose result already exists, so the loop is
resumable across crashes.

Run:
    python experiments/run_sweep.py --sweep-id v2 --device cuda:0
"""
import argparse
import json
import time
from functools import partial
from pathlib import Path
import traceback
import torch
from tqdm import tqdm
from byzfl import NodeBank, Topology, run_simulation, NodeDataLoader, sliding_window_partition
from byzfl import STRATEGIES, ATTACKS
from byzfl.network.generate import small_world_graph, scale_free_graph, random_graph

from datasets import DATASETS, AUGMENTS
from models import MODELS

# torch.autograd.set_detect_anomaly(True)


# ---------------------------------------------------------------------------
# Sweep dimensions (mirrors reviewer_fixes_v1: 880 configs)
# ---------------------------------------------------------------------------

DATASETS_SWEEP = ["mnist", "cifar10"]
TOPOLOGIES = ["small-world", "scale-free"]
AGGREGATIONS = ["fedavg", "krum", "geomed", "trimmedmean"]
MALICIOUS_PROPORTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
PLACEMENTS = ["random", "strategic"]
ATTACK_TYPES = ["add_noise","set_random"]
ITERATIONS = [0, 1, 2, 3, 4]

# ---------------------------------------------------------------------------
# Fixed knobs (match run_mnist.py defaults)
# ---------------------------------------------------------------------------

N = 128 # number of nodes;
S = 10000 # number of samples per node;
B = 256  # per-node batch size;

NUM_ROUNDS = 100
STEPS_PER_ROUND = S // B  # ≈ 1 epoch over the per-node pool

EVAL_EVERY = 10
EVAL_BATCH_SIZE = 1024

SMALL_WORLD_K = 4
SMALL_WORLD_BETA = 0.1
SCALE_FREE_M = 3
EDGE_DENSITY = 0.1

SGD_OPTIMIZER_KWARGS = {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4} # ,"fused": True}
ADAMW_OPTIMIZER_KWARGS = {"lr": 1e-3, "weight_decay": 1e-2}

AGG_KWARGS = {
    "fedavg": {},
    "krum": {"m_select": 5},          # multi-krum; f filled per-config from mal_count
    "geomed": {"maxiter": 16, "ftol": 1e-5},
    "trimmedmean": {"beta": 0.2},
    "median": {},
}

ATTACK_KWARGS = {
    "add_noise": {"strength": 1.0},
    "signflip": {},
    "set_random": {},
}



# ---------------------------------------------------------------------------
# Sweep iteration
# ---------------------------------------------------------------------------


def attack_options(malicious_proportion: float) -> list[str]:
    """At 0% malicious, attack_type is meaningless — single 'none' run."""
    if malicious_proportion == 0.0:
        return ["none"]
    return ATTACK_TYPES


def placement_options(malicious_proportion: float) -> list[str]:
    """At 0% malicious, placement is meaningless — single 'random' run."""
    if malicious_proportion == 0.0:
        return ["random"]
    return PLACEMENTS


def run_id_for(cfg: dict) -> str:
    return (
        f"{cfg['dataset']}__{cfg['topology'].replace('-', '_')}__{cfg['aggregation']}"
        f"__b{int(round(cfg['malicious_proportion'] * 100)):03d}__{cfg['placement']}"
        f"__{cfg['attack_type']}__iter{cfg['iteration']}"
    )


def iter_combos():
    for dataset in DATASETS_SWEEP:
        for topology in TOPOLOGIES:
            for aggregation in AGGREGATIONS:
                for iteration in ITERATIONS:
                    for mal in MALICIOUS_PROPORTIONS:
                        for placement in placement_options(mal):
                            for attack in attack_options(mal):
                                yield {
                                    "dataset": dataset,
                                    "topology": topology,
                                    "aggregation": aggregation,
                                    "malicious_proportion": mal,
                                    "placement": placement,
                                    "attack_type": attack,
                                    "iteration": iteration,
                                    "seed": 42 + iteration,
                                }


# ---------------------------------------------------------------------------
# Per-config builders
# ---------------------------------------------------------------------------


def make_topology(cfg: dict, device: torch.device) -> Topology:
    kind = cfg["topology"]
    common = dict(
        malicious_proportion=cfg["malicious_proportion"],
        placement=cfg["placement"],
        seed=cfg["seed"],
    )
    if kind == "small-world":
        td = small_world_graph(N, SMALL_WORLD_K, SMALL_WORLD_BETA, **common)
    elif kind == "scale-free":
        td = scale_free_graph(N, SCALE_FREE_M, **common)
    elif kind == "random":
        td = random_graph(N, EDGE_DENSITY, **common)
    else:
        raise ValueError(kind)
    return Topology.from_dict(td, device)


def make_bank(cfg: dict, device: torch.device) -> NodeBank:
    torch.manual_seed(cfg["seed"])
    model_cls = MODELS[cfg["dataset"]]
    models_ = [model_cls() for _ in range(N)]
    if cfg["dataset"] == "mnist":
        #using the sgd for mnist caused so many nans bruh
        opt_cls, opt_kwargs = torch.optim.AdamW, ADAMW_OPTIMIZER_KWARGS
    else:
        opt_cls, opt_kwargs = torch.optim.SGD, SGD_OPTIMIZER_KWARGS
    return NodeBank(
        models=models_,
        device=device,
        optimizer_cls=opt_cls,
        optimizer_kwargs=opt_kwargs,
    )


def make_aggregate_fn(cfg: dict):
    agg_fn = STRATEGIES[cfg["aggregation"]]
    kwargs = dict(AGG_KWARGS[cfg["aggregation"]])
    if cfg["aggregation"] == "krum":
        kwargs["f"] = int(N * cfg["malicious_proportion"])
    return partial(agg_fn, **kwargs)


def make_attack_fn(cfg: dict):
    if cfg["malicious_proportion"] == 0.0:
        return None
    atk_fn = ATTACKS[cfg["attack_type"]]
    kwargs = ATTACK_KWARGS.get(cfg["attack_type"], {})
    return partial(atk_fn, **kwargs)


def save_history(history: list[dict], cfg: dict, topo:Topology, path: Path) -> None:
    payload = {
        "config": cfg,
        "topology": {
            "edge_index": topo.edges.cpu().tolist(),
            "malicious_mask": topo.malicious_mask.cpu().tolist(),
        },
        "history": [
            {
                "round": int(h["round"]),
                "mean_benign_acc": float(h["mean_benign_acc"]),
                "per_node_acc": h["per_node_acc"].tolist(),
                "train_loss_per_node": h["train_loss_per_node"].tolist(),
            }
            for h in history
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-id", required=True, help="Output subdirectory name under --results-dir.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-dir", default="experiments/data/datasets")
    parser.add_argument("--results-dir", default="experiments/data/results")
    parser.add_argument("--dry-run", action="store_true", help="List configs without running.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N configs (after skipping existing).")
    parser.add_argument("--shard", type=int, default=0, help="Shard index (0-based) of this worker.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards splitting the sweep.")
    parser.add_argument("--log-metrics", action="store_true", help="Print per-round simulator metrics (mangles tqdm; use for debugging).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    results_root = Path(args.results_dir) / args.sweep_id
    results_root.mkdir(parents=True, exist_ok=True)

    combos = list(iter_combos())
    total = len(combos)
    my_combos = [(i, cfg) for i, cfg in enumerate(combos) if i % args.num_shards == args.shard]
    print(
        f"sweep_id={args.sweep_id}  device={device}  total_configs={total}  "
        f"shard={args.shard}/{args.num_shards}  shard_configs={len(my_combos)}"
    )
    if args.dry_run:
        for i, cfg in my_combos:
            print(f"[{i + 1}/{total}] {run_id_for(cfg)}")
        return

    # Load datasets once into GPU memory.
    needed = sorted({cfg["dataset"] for _, cfg in my_combos})
    dataset_cache: dict = {}

    print("Loading datasets...")
    for name in needed:
        print(f"Loading dataset: {name}")
        dataset_cache[name] = DATASETS[name](data_dir=args.data_dir, device=device)
        x_train = dataset_cache[name][0]
        print(f"  shape={tuple(x_train.shape)}")

    # Reuse the same per-node sample index across iterations (depends only on N, S, M).
    partition_cache: dict[tuple[int, int], list[torch.Tensor]] = {}

    ran = 0
    skipped = 0
    failed = 0
    t_sweep = time.time()
    pbar = tqdm(my_combos, desc=f"shard {args.shard}/{args.num_shards}", unit="cfg", dynamic_ncols=True)
    for i, cfg in pbar:
        run_id = run_id_for(cfg)
        out_path = results_root / f"{run_id}.json"
        if out_path.exists():
            skipped += 1
            pbar.set_postfix(ran=ran, skip=skipped, fail=failed)
            continue
        if args.limit is not None and ran >= args.limit:
            break

        x_train, y_train, x_test, y_test = dataset_cache[cfg["dataset"]]
        M = x_train.shape[0]
        if (M, cfg["seed"]) not in partition_cache:
            partition_cache[(M, cfg["seed"])] = sliding_window_partition(M, N, S)
        idx = partition_cache[(M, cfg["seed"])]

        topology = make_topology(cfg, device)
        train_loader = NodeDataLoader(
            x_train, y_train, idx,
            batch_size=B, seed=cfg["seed"],
            augment_fn=AUGMENTS.get(cfg["dataset"]),
        )
        bank = make_bank(cfg, device)
        agg = make_aggregate_fn(cfg)
        attack = make_attack_fn(cfg)

        pbar.set_description(f"shard {args.shard}/{args.num_shards} | {run_id}")
        t0 = time.time()
        try:
                    history = run_simulation(
                        bank=bank,
                        topology=topology,
                        train_minibatch_iter=train_loader,
                        test_x=x_test,
                        test_y=y_test,
                        aggregate_fn=agg,
                        attack_fn=attack,
                        num_rounds=NUM_ROUNDS,
                        steps_per_round=STEPS_PER_ROUND,
                        eval_every=EVAL_EVERY,
                        eval_batch_size=EVAL_BATCH_SIZE,
                        log_metrics=args.log_metrics,
                    )
        except Exception as e:
            failed += 1
            tqdm.write(f"[{i + 1}/{total}] {run_id}  FAIL: {type(e).__name__}: {e}")
            traceback.print_exc()
            pbar.set_postfix(ran=ran, skip=skipped, fail=failed)
            continue
        dt = time.time() - t0

        save_history(history, cfg, topology, out_path)
        ran += 1
        last_acc = history[-1]["mean_benign_acc"]
        tqdm.write(f"[{i + 1}/{total}] {run_id}  {dt:5.1f}s  acc={last_acc:.4f}")
        pbar.set_postfix(ran=ran, skip=skipped, fail=failed)

    pbar.close()
    print(f"\ndone. ran={ran}  skipped={skipped}  failed={failed}  total_time={(time.time() - t_sweep) / 60:.1f}m")


if __name__ == "__main__":
    main()
