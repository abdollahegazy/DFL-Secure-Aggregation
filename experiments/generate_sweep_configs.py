import argparse
import copy
import csv
import re
from pathlib import Path

import yaml


DATASETS = ["mnist", "cifar10"]
TOPOLOGIES = ["small-world", "scale-free"]
AGGREGATIONS = ["fedavg", "krum", "geomed", "trimmedmean"]
MALICIOUS_PROPORTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
PLACEMENTS = ["random", "strategic"]
ATTACK_TYPES = ["noise"]
ITERATIONS = [0, 1, 2, 3, 4]

MANIFEST_COLUMNS = [
    "task_id",
    "config_path",
    "id",
    "iteration",
    "seed",
    "dataset",
    "model_name",
    "topology",
    "aggregation",
    "malicious_proportion",
    "placement",
    "attack_type",
    "status",
]

TOPOLOGY_MANIFEST_COLUMNS = [
    "topology_key",
    "topology_file",
    "seed",
    "iteration",
    "nodes",
    "topology",
    "small_world_k",
    "small_world_beta",
    "scale_free_m",
    "edge_density",
    "malicious_proportion",
    "placement",
    "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate YAML configs and a manifest CSV for a DFL experiment sweep."
    )
    parser.add_argument(
        "--base-config",
        required=True,
        type=Path,
        help="Base YAML config to copy before applying sweep overrides.",
    )
    parser.add_argument(
        "--sweep-id",
        required=True,
        help="Short name for this sweep, used in generated ids and topology paths.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where manifest.csv and configs/ will be written.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device string to write into each generated YAML, for example cpu or cuda:0.",
    )
    return parser.parse_args()


def load_base_config(path: Path) -> dict:
    with path.open() as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Base config must be a YAML mapping: {path}")
    return config


def slug(value: object) -> str:
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def malicious_tag(proportion: float) -> str:
    return f"b{int(round(proportion * 100)):03d}"


def combo_id(
    sweep_id: str,
    dataset: str,
    topology: str,
    aggregation: str,
    malicious_proportion: float,
    placement: str,
    attack_type: str,
) -> str:
    parts = [
        sweep_id,
        dataset,
        topology,
        aggregation,
        malicious_tag(malicious_proportion),
        placement,
        attack_type,
    ]
    return "__".join(slug(part) for part in parts)


def topology_key(
    sweep_id: str,
    topology: str,
    malicious_proportion: float,
    placement: str,
    iteration: int,
) -> str:
    parts = [
        sweep_id,
        "topology",
        topology,
        malicious_tag(malicious_proportion),
        placement,
        f"replica_{iteration}",
    ]
    return "__".join(slug(part) for part in parts)


def topology_file(
    sweep_id: str,
    topology: str,
    malicious_proportion: float,
    placement: str,
    iteration: int,
) -> str:
    key = topology_key(
        sweep_id=sweep_id,
        topology=topology,
        malicious_proportion=malicious_proportion,
        placement=placement,
        iteration=iteration,
    )
    return f"../data/topologies/{sweep_id}/{key}.json"


def status_for(placement: str) -> str:
    return "active"


def placement_options(malicious_proportion: float) -> list[str]:
    if malicious_proportion == 0.0:
        return ["random"]
    return PLACEMENTS


def build_config(
    base_config: dict,
    sweep_id: str,
    device: str,
    dataset: str,
    topology: str,
    aggregation: str,
    malicious_proportion: float,
    placement: str,
    attack_type: str,
    iteration: int,
) -> dict:
    seed = 42 + iteration
    run_id = combo_id(
        sweep_id=sweep_id,
        dataset=dataset,
        topology=topology,
        aggregation=aggregation,
        malicious_proportion=malicious_proportion,
        placement=placement,
        attack_type=attack_type,
    )

    config = copy.deepcopy(base_config)
    checkpoint_config = copy.deepcopy(config.get("checkpoint", {}) or {})
    checkpoint_config.setdefault("enabled", True)
    checkpoint_config.setdefault("keep_last_rounds", 2)
    config.update(
        {
            "id": run_id,
            "iteration": iteration,
            "seed": seed,
            "description": (
                f"{sweep_id}: {dataset}, {topology}, {aggregation}, "
                f"{malicious_tag(malicious_proportion)}, {placement}, {attack_type}"
            ),
            "use_saved_topology": True,
            "topology_file": topology_file(
                sweep_id=sweep_id,
                topology=topology,
                malicious_proportion=malicious_proportion,
                placement=placement,
                iteration=iteration,
            ),
            "dataset": dataset,
            "model_name": dataset,
            "topology": topology,
            "aggregation": aggregation,
            "malicious_proportion": malicious_proportion,
            "placement": placement,
            "attack_type": attack_type,
            "resume": config.get("resume", True),
            "checkpoint": checkpoint_config,
            "device": device,
            "num_workers": 8,
            "model_ckpt_dir": "../data/ckpts",
            "results_dir": "../data/results",
        }
    )
    return config


def iter_configs(base_config: dict, sweep_id: str, device: str):
    for dataset in DATASETS:
        for topology in TOPOLOGIES:
            for aggregation in AGGREGATIONS:
                for iteration in ITERATIONS:
                    for malicious_proportion in MALICIOUS_PROPORTIONS:
                        for placement in placement_options(malicious_proportion):
                            for attack_type in ATTACK_TYPES:
                                config = build_config(
                                    base_config=base_config,
                                    sweep_id=sweep_id,
                                    device=device,
                                    dataset=dataset,
                                    topology=topology,
                                    aggregation=aggregation,
                                    malicious_proportion=malicious_proportion,
                                    placement=placement,
                                    attack_type=attack_type,
                                    iteration=iteration,
                                )
                                yield config


def write_outputs(base_config: dict, sweep_id: str, output_dir: Path, device: str) -> tuple[int, int, int, int]:
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"
    topology_manifest_path = output_dir / "topologies.csv"
    active_count = 0
    pending_count = 0
    topology_rows = {}

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()

        for task_id, config in enumerate(iter_configs(base_config, sweep_id, device)):
            config_path = configs_dir / f"{task_id:04d}.yaml"
            with config_path.open("w") as config_file:
                yaml.safe_dump(config, config_file, sort_keys=False)

            status = status_for(config["placement"])
            if status == "active":
                active_count += 1
            else:
                pending_count += 1

            topo_key = topology_key(
                sweep_id=sweep_id,
                topology=config["topology"],
                malicious_proportion=config["malicious_proportion"],
                placement=config["placement"],
                iteration=config["iteration"],
            )
            topology_rows[topo_key] = {
                "topology_key": topo_key,
                "topology_file": config["topology_file"],
                "seed": config["seed"],
                "iteration": config["iteration"],
                "nodes": config["nodes"],
                "topology": config["topology"],
                "small_world_k": config["small_world_k"],
                "small_world_beta": config["small_world_beta"],
                "scale_free_m": config["scale_free_m"],
                "edge_density": config["edge_density"],
                "malicious_proportion": config["malicious_proportion"],
                "placement": config["placement"],
                "status": status,
            }

            writer.writerow(
                {
                    "task_id": task_id,
                    "config_path": config_path.as_posix(),
                    "id": config["id"],
                    "iteration": config["iteration"],
                    "seed": config["seed"],
                    "dataset": config["dataset"],
                    "model_name": config["model_name"],
                    "topology": config["topology"],
                    "aggregation": config["aggregation"],
                    "malicious_proportion": config["malicious_proportion"],
                    "placement": config["placement"],
                    "attack_type": config["attack_type"],
                    "status": status,
                }
            )

    with topology_manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TOPOLOGY_MANIFEST_COLUMNS)
        writer.writeheader()
        for topo_key in sorted(topology_rows):
            writer.writerow(topology_rows[topo_key])

    return active_count + pending_count, active_count, pending_count, len(topology_rows)


def main() -> None:
    args = parse_args()
    base_config = load_base_config(args.base_config)
    total_count, active_count, pending_count, topology_count = write_outputs(
        base_config=base_config,
        sweep_id=args.sweep_id,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(f"Wrote {total_count} configs to {args.output_dir / 'configs'}")
    print(f"Wrote manifest to {args.output_dir / 'manifest.csv'}")
    print(f"Wrote topology manifest to {args.output_dir / 'topologies.csv'}")
    print(f"Unique topology files: {topology_count}")
    print(f"Active configs: {active_count}")
    print(f"Pending configs: {pending_count}")


if __name__ == "__main__":
    main()
