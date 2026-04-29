from pathlib import Path
import argparse
from torchvision import transforms
import yaml
import shutil 
from dfl_secure_aggregation.network import graph
from dfl_secure_aggregation .simulator import DFLTrainer
from dfl_secure_aggregation.evaluation import save_results, make_plot

import torch
import numpy as np
rng = np.random.default_rng(seed=42)

trainer = None
active_params = None


def load_experiment_params(config_path):
    """
    Load experiment parameters from the explicit YAML config path.
    """
    with open(config_path) as f:
        params = yaml.safe_load(f)
    if params is None:
        raise ValueError(f'Config file is empty: {config_path}')
    return params

def save_params_snapshot(
        params,
        root_save_dir: Path):
    """
    Save the exact parameters used for this run beside the node metrics.
    """
    save_dir = root_save_dir / f"experiment_{params['id']}" / f"{params['iteration']}"
    save_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = save_dir / 'params.yaml'
    with open(snapshot_path, 'w') as f:
        yaml.safe_dump(params, f, sort_keys=False)
    print("Saved experiment params to", snapshot_path)


def run_simulation(params):
    """
    Runs the simulation with the experiment arguments.

    - Creates a network graph, adds nodes to the network, and makes connections between them.
    - Starts the nodes.
    - Starts the server.

    Args:
        params (dict): A dictionary of parameters for the simulation.

    """
    # global topology, trainer, active_params
    # active_params = params

    # get args
    num_nodes = params['nodes']
    malicious_proportion = params['malicious_proportion']
    exp_id = params['id']

    topology = graph.Topology()
    topology_file = params['topology_file']
    
    if not params['use_saved_topology']:
        print(f'Creating topology with {num_nodes} nodes')
        
        malicous_nodes = rng.choice(num_nodes, int(malicious_proportion*num_nodes), replace=False).tolist()
        print("Malicious nodes: ", malicous_nodes)

        if params['topology']=='random':
            edge_density = params['edge_density']
            topology.create_random_graph(num_nodes, edge_density, malicous_nodes)
        elif params['topology']=='small-world':
            k = params['small_world_k']
            p = params['small_world_beta']
            topology.create_small_world_graph(num_nodes, k, p, malicous_nodes)
        elif params['topology']=='scale-free':
            m = params['scale_free_m']
            topology.create_scale_free_graph(num_nodes, m,malicous_nodes)
        else:
            raise ValueError('Invalid topology: must be random, small-world, or scale-free')
        # save topology
        topology_dir = Path(topology_file).parent   
        topology_dir.mkdir(parents=True, exist_ok=True)
        topology.save(topology_file)
    else:
        topology.load(topology_file)
        print('Using saved topology')

    dataset_kwargs = params.get('dataset_kwargs', {})
    dataset_kwargs['transform'] = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dfl_trainer = DFLTrainer(num_nodes=num_nodes, 
                            topology=topology, 
                            num_workers=params['num_workers'], 
                            num_rounds=params['rounds'],
                            epochs_per_round=params['epochs_per_round'], 
                            batch_size=params['batch_size'], 
                            num_samples=params['num_samples'],
                            aggregation_method=params['aggregation'], 
                            attack_method=params['attack_type'], 
                            model=params.get('model_name', params['dataset']),
                            exp_id=params['id'],
                            exp_iteration=params['iteration'],
                            dataset_name=params['dataset'],
                            dataset_kwargs=dataset_kwargs,
                            params=params,
                            device=params.get('device', 'cpu'))


    dfl_trainer.load_data()
    dfl_trainer.run()

    save_results(params)
    make_plot(exp_id)

def signal_handler(sig, frame):
    global trainer, active_params
    if trainer is not None:
        for p in getattr(trainer, "processes", []):
            p.kill()
        if getattr(trainer, "device", None) is not None and trainer.device.type == "cuda":
            torch.cuda.empty_cache()
    exit(0)

def args_parser():
    parser = argparse.ArgumentParser(description='Run a DFL simulation from an explicit YAML config.')
    parser.add_argument('--config', required=True, help='Path to the experiment YAML config.')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = args_parser()
    experiment_params = load_experiment_params(args.config)

    print("Starting simulation with the following parameters:\n")
    print("Config:", args.config)
    print(experiment_params)

    # delete old expirement file
    # delete_files(experiment_params['id'], experiment_params['iteration'], node_metrics=True)
    save_params_snapshot(experiment_params, root_save_dir=Path('../data/results'))

    run_simulation(experiment_params)
