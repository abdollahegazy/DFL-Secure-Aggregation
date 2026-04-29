


def delete_files(exp_id, iteration, node_metrics=False):
    """
    Delete files in the models and core* files

    Args:
        exp_id (): The experiment id.
        iteration (): The experiment iteration.
        node_metrics (): Whether to delete node metrics json files.
    """
    models_dir = os.path.join('src','training','models', f'experiment_{exp_id}', f'{iteration}','nodes')
    if os.path.exists(models_dir):
        for round_dir in os.listdir(models_dir):
            for file in os.listdir(os.path.join(models_dir, round_dir)):
                os.remove(os.path.join(models_dir, round_dir, file))

    # remove json
    if node_metrics:
        node_metrics_dir = os.path.join('src','training','results',f'experiment_{exp_id}',f'{iteration}','node_metrics')
        if os.path.exists(node_metrics_dir):
            for file in os.listdir(node_metrics_dir):
                os.remove(os.path.join(node_metrics_dir, file))

    # remove core files
    for file in os.listdir('.'):
        if file.startswith('core'):
            os.remove(file)

def load_experiment_params(config_path):
    """
    Load experiment parameters from the explicit YAML config path.
    """
    with open(config_path) as f:
        params = yaml.safe_load(f)
    if params is None:
        raise ValueError(f'Config file is empty: {config_path}')
    return params

def save_params_snapshot(params):
    """
    Save the exact parameters used for this run beside the node metrics.
    """
    save_dir = os.path.join('src','training','results', f"experiment_{params['id']}", f"{params['iteration']}")
    os.makedirs(save_dir, exist_ok=True)
    snapshot_path = os.path.join(save_dir, 'params.yaml')
    with open(snapshot_path, 'w') as f:
        yaml.safe_dump(params, f, sort_keys=False)
    print("Saved experiment params to", snapshot_path)

trainer = None
active_params = None
def run_simulation(params):
    """
    Runs the simulation with the experiment arguments.

    - Creates a network graph, adds nodes to the network, and makes connections between them.
    - Starts the nodes.
    - Starts the server.

    Args:
        params (dict): A dictionary of parameters for the simulation.

    """
    global topology, trainer, active_params
    active_params = params

    # get args
    num_nodes = params['nodes']
    malicious_proportion = params['malicious_proportion']
    exp_id = params['id']

    topology = graph.Topology()
    topology_file = params['topology_file']
    
    if not params['use_saved_topology']:
        print(f'Creating topology with {num_nodes} nodes')
        malicous_nodes = random.sample(range(num_nodes), int(malicious_proportion*num_nodes))
        print("Malicious nodes: ", malicous_nodes)

        #### add nodes to network
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
        # this is not defined in the topology class 
        # elif params['topology'] == 'two-f-1':
            # topology.create_2f1_disjoint_graph(num_nodes, malicous_nodes)
        else:
            raise ValueError('Invalid topology: must be random, small-world, or scale-free')
        # save topology
        topology_dir = os.path.dirname(topology_file)
        if topology_dir:
            os.makedirs(topology_dir, exist_ok=True)
        topology.save(topology_file)
    else:
        topology.load(topology_file)
        print('Using saved topology')

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
                             dataset=params['dataset'],
                             params=params,
                             device=params.get('device', 'cpu'))


    trainer = dfl_trainer
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
    if active_params is not None:
        delete_files(active_params['id'], active_params['iteration'])
    exit(0)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run a DFL simulation from an explicit YAML config.')
    parser.add_argument('--config', required=True, help='Path to the experiment YAML config.')
    args = parser.parse_args()

    experiment_params = load_experiment_params(args.config)
    print("Starting simulation with the following parameters:\n")
    print("Config:", args.config)
    print(experiment_params)
    print()

    # delete old expirement files
    delete_files(experiment_params['id'], experiment_params['iteration'], node_metrics=True)
    save_params_snapshot(experiment_params)

    run_simulation(experiment_params)
