'''
This file is the main entry point for the simulator. It sets up the server and runs the nodes.
'''
import argparse
from network import graph
import multiprocessing
from multiprocessing import Manager
import os
import random
import yaml
import evaluation
import torch
from torch.utils.data import Subset
from training.models.model_loader import load_model_by_name
from training.dataloader import load_data_by_name
from training.device import resolve_device
from aggregation import strategies
from attack import attacks
# seed
random.seed(42)


def _train_worker_wrapper(args):
    trainer, worker_id = args
    trainer.train_worker(worker_id)




class DFLTrainer:
    def __init__(self, 
                 num_nodes, 
                 topology, 
                 num_workers, 
                 num_rounds, 
                 epochs_per_round, 
                 batch_size,
                 num_samples, 
                 aggregation_method, 
                 attack_method, 
                 model,
                 exp_id,
                 exp_iteration,
                 dataset,
                 params,
                 device='cpu'):
        """
        Args:
            num_nodes (int): The number of nodes in the network.
            topology (graph.Topology): The topology of the network.
            num_workers (int): The number of workers to use for training and aggregation.
            num_rounds (int): The number of rounds to run the simulation for.
            epochs_per_round (int): The number of epochs to train for each round.
            batch_size (int): The batch size to use for training.
            num_samples (int): The number of samples to use for training each node.
            aggregation_method (str): The method to use for aggregation.
            attack_method (str): The method to use for attacks.
            model (str): The model architecture to use for training.
            exp_id (str): The experiment id.
            exp_iteration (str): The experiment iteration.
            dataset (str): The dataset to use for training and evaluation.
            params (dict): The full experiment config loaded from YAML.
            device (str): The torch device to use for this simulation.
        """
        self.num_nodes = num_nodes
        self.topology = topology
        self.num_workers = num_workers
        self.num_rounds = num_rounds
        self.epochs_per_round = epochs_per_round
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.aggregation_method = aggregation_method
        self.attack_method = attack_method
        self.params = params
        self.trimmed_mean_beta = params['trimmed_mean_beta']
        self.attack_args = dict(params['attack_args'])

        self.node_idx = list(range(self.num_nodes))
        self.malicious_nodes = set(node_hash for node_hash in self.node_idx if \
                                 self.topology.nodes[node_hash]['malicious'])
        
        self.exp_id = exp_id
        self.exp_iteration = exp_iteration
        self.dataset_name = dataset
        self.dataset = None
        self.model = model
        self.device = resolve_device(device)

        self.models_base_dir = os.path.join('src','training','models', 
                                            f'experiment_{self.exp_id}', f'{self.exp_iteration}','nodes')
        self.current_round=0

        manager = Manager()
        metrics_dict = {round_num: {'accuracy': 0, 'loss': 0} for round_num in range(self.num_rounds)}
        self.metrics_dict = manager.dict(metrics_dict)
        
    def load_data(self):
        """Load the data for the simulation."""
        print('Loading data')
        self.dataset = load_data_by_name(self.dataset_name)
        
    def run(self):
        """
        Run the simulation.
        """
        print('Starting simulation')
        print(f'Number of nodes: {self.num_nodes}')
        print(f'Number of workers: {self.num_workers}')
        print(f'Number of rounds: {self.num_rounds}')
        print(f'Epochs per round: {self.epochs_per_round}')
        print(f'Batch size: {self.batch_size}')
        print(f'Number of samples: {self.num_samples}')
        print(f'Aggregation method: {self.aggregation_method}')
        print(f'Attack method: {self.attack_method}')
        print(f'Device: {self.device}')

        for round_num in range(self.num_rounds):
            print(f'\n\tStarting round {round_num}')
            # train models
            if not os.path.exists(os.path.join(self.models_base_dir, f'round_{self.current_round}')):
                os.makedirs(os.path.join(self.models_base_dir, f'round_{self.current_round}'))
            self.train_network()
            self.aggregate_network()

            # delete files from current round
            # TODO: add some checkpoint flagg to the yaml to avoid this
            if self.current_round>0:
                prev_dir = os.path.join(self.models_base_dir, f'round_{self.current_round-1}')
                for file in os.listdir(prev_dir):
                    os.remove(os.path.join(prev_dir, file))

            self.current_round+=1
    def run_tasks(self, processes):
        for p in processes:
            p.start()

        self.processes = processes

        for p in processes:
            p.join()

        # kill all processes just in case
        for p in processes:
            p.terminate()
        self.processes = []


    def train_network(self):
        """
        Train the models on each node in parallel.
        """
        print('Training models')
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            pool.map(_train_worker_wrapper, [(self, i) for i in range(self.num_nodes)])
            
    def train_worker(self, node_hash):
        """
        Train a model for a single node.

        Args:
            node_hash (int): The node hash. It's rlly a node idx but too lazy to change variable name.
        """
        print(f'Training model for node {node_hash} round {self.current_round}')
        # create model
        model_architecture = load_model_by_name(self.model)
        model = model_architecture(epochs=self.epochs_per_round, batch_size=self.batch_size, num_samples=self.num_samples,
                         node_hash=node_hash, evaluating=False, device=self.device)
        
        if self.current_round>0:
            model.load_model(os.path.join(self.models_base_dir, f'round_{self.current_round}', f'node_{node_hash}.pt'))
        
        if self.dataset is None: raise ValueError('Dataset not loaded')

        # TODO: reviewers might not want IID data in all nodes.
        start_index = (node_hash*self.num_samples)% len(self.dataset)
        end_index = start_index + self.num_samples

        if end_index < len(self.dataset):
            subset_dataset = Subset(self.dataset, list(range(start_index, end_index)))
        else:
            subset_dataset = Subset(self.dataset, list(range(start_index, len(self.dataset))))
            subset_dataset += Subset(self.dataset, list(range(0, end_index%len(self.dataset))))

        model.train(subset_dataset)
    
        # save model in current round dir
        filename = os.path.join(self.models_base_dir, f'round_{self.current_round}', f'node_{node_hash}.pt')
        model.save_model(filename)
        print("Saved model for node ", node_hash, "to", filename)

    def aggregate_network(self):

        # save model in round+1 dir
        print('\nAggregating models')
        round_dir = os.path.join(self.models_base_dir, f'round_{self.current_round}')
        print(f'Files in {round_dir}: {os.listdir(round_dir)}')
        # malicious nodes should aggregate first
        # then benign nodes aggregate
        malicious = [n for n in self.node_idx if n in self.malicious_nodes]
        benign = [n for n in self.node_idx if n not in self.malicious_nodes]

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            if malicious:
                pool.starmap(self.aggregate_worker, [(n, self.metrics_dict) for n in malicious])
            pool.starmap(self.aggregate_worker, [(n, self.metrics_dict) for n in benign])
        

    def aggregate_worker(self, node_id, metrics_dict):
        """
        Aggregate the models for a single node.

        Args:
            node_id (int): The node id.
            metrics_dict (dict): A dictionary to store the metrics for each round.
        """
        aggregated_model = self._aggregate_models(node_id)
        self._save_model_for_next_round(node_id, aggregated_model)
        
        if node_id in self.malicious_nodes:
            self._execute_attack(node_id)
        else:
            self._evaluate_and_log(node_id, metrics_dict)


    def _aggregate_models(self, node_id):
        neighbors = self.topology.get_neighbors(node_id)
        is_malicious = node_id in self.malicious_nodes

        if not is_malicious:
            model_paths = [os.path.join(self.models_base_dir, f'round_{self.current_round}', f'node_{n}.pt')
                        for n in neighbors]
            model_paths.append(os.path.join(self.models_base_dir, f'round_{self.current_round}', f'node_{node_id}.pt'))
        else:
            model_paths = [os.path.join(self.models_base_dir, f'round_{self.current_round}', f'node_{n}.pt')
                        for n in neighbors if not self.topology.nodes[n]['malicious']]

        should_aggregate = (not is_malicious and len(model_paths) > 1) or (is_malicious and len(model_paths) > 0)

        if should_aggregate:
            agg_args = {
                'f': len(model_paths),
                'm': len([n for n in neighbors if self.topology.nodes[n]['malicious']]),
                'trimmed_mean_beta': self.trimmed_mean_beta,
                'aggregation': self.aggregation_method,
                'device': self.device,
            }
            aggregator = strategies.create_aggregator(node_id, agg_args)
            return aggregator.aggregate(model_paths)
        else:
            return torch.load(
                os.path.join(self.models_base_dir, f'round_{self.current_round}', f'node_{node_id}.pt'),
                weights_only=True
            )

    def _save_model_for_next_round(self, node_id, state_dict):
        model = load_model_by_name(self.model)(
            epochs=self.epochs_per_round, batch_size=self.batch_size,
            num_samples=self.num_samples, node_hash=node_id, evaluating=True, device=self.device
        )
        model.model.load_state_dict(state_dict)
        save_dir = os.path.join(self.models_base_dir, f'round_{self.current_round + 1}')
        os.makedirs(save_dir, exist_ok=True)
        model.save_model(os.path.join(save_dir, f'node_{node_id}.pt'))

    def _execute_attack(self, node_id):
        neighbors = self.topology.get_neighbors(node_id)
        attack_type = self.attack_method.lower()
        attack_args = dict(self.attack_args)
        attack_args['defense'] = self.aggregation_method.lower()
        attack_args['device'] = self.device
        attack_args['nodes'] = len(neighbors)
        attack_args['malicious_nodes'] = len([n for n in neighbors if self.topology.nodes[n]['malicious']])

        attacker = attacks.create_attacker(attack_type, attack_args, node_id)

        model_path = os.path.join(self.models_base_dir, f'round_{self.current_round}', f'node_{node_id}.pt')
        model = load_model_by_name(self.model)(
            epochs=self.epochs_per_round, batch_size=self.batch_size,
            num_samples=self.num_samples, node_hash=node_id, evaluating=True, device=self.device
        )
        model.model.load_state_dict(torch.load(model_path, weights_only=True))

        if attack_type == 'alie':
            benign_paths = [os.path.join(self.models_base_dir, f'round_{self.current_round}', f'node_{n}.pt')
                            for n in neighbors if not self.topology.nodes[n]['malicious']]
            poisoned_model = attacker.attack(benign_paths)
        else:
            poisoned_model = attacker.attack(model.model.state_dict())

        model.model.load_state_dict(poisoned_model)
        model.save_model(model_path)

    def _evaluate_and_log(self, node_id, metrics_dict):
        model_path = os.path.join(self.models_base_dir, f'round_{self.current_round + 1}', f'node_{node_id}.pt')
        model = load_model_by_name(self.model)(
            epochs=self.epochs_per_round, batch_size=self.batch_size,
            num_samples=self.num_samples, node_hash=node_id, evaluating=True, device=self.device
        )
        model.model.load_state_dict(torch.load(model_path, weights_only=True))

        accuracy, loss = model.evaluate(self.dataset)
        metrics_dict[self.current_round]['accuracy'] += accuracy
        metrics_dict[self.current_round]['loss'] += loss
        print(f'Node {node_id} round {self.current_round} accuracy: {accuracy} loss: {loss}')
        evaluation.save_node_metrics(node_id, accuracy, loss, self.exp_id, self.exp_iteration)

    # def __del__(self):
        # if getattr(self, "device", None) is not None and self.device.type == "cuda":
            # torch.cuda.empty_cache()
        # delete_files(self.exp_id, self.exp_iteration)
        
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

    evaluation.save_results(params)
    evaluation.make_plot(exp_id)

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
