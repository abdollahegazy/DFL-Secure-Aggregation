'''
This file is the main entry point for the simulator. It sets up the server and runs the nodes.
'''
import multiprocessing
from multiprocessing import Manager
import torch
from torch.utils.data import Subset
from pathlib import Path
import shutil

from dfl_secure_aggregation.evaluation import save_node_metrics,save_results,make_plot
from dfl_secure_aggregation.aggregation import strategies
from dfl_secure_aggregation.network import graph
from dfl_secure_aggregation.training.datasets.registry import load_dataset_by_name
from dfl_secure_aggregation.training.trainers.model_loader import load_model_by_name
from dfl_secure_aggregation.attack import attacks

def _train_worker_wrapper(args):
    trainer, worker_id = args
    trainer.train_worker(worker_id)

def load_model_and_weights(
        model_name, 
        model_path: None | Path, 
        node_hash, 
        device):
    model_architecture = load_model_by_name(model_name)
    model = model_architecture(node_hash=node_hash, device=device)
    if model_path is not None:
        model.load_weights(model_path)
    model.model.to(device)
    return model


def resolve_device(device=None):
    """
    Resolve a configured device string to a torch.device.

    Device selection belongs at the simulation/config boundary. Lower-level
    model, aggregation, and attack code should receive an explicit device.
    """
    if device is None or device == "":
        return torch.device("cpu")

    if isinstance(device, torch.device):
        return device

    device = str(device).lower()
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(device)


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
                 dataset_name,
                 dataset_kwargs,
                 params,
                 results_dir,
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
            results_dir (Path): The directory to save simulation results.
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
        self.train_dataset_name = dataset_name
        self.train_dataset_kwargs = dataset_kwargs
        self.train_dataset = None
        self.model = model
        self.device = resolve_device(device)
        self.model_ckpt_dir = Path(params['model_ckpt_dir']) / f"{exp_id}" / f"replica-{exp_iteration}" / "nodes"
        self.results_dir = results_dir
        self.current_round=0

        manager = Manager()
        metrics_dict = {round_num: {'accuracy': 0, 'loss': 0} for round_num in range(self.num_rounds)}
        self.metrics_dict = manager.dict(metrics_dict)
        
    def load_data(self):
        """Load the data for the simulation."""
        print('Loading data')
        bundle = load_dataset_by_name(self.train_dataset_name, **self.train_dataset_kwargs)
        self.train_dataset = bundle.train
        self.test_dataset = bundle.test
        if bundle.val is not None:
            self.val_dataset = bundle.val
        else:
            self.val_dataset = None
        
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
            round_path = self.model_ckpt_dir / f'round_{round_num}'
            round_path.mkdir(parents=True, exist_ok=True)

            self.train_network()
            self.aggregate_network()

            # delete files from current round
            # TODO: add some checkpoint flagg to the yaml to avoid this
            if self.current_round>0:
                prev_dir = self.model_ckpt_dir / f'round_{self.current_round-1}'
                shutil.rmtree(prev_dir)
            self.current_round+=1


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

        model_architecture = load_model_by_name(
            self.model)
        
        model = model_architecture(
            node_hash=node_hash, 
            device=self.device)
        
        weights_path = self.model_ckpt_dir / f'round_{self.current_round}' / f'node_{node_hash}.pt'        
        
        if self.current_round>0:
            model.load_weights(weights_path)
        
        if self.train_dataset is None: raise ValueError('Dataset not loaded')

        # TODO: reviewers might not want IID data in all nodes.
        start_index = (node_hash*self.num_samples)% len(self.train_dataset)
        end_index = start_index + self.num_samples

        if end_index < len(self.train_dataset):
            subset_dataset = Subset(self.train_dataset, list(range(start_index, end_index)))
        else:
            subset_dataset = Subset(self.train_dataset, list(range(start_index, len(self.train_dataset))))
            subset_dataset += Subset(self.train_dataset, list(range(0, end_index%len(self.train_dataset))))

        model.train(
            dataset=subset_dataset,
            epochs=self.epochs_per_round,
            batch_size=self.batch_size,
            num_samples=self.num_samples
            )
    
        # save model in current round dir
        model.save_weights(weights_path)
        print("Saved model for node ", node_hash, "to", weights_path)

    def aggregate_network(self):

        # save model in round+1 dir
        print('\nAggregating models')
        # round_dir = self.model_ckpt_dir / f'round_{self.current_round}'
        # print(f'Files in {round_dir}: {os.listdir(round_dir)}')
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

        # benign aggregates from everyone 
        if not is_malicious:
            model_paths = [self.model_ckpt_dir / f'round_{self.current_round}' / f'node_{n}.pt'
                        for n in neighbors]

            model_paths.append(self.model_ckpt_dir / f'round_{self.current_round}' / f'node_{node_id}.pt')
        # but malicious only aggregates from benign to avoid poisoning themselves with other malicious nodes
        else:
            model_paths = [self.model_ckpt_dir / f'round_{self.current_round}' / f'node_{n}.pt'
                        for n in neighbors if not self.topology.nodes[n]['malicious']]

        # either benign with neighbor besides itself, or malicious with at least one neighbor
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
            return torch.load(self.model_ckpt_dir / f'round_{self.current_round}' / f'node_{node_id}.pt', weights_only=True)


    def _save_model_for_next_round(self, node_id, state_dict):
        save_dir = self.model_ckpt_dir / f'round_{self.current_round + 1}'
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, save_dir / f'node_{node_id}.pt')

    def _execute_attack(self, node_id):
        neighbors = self.topology.get_neighbors(node_id)
        attack_type = self.attack_method.lower()
        attack_args = dict(self.attack_args)
        attack_args['defense'] = self.aggregation_method.lower()
        attack_args['device'] = self.device
        # attack_args['nodes'] = len(neighbors) 

        attacker = attacks.create_attacker(attack_type, attack_args, node_id)

        model_path = Path(self.model_ckpt_dir) / f'round_{self.current_round}' / f'node_{node_id}.pt'

        
        model = load_model_and_weights(
            model_name=self.model,
            model_path=model_path,
            node_hash=node_id,
            device=self.device
        )

        if attack_type == 'alie':
            benign_paths = [self.model_ckpt_dir / f'round_{self.current_round}' / f'node_{n}.pt'
                            for n in neighbors if not self.topology.nodes[n]['malicious']]
            poisoned_model = attacker.attack(benign_paths)
        else:

            poisoned_model =  attacker.attack(model.model.state_dict())

        model.model.load_state_dict(poisoned_model)
        model.save_weights(model_path)

    def _evaluate_and_log(self,
     node_id, 
     metrics_dict):
        model_path = self.model_ckpt_dir / f'round_{self.current_round + 1}' / f'node_{node_id}.pt'
        model = load_model_and_weights(
            model_name=self.model,
            model_path=model_path,
            node_hash=node_id,
            device=self.device
        )  
        # is_malicious = node_id in self.malicious_nodes

        accuracy, loss = model.evaluate(self.test_dataset)
        metrics_dict[self.current_round]['accuracy'] += accuracy
        metrics_dict[self.current_round]['loss'] += loss
        print(f'Node {node_id} round {self.current_round} accuracy: {accuracy} loss: {loss}')
        save_node_metrics(node_id, accuracy, loss, self.exp_id, self.exp_iteration, self.results_dir)

    # def __del__(self):
        # if getattr(self, "device", None) is not None and self.device.type == "cuda":
            # torch.cuda.empty_cache()
        # delete_files(self.exp_id, self.exp_iteration)




def delete_files(exp_id, 
                 iteration, 
                 ckpt_base,
                 results_base,
                 remove_results=False,
                 remove_all=False,
            ):
    """
    Delete files in the models and core* files

    Args:
        exp_id (): The experiment id.
        iteration (): The experiment iteration.
        node_metrics (): Whether to delete node metrics json files.
        remove_results (): Whether to delete result files.
    """
    if remove_all:
        exp_dir_ckpt = Path(ckpt_base) / f"{exp_id}"
        exp_dir_results = Path(results_base) / f"{exp_id}"
        shutil.rmtree(exp_dir_ckpt, ignore_errors=True)
        shutil.rmtree(exp_dir_results, ignore_errors=True)
        return
    
    ckpt_dir = Path(ckpt_base) / f"{exp_id}" / f"replica-{iteration}" 
    shutil.rmtree(ckpt_dir,ignore_errors=True)
    # remove the dir itself if empty
    if ckpt_dir.exists() and not any(ckpt_dir.iterdir()):
        ckpt_dir.rmdir()

    results_dir = Path(results_base) / f"{exp_id}" / f"replica-{iteration}"
    # remove json
    shutil.rmtree(results_dir, ignore_errors=True)
    # remove the dir itself if empty
    if results_dir.exists() and not any(results_dir.iterdir()):
        results_dir.rmdir()