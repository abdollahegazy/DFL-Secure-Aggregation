from typing import OrderedDict
import torch

class InnerProductAttack:
    """
    Function to perform inner product attack on the received weights.
    
    """
    def __init__(self, attack_args: dict):
        self.defense = attack_args['defense']
        self.epsilon = attack_args['epsilon']
        
        print(f"[InnerProductAttack] Initialized for defense: {self.defense} with epsilon: {self.epsilon}")
    def get_poisoned_model(self, models: list, prev_global: OrderedDict):
        """
        Get the mean of the models.
        :param models: A list of tuples (model, num_samples) from benign clients.
        :return: The mean model.
        """
        # Create a zero gradients
        accum = {layer: torch.zeros_like(param) for layer, param in prev_global.items()}
        
        for model in models:
            for layer in accum:
                accum[layer] += (model[layer]-prev_global[layer])/len(models) # only add gradient vec
        for layer in accum:
            cop = torch.clone(accum[layer]).to('cpu').numpy()
            print("Inner product" + str(cop.T.dot((cop*-1/len(models)))))
            accum[layer] *= -1*self.epsilon
            accum[layer]+=prev_global[layer]
        return accum
    def attack(self, models: list):
        """
        Perform inner product attack on the received weights.
        :param models: A list of models where models[:-1] are benign and the last one is the previous global model
        :return: A dictionary containing the attacked weights.
        """
        models, prev_global = models[:-1], models[-1]
        print(f"[InnerProductAttack] Performing inner product attack num={len(models)}")
        # Get the mean of the models
        if self.defense=='fedavg':
            attack_model = self.get_poisoned_model(models, prev_global)
            return attack_model
        
        else:
            raise NotImplementedError(f"Defense {self.defense} not implemented")