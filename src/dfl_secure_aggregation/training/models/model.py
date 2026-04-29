from os import path

import torch
import os
from pathlib import Path

class BaseTrainer:
    def __init__(self, 
                 node_hash: int, 
                 device):
        self.node_hash = node_hash
        self.device = device
        self.model = None

    def train(self,
            dataset,
            epochs,
            batch_size,
            num_samples: int | None = None, 
              ):
        raise NotImplementedError
    
    def evaluate(self,
                 dataset):
        raise NotImplementedError


    def load_weights(self,path):
        if self.model is None:
            raise ValueError("Model must be initialized before loading weights")
        self.model.load_state_dict(torch.load(path, map_location=self.device,weights_only=True))

    def save_weights(self, path:str | Path):
        if self.model is None:
            raise ValueError("Model must be initialized before saving model")
        
        if type(path) == str:
            path = Path(path)
        assert isinstance(path, Path) # for my linter to shut up
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), path)