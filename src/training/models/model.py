import torch
import os


class BaseModel:
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
    
    def load_state_dict(self, state_dict):
        
        if self.model is None:
            raise ValueError("Model must be initialized before loading state dict")
    
        self.model.load_state_dict(state_dict)

    def load_model(self, path):

        if self.model is None:
            raise ValueError("Model must be initialized before loading model")
        
        self.model.load_state_dict(torch.load(path, weights_only=True))

    def save_model(self, path):

        if self.model is None:
            raise ValueError("Model must be initialized before saving model")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
