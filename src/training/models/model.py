import torch
import os
from training.device import resolve_device

class BaseModel:
    def __init__(self, num_samples: int, node_hash: int, epochs: int, batch_size: int, evaluating=False, device=None):
        self.num_samples = num_samples
        self.node_hash = node_hash
        self.epochs = epochs
        self.batch_size = batch_size
        self.evaluating = evaluating
        self.device = resolve_device(device)
        self.data = None
        self.X_train = None
        self.X_valid = None
        self.y_train = None
        self.y_valid = None
        self.model = None


    def train(self,dataset):
        raise NotImplementedError
    
    def evaluate(self,dataset):
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
