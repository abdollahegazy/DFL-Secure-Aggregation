import torch
import os
class BaseModel:
    def __init__(self, num_samples: int, node_hash: int, epochs: int, batch_size: int, evaluating=False, device=None):
        self.num_samples = num_samples
        self.node_hash = node_hash
        self.epochs = epochs
        self.batch_size = batch_size
        self.evaluating = evaluating
        self.device = device if device else torch.device('cpu')
        self.data = None
        self.X_train = None
        self.X_valid = None
        self.y_train = None
        self.y_valid = None
        self.model = None

        # get number of gpus and assign device based on node hash
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available() and torch.cuda.device_count() > 0:
            self.device = torch.device(f'cuda:{self.node_hash % torch.cuda.device_count()}')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')


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
        
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.model.state_dict(), path)