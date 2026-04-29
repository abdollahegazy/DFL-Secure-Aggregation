import torch

class ALittleIsEnough:
    def __init__(self, attack_args: dict):
        '''
        https://github.com/lishenghui/blades/blob/master/blades/adversaries/alie_adversary.py
        '''

        print(f"[ALIE] Initialized with args {attack_args}")
        self.device = attack_args.get('device', torch.device('cpu'))
        num_nodes = attack_args['nodes']
        num_byz = attack_args['malicious_nodes']

        s = num_nodes // 2 + 1 - num_byz
        cdf_value = torch.tensor((num_nodes - num_byz - s) / (num_nodes - num_byz), device=self.device)
        dist = torch.distributions.normal.Normal(torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))
        self.z_max = dist.icdf(cdf_value)
    def attack(self, model_paths: list):
        '''
        ALIE attack on the model paths
        
        Args:
            model_paths: list of model paths
            
        Returns:
            poisoned_model: poisoned model
        '''
        # get mean of model and std of models
        poisoned_model = None
        std_model = None
        for model_path in model_paths:
            model = torch.load(model_path, map_location=self.device)
            if poisoned_model is None:
                poisoned_model = {name: torch.zeros_like(param).to(self.device) for name, param in model.items()}
                std_model = {name: torch.zeros_like(param).to(self.device) for name, param in model.items()}
            
            assert std_model is not None, "This is here for my linter to shut up"
            
            for layer in model:
                poisoned_model[layer] += model[layer].to(self.device)
                std_model[layer] += model[layer].to(self.device)**2

        assert poisoned_model is not None and std_model is not None, "This is here for my linter to shut up"
        
        for layer in poisoned_model:
            poisoned_model[layer] /= len(model_paths)
            std_model[layer] = torch.sqrt(std_model[layer]/len(model_paths) - poisoned_model[layer]**2)

            poisoned_model[layer] += std_model[layer] * self.z_max
        print("[ALIE] Poisoned model")
        return poisoned_model
      