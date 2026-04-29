from typing import OrderedDict
import torch
import copy

class RandomNoise():
    """
    Function to perform noise injection attack on the received weights.
    """
    def __init__(self, attack_args: dict):
        self.strength = attack_args['strength']
        self.device = attack_args.get('device', torch.device('cpu'))
        print(f"[NoiseInjectionAttack] Initialized with strength: {self.strength}")

    def attack(self, model: OrderedDict):
        print("[NoiseInjectionAttack] Performing noise injection attack")
        random_model = copy.deepcopy(model)
        lkeys = list(random_model.keys())
        for k in lkeys:
            #print(f"Layer noised: {k}")
            # adding noise with mu =strength, sigma=1 * strength
            random_model[k] = torch.randn(random_model[k].shape, device=self.device) * self.strength
        return random_model