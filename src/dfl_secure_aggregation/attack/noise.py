import torch
from typing import OrderedDict

class Noise():
    """
    Function to perform noise injection attack on the received weights.

    Modifed from: https://github.com/enriquetomasmb/fedstellar/blob/main/fedstellar/attacks/aggregation.py
        under GPL 3.0 License
    """
    def __init__(self, attack_args: dict):
        self.strength = attack_args['strength']
        self.device = attack_args.get('device', torch.device('cpu'))


    def attack(self, model: OrderedDict):
        """
        Perform noise injection attack on the received weights. 
        :param received_weights: A dictionary containing the received weights.
        Returns A dictionary containing the noise injected weights.
        """
        print("[NoiseInjectionAttack] Performing noise injection attack")
        lkeys = list(model.keys())
        for k in lkeys:
            #print(f"Layer noised: {k}")
            # adding noise with mu =0, sigma=1 * strength
            model[k] = model[k].to(self.device)
            model[k].data += torch.randn(model[k].shape, device=self.device) * self.strength 
            
        return model