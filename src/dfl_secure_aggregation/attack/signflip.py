from typing import OrderedDict

class SignFlip:
    def __init__(self, attack_args: dict):
        # flipped model will go in opposite direction as the normally trained model
        self.flipped_model = None
    def attack(self, model: OrderedDict):
        print("[SignFlippingAttack]")
        for k in model:
            model[k] = -model[k]
        return model
