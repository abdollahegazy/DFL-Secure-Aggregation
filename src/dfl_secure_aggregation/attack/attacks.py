from .noise import Noise
from .signflip import SignFlip
from .random_noise import RandomNoise
from .alie import ALittleIsEnough


def create_attacker(attack_type, attack_args, node_hash):
    attack_type = attack_type.lower()
    if attack_type == 'noise':
        return Noise(attack_args)
    # elif attack_type == 'innerproduct':
    #     return InnerProductAttack(attack_args, logger)
    elif attack_type=='signflip':
        return SignFlip(attack_args)
    elif attack_type=='randomnoise':
        return RandomNoise(attack_args)
    elif attack_type =='alie':
        return ALittleIsEnough(attack_args)
    else:
        raise ValueError(f'Unknown attack type: {attack_type}')
