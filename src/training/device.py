import torch


def resolve_device(device=None):
    """
    Resolve a configured device string to a torch.device.

    Device selection belongs at the simulation/config boundary. Lower-level
    model, aggregation, and attack code should receive an explicit device.
    """
    if device is None or device == "":
        return torch.device("cpu")

    if isinstance(device, torch.device):
        return device

    device = str(device).lower()
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(device)
