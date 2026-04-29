from dataclasses import dataclass
from torch.utils.data import Dataset

@dataclass
class DataBundle:
    train: Dataset
    test: Dataset | None = None
    val: Dataset | None = None
