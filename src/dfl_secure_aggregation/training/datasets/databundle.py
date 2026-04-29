from dataclasses import dataclass
from torch.utils.data import Dataset

@dataclass
class DatasetBundle:
    train: Dataset
    test: Dataset | None = None
    val: Dataset | None = None
