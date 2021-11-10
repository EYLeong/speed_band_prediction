from pathlib import Path
from torch.utils.data import Dataset
import torch
import numpy as np

class SpeedbandDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.length = len(list((self.dataset_dir / "inputs").glob("*")))
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        data = np.load(self.dataset_dir/"inputs"/"{}.npy".format(idx))
        target = np.load(self.dataset_dir/"targets"/"{}.npy".format(idx))
        return torch.Tensor(data), torch.Tensor(target)