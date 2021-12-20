from pathlib import Path
from torch.utils.data import Dataset
import torch
import numpy as np

# inputs loaded as timesteps x nodes x features transformed to nodes x timesteps x features
# targets loaded as timesteps x nodes transformed to nodes x timesteps
class SpeedbandDataset(Dataset):
    def __init__(self, dataset_dir, means, stds, out_idx = -1):
        self.dataset_dir = Path(dataset_dir)
        self.length = len(list((self.dataset_dir / "inputs").glob("*")))
        input_sample = np.load(self.dataset_dir/"inputs"/"0.npy")
        target_sample = np.load(self.dataset_dir/"targets"/"0.npy")
        self.num_timesteps_input = input_sample.shape[0]
        self.num_timesteps_output = target_sample.shape[0] if out_idx == -1 else 1
        self.num_features = input_sample.shape[2]
        self.num_nodes = input_sample.shape[1]
        self.means = means
        self.stds = stds
        self.out_idx = out_idx
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        data = np.load(self.dataset_dir/"inputs"/"{}.npy".format(idx))
        target = np.load(self.dataset_dir/"targets"/"{}.npy".format(idx))
        data = (data - self.means) / self.stds
        target = (target - self.means[0]) / self.stds[0]
        target = target[self.out_idx:self.out_idx + 1, :] if self.out_idx != -1 else target
        return torch.Tensor(data).permute(1, 0, 2), torch.Tensor(target).permute(1, 0)