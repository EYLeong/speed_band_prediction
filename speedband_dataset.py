from pathlib import Path
from torch.utils.data import Dataset
import torch
import numpy as np

# inputs loaded as timesteps x nodes x features transformed to nodes x timesteps x features
# targets loaded as timesteps x nodes transformed to nodes x timesteps
class SpeedbandDataset(Dataset):
    '''
    Dataset class for Singapore DataMall traffic speed bands
    -----------------------------
    :constructor params:
        Path/string dataset_dir: Path to the processed data directory
        List means: means of the input features
        List stds: standard deviations of the input features
        int out_idx: index of the output timestep to yield as the target. Yields all output timesteps as an array if -1
        List features: the indices of the features to be yielded as model data. If empty, yields all
    -----------------------------
    '''
    def __init__(self, dataset_dir, means, stds, out_idx = -1, features = []):
        self.dataset_dir = Path(dataset_dir)
        self.length = len(list((self.dataset_dir / "inputs").glob("*")))
        input_sample = np.load(self.dataset_dir/"inputs"/"0.npy")
        target_sample = np.load(self.dataset_dir/"targets"/"0.npy")
        self.num_timesteps_input = input_sample.shape[0]
        self.num_timesteps_output = target_sample.shape[0] if out_idx == -1 else 1
        self.num_features = input_sample.shape[2] if len(features) == 0 else len(features)
        self.num_nodes = input_sample.shape[1]
        self.means = means
        self.stds = stds
        self.out_idx = out_idx
        self.features = features
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        data = np.load(self.dataset_dir/"inputs"/"{}.npy".format(idx))
        target = np.load(self.dataset_dir/"targets"/"{}.npy".format(idx))
        data = (data - self.means) / self.stds
        data = data[:,:,self.features] if len(self.features) != 0 else data
        target = (target - self.means[0]) / self.stds[0]
        target = target[self.out_idx:self.out_idx + 1, :] if self.out_idx != -1 else target
        return torch.Tensor(data).permute(1, 0, 2), torch.Tensor(target).permute(1, 0)