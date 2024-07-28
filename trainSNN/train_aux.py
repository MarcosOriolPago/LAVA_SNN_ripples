from torch import tensor
from torch.utils.data import Dataset
import torch


class torchSlayer_Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert the sample to tensor (if it's not already)
        sample = tensor(self.data[idx], dtype=torch.float32)
        # Get the corresponding label as int
        label = int(self.labels[idx])
        return sample, label