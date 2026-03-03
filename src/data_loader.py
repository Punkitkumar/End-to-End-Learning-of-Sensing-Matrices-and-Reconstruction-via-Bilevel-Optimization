import torch
import numpy as np
from torch.utils.data import Dataset

class SparseSignalsDataset(Dataset):
    def __init__(self, file_path, samples_number=10):
        data = np.load(file_path)[:samples_number]
        data = data.T
        self.data = torch.tensor(data).float()
        self.n_samples = self.data.shape[1]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        # Return index along with data for correct warm starting
        return self.data[:, index], index

def collate_signals(batch):
    signals, indices = zip(*batch)
    return torch.stack(signals, dim=1), torch.tensor(indices)
