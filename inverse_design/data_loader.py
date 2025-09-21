from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
class MaterialDataset(Dataset):
    """Dataset class for material features and properties"""

    def __init__(self, features: np.ndarray, properties: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.properties = torch.FloatTensor(properties)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.properties[idx]
