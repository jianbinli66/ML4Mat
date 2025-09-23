from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class MaterialDataset(Dataset):
    """Dataset class for material features and properties with condition embeddings"""

    def __init__(self, formula_features: np.ndarray, condition_features: np.ndarray, properties: np.ndarray):
        self.formula_features = torch.FloatTensor(formula_features)
        self.condition_features = torch.LongTensor(condition_features)  # Changed to LongTensor for embedding
        self.properties = torch.FloatTensor(properties)

    def __len__(self):
        return len(self.formula_features)

    def __getitem__(self, idx):
        return self.formula_features[idx], self.condition_features[idx], self.properties[idx]