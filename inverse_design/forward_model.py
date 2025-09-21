
from typing import List
import torch
import torch.nn as nn


class ForwardModel(nn.Module):
    """Neural network for forward prediction (features → properties)"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.2):
        super(ForwardModel, self).__init__()

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
