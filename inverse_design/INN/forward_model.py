from typing import List
import torch
import torch.nn as nn


class ForwardModel(nn.Module):
    """Neural network for forward prediction (features → properties) with condition embeddings"""

    def __init__(self, formula_dim: int, condition_dim: int, output_dim: int,
                 hidden_dims: List[int] = [128, 64, 32], embedding_dim: int = 2,
                 dropout_rate: float = 0.2):
        super(ForwardModel, self).__init__()

        # Embedding layer for condition features
        self.condition_embedding = nn.Embedding(condition_dim, embedding_dim)

        # Calculate input dimension after embedding
        input_dim = formula_dim + embedding_dim

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

    def forward(self, formula_features, condition_features):
        # Embed condition features
        condition_embedded = self.condition_embedding(condition_features)
        condition_embedded = condition_embedded.mean(dim=1)  # Average over condition features

        # Concatenate formula features with embedded condition features
        x = torch.cat([formula_features, condition_embedded], dim=1)

        return self.network(x)