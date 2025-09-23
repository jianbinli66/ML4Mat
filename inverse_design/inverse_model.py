from torch import nn
import torch
from typing import List, Optional, Tuple
import numpy as np


class InvertibleNeuralNetwork(nn.Module):
    """Invertible Neural Network for bidirectional mapping between features and properties with condition embeddings"""

    def __init__(self, formula_dim: int, condition_dim: int, property_dim: int,
                 hidden_dims: List[int] = [128, 64], num_coupling_layers: int = 4,
                 embedding_dim: int = 3):
        super(InvertibleNeuralNetwork, self).__init__()

        self.formula_dim = formula_dim
        self.condition_dim = condition_dim
        self.property_dim = property_dim
        self.embedding_dim = embedding_dim

        # Embedding layer for condition features
        self.condition_embedding = nn.Embedding(condition_dim, embedding_dim)

        # Total dimension after embedding
        self.total_dim = formula_dim + embedding_dim + property_dim

        # Create coupling layers
        self.coupling_layers = nn.ModuleList()
        for i in range(num_coupling_layers):
            self.coupling_layers.append(
                AffineCouplingLayer(self.total_dim, hidden_dims)
            )

        # Learnable prior parameters
        self.register_parameter('prior_mean', nn.Parameter(torch.zeros(property_dim)))
        self.register_parameter('prior_log_std', nn.Parameter(torch.zeros(property_dim)))

    def forward(self, formula_features: torch.Tensor, condition_features: torch.Tensor,
                properties: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Embed condition features
        condition_embedded = self.condition_embedding(condition_features)
        condition_embedded = condition_embedded.mean(dim=1)  # Average over condition features
        # Concatenate all features
        x = torch.cat([formula_features, condition_embedded, properties], dim=1)
        log_det_jacobian = torch.zeros(x.shape[0], device=x.device)

        for layer in self.coupling_layers:
            x, ldj = layer(x)
            log_det_jacobian += ldj

        return x, log_det_jacobian

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = z
        for layer in reversed(self.coupling_layers):
            x = layer.inverse(x)

        # Split the output
        formula_features = x[:, :self.formula_dim]
        condition_embedded = x[:, self.formula_dim:self.formula_dim + self.embedding_dim]
        properties = x[:, self.formula_dim + self.embedding_dim:]

        # Convert embedded conditions back to logits (for binary classification)
        condition_logits = self._embedded_to_logits(condition_embedded)

        return formula_features, condition_logits, properties

    def _embedded_to_logits(self, embedded: torch.Tensor) -> torch.Tensor:
        # Simple linear transformation to convert embeddings back to logits
        # This is a simplified approach; you might want to use a more complex decoder
        return torch.matmul(embedded, self.condition_embedding.weight.t())

    def log_prob(self, formula_features: torch.Tensor, condition_features: torch.Tensor,
                 properties: torch.Tensor) -> torch.Tensor:
        z, log_det_jacobian = self.forward(formula_features, condition_features, properties)

        # Prior log probability
        prior_log_prob = -0.5 * torch.sum(
            (z[:, self.formula_dim + self.embedding_dim:] - self.prior_mean) ** 2 / torch.exp(2 * self.prior_log_std) +
            2 * self.prior_log_std + np.log(2 * np.pi), dim=1
        )

        return prior_log_prob + log_det_jacobian


class AffineCouplingLayer(nn.Module):
    """Affine coupling layer for invertible neural network"""

    def __init__(self, dim: int, hidden_dims: List[int] = [64, 32]):
        super(AffineCouplingLayer, self).__init__()

        self.dim = dim
        self.split_idx = dim // 2

        # Scale and translation networks
        self.scale_net = self._build_network(hidden_dims, self.split_idx, self.dim - self.split_idx)
        self.translate_net = self._build_network(hidden_dims, self.split_idx, self.dim - self.split_idx)

    def _build_network(self, hidden_dims: List[int], input_dim: int, output_dim: int) -> nn.Sequential:
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x[:, :self.split_idx], x[:, self.split_idx:]

        s = torch.tanh(self.scale_net(x1))
        t = self.translate_net(x1)

        x2 = x2 * torch.exp(s) + t
        x = torch.cat([x1, x2], dim=1)

        log_det_jacobian = s.sum(dim=1)
        return x, log_det_jacobian

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[:, :self.split_idx], x[:, self.split_idx:]

        s = torch.tanh(self.scale_net(x1))
        t = self.translate_net(x1)

        x2 = (x2 - t) * torch.exp(-s)
        x = torch.cat([x1, x2], dim=1)

        return x