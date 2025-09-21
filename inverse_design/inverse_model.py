from torch import nn
import torch
from typing import List, Optional, Tuple
import numpy as np


class InvertibleNeuralNetwork(nn.Module):
    """Invertible Neural Network for bidirectional mapping between features and properties"""

    def __init__(self, feature_dim: int, property_dim: int, hidden_dims: List[int] = [128, 64],
                 num_coupling_layers: int = 4, condition_dim: int = 0):
        super(InvertibleNeuralNetwork, self).__init__()

        self.feature_dim = feature_dim
        self.property_dim = property_dim
        self.total_dim = feature_dim + property_dim
        self.condition_dim = condition_dim

        # Create coupling layers
        self.coupling_layers = nn.ModuleList()
        for i in range(num_coupling_layers):
            self.coupling_layers.append(
                AffineCouplingLayer(self.total_dim, hidden_dims, condition_dim)
            )

        # Learnable prior parameters
        self.register_parameter('prior_mean', nn.Parameter(torch.zeros(property_dim)))
        self.register_parameter('prior_log_std', nn.Parameter(torch.zeros(property_dim)))

    def forward(self, features: torch.Tensor, properties: torch.Tensor,
                condition: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([features, properties], dim=1)
        log_det_jacobian = torch.zeros(x.shape[0], device=x.device)

        for layer in self.coupling_layers:
            x, ldj = layer(x, condition)
            log_det_jacobian += ldj

        return x, log_det_jacobian

    def inverse(self, z: torch.Tensor, condition: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = z
        for layer in reversed(self.coupling_layers):
            x = layer.inverse(x, condition)

        features = x[:, :self.feature_dim]
        properties = x[:, self.feature_dim:]
        return features, properties

    def sample(self, features: torch.Tensor, condition: Optional[torch.Tensor] = None,
               num_samples: int = 1) -> torch.Tensor:
        # Sample from prior
        properties_samples = []
        for _ in range(num_samples):
            eps = torch.randn_like(features[:, :self.property_dim])
            properties = self.prior_mean + torch.exp(self.prior_log_std) * eps
            properties_samples.append(properties)

        return torch.stack(properties_samples, dim=1)

    def log_prob(self, features: torch.Tensor, properties: torch.Tensor,
                 condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        z, log_det_jacobian = self.forward(features, properties, condition)

        # Prior log probability
        prior_log_prob = -0.5 * torch.sum(
            (z[:, self.feature_dim:] - self.prior_mean) ** 2 / torch.exp(2 * self.prior_log_std) +
            2 * self.prior_log_std + np.log(2 * np.pi), dim=1
        )

        return prior_log_prob + log_det_jacobian

class AffineCouplingLayer(nn.Module):
    """Affine coupling layer for invertible neural network"""

    def __init__(self, dim: int, hidden_dims: List[int] = [64, 32], condition_dim: int = 0):
        super(AffineCouplingLayer, self).__init__()

        self.dim = dim
        self.split_idx = dim // 2
        self.condition_dim = condition_dim

        # Scale and translation networks
        self.scale_net = self._build_network(hidden_dims, self.split_idx + condition_dim, self.dim - self.split_idx)
        self.translate_net = self._build_network(hidden_dims, self.split_idx + condition_dim, self.dim - self.split_idx)

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

    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x[:, :self.split_idx], x[:, self.split_idx:]

        if condition is not None:
            scale_input = torch.cat([x1, condition], dim=1) if self.condition_dim > 0 else x1
            translate_input = torch.cat([x1, condition], dim=1) if self.condition_dim > 0 else x1
        else:
            scale_input, translate_input = x1, x1

        s = torch.tanh(self.scale_net(scale_input))
        t = self.translate_net(translate_input)

        x2 = x2 * torch.exp(s) + t
        x = torch.cat([x1, x2], dim=1)

        log_det_jacobian = s.sum(dim=1)
        return x, log_det_jacobian

    def inverse(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        x1, x2 = x[:, :self.split_idx], x[:, self.split_idx:]

        if condition is not None:
            scale_input = torch.cat([x1, condition], dim=1) if self.condition_dim > 0 else x1
            translate_input = torch.cat([x1, condition], dim=1) if self.condition_dim > 0 else x1
        else:
            scale_input, translate_input = x1, x1

        s = torch.tanh(self.scale_net(scale_input))
        t = self.translate_net(translate_input)

        x2 = (x2 - t) * torch.exp(-s)
        x = torch.cat([x1, x2], dim=1)

        return x
