"""
Neural network model for reward prediction.
Simple MLP that takes [user_emb || repo_emb] and predicts reward.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Tuple


class RewardModel(nn.Module):
    """
    Neural network for predicting user-repo interaction reward.

    Architecture:
        Input: 768-dim (384 user + 384 repo)
        Hidden: 256 -> 64
        Output: 1 (predicted reward)
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: Tuple[int, ...] = (256, 64),
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # the output is a single number for reward modeling
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)  # *args – Unpacking ho rhi hai
        
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor, shape (batch_size, 768)
               [user_emb (384) || repo_emb (384)]
        
        Returns:
            Predicted rewards, shape (batch_size, 1)
        """
        return self.network(x)
    
if __name__ == "__main__":
    # Test the model
    model = RewardModel()
    
    # Dummy input: batch of 10 user-repo pairs
    batch_size = 10
    dummy_input = torch.randn(batch_size, 768)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample predictions: {output.squeeze().detach().numpy()[:5]}")