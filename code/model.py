import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Toy model for superposition: encodes m features into n dimensions.
    
    Architecture:
        - Encoder: x @ W.T -> (batch, n)
        - Decoder: ReLU(encoder @ W + b) -> (batch, m)
        
    Args:
        n: Number of hidden dimensions
        m: Number of features
    """
    
    def __init__(self, n: int, m: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n, m) * 0.01)
        self.b = nn.Parameter(torch.zeros(m))
        self.n = n
        self.m = m

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, m)
            
        Returns:
            Reconstructed features of shape (batch, m)
        """
        # Encode
        encoder = x @ self.w.T  # (batch, n)
        
        # Decode with ReLU activation
        return F.relu(encoder @ self.w + self.b)  # (batch, m)
