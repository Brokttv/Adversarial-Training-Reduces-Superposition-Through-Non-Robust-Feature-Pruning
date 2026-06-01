import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Set random seed for reproducibility across numpy, torch, and CUDA.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def superposition(model):
    """
    Compute superposition metric: Frobenius norm of W divided by number of dimensions.
    Φ = ||W||_F / n
    
    Higher values indicate more superposition.
    
    Args:
        model: Model with weight matrix W
        
    Returns:
        Superposition value (scalar)
    """
    return ((model.w ** 2).sum() / model.w.shape[0]).item()


def rep_power(model):
    """
    Compute representational power: squared Frobenius norm of W.
    P = ||W||_F^2
    
    Measures total energy distributed across all feature directions.
    
    Args:
        model: Model with weight matrix W
        
    Returns:
        Representational power (scalar)
    """
    return (model.w ** 2).sum().item()


def compute_interference(model):
    """
    Compute mean interference from Gram matrix of W.
    
    Measures the degree of non-orthogonality between feature directions.
    
    Args:
        model: Model with weight matrix W
        
    Returns:
        Mean interference (scalar)
    """
    W = model.w
    W_norm = W / (W.norm(dim=0, keepdim=True) + 1e-8)
    gram = W_norm.T @ W_norm  # (m, m)
    
    # Zero out diagonal, sum absolute off-diagonal entries
    off_diag = gram - torch.diag(torch.diag(gram))
    
    return off_diag.mean().item()


def count_dropped_features(model, threshold=0.01):
    """
    Count how many features are dropped (column norm below threshold).
    
    Args:
        model: Model with weight matrix W
        threshold: Threshold for dropping features
        
    Returns:
        Number of dropped features and indices
    """
    col_norms = (model.w ** 2).sum(dim=0)
    dropped_mask = col_norms < threshold
    dropped_count = dropped_mask.sum().item()
    dropped_indices = torch.where(dropped_mask)[0]
    
    return dropped_count, dropped_indices, col_norms
