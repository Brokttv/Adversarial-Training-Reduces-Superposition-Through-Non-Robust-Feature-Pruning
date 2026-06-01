import torch


def normal_data(sparsity: float, 
                num_samples: int, 
                n_features: int = 100):
    """
    Create normal random data with sparsity mask.
    
    Args:
        sparsity: Threshold for sparsity mask (values > sparsity are kept)
        num_samples: Number of samples to generate
        n_features: Number of features
        
    Returns:
        input_data: Input features with sparsity applied
        target: Clone of input (reconstruction task)
    """
    # Base random values
    values = torch.rand(num_samples, n_features)
    
    # Sparsity mask
    mask = values > sparsity
    input_data = values * mask.float()
    
    target = input_data.clone()
    
    return input_data, target


def structured_data(sparsity: float, 
                    num_samples: int, 
                    n_features: int = 100, 
                    n_robust: int = 70):
    """
    Create structured data with known robust and non-robust features.
    Robust features have high amplitude (6.0), non-robust have low (0.2).
    
    Args:
        sparsity: Threshold for sparsity mask (values > sparsity are kept)
        num_samples: Number of samples to generate
        n_features: Number of features
        n_robust: Number of robust features (rest are non-robust)
        
    Returns:
        input_data: Input features with sparsity and amplitude scaling
        target: Clone of input
        robust_idx: Indices of robust features
        non_robust_idx: Indices of non-robust features
    """
    # Base random values
    values = torch.rand(num_samples, n_features)
    
    # Sparsity mask
    mask = values > sparsity
    input_data = values * mask.float()
    
    # Randomly assign robust vs non-robust features
    feature_indices = torch.randperm(n_features)
    robust_idx = feature_indices[:n_robust]           # e.g. 70 robust
    non_robust_idx = feature_indices[n_robust:]       # e.g. 30 non-robust
    
    # Apply amplitude scaling
    input_data[:, robust_idx] *= 6.0      # Strong robust features
    input_data[:, non_robust_idx] *= 0.2  # Weak non-robust features
    
    target = input_data.clone()
    
    return input_data, target, robust_idx, non_robust_idx
