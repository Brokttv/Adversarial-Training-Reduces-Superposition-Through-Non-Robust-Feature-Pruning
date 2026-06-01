import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import gc
import argparse

from data import structured_data, normal_data
from model import Model
from training import adv_train, clean_train
from utils import (
    set_seed, 
    superposition, 
    rep_power, 
    compute_interference,
    count_dropped_features
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run superposition experiments with adversarial training"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=1920,
        help="Number of training epochs (default: 1920)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)"
    )
    
    parser.add_argument(
        "--sparsity",
        type=float,
        nargs="+",
        default=[0.78, 0.80, 0.88, 0.90],
        help="Sparsity levels to sweep over (default: 0.78 0.80 0.88 0.90)"
    )
    
    parser.add_argument(
        "--n-hidden",
        type=int,
        default=20,
        help="Number of hidden dimensions (default: 20)"
    )
    
    parser.add_argument(
        "--n-features",
        type=int,
        default=100,
        help="Number of input features (default: 100)"
    )
    
    parser.add_argument(
        "--n-robust",
        type=int,
        default=70,
        help="Number of robust features (default: 70)"
    )
    
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Number of training samples (default: 5000)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Feature dropping threshold (default: 0.01)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Device to use (default: auto-detect)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    
    parser.add_argument(
        "--data-type",
        type=str,
        choices=["normal", "structured"],
        default="structured",
        help="Data type: 'normal' (random features) or 'structured' (with known robust/non-robust) (default: structured)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    # ====================== Setup ======================
    args = parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    set_seed(args.seed)
    criterion = nn.MSELoss()
    
    # Hyperparameters from CLI
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    SPARSITY_LEVELS = args.sparsity
    N_HIDDEN = args.n_hidden
    N_FEATURES = args.n_features
    N_ROBUST = args.n_robust
    N_SAMPLES = args.n_samples
    FEATURE_DROP_THRESHOLD = args.threshold
    LEARNING_RATE = args.lr
    
    # Print configuration
    print(f"\n{'='*60}")
    print("Configuration")
    print(f"{'='*60}")
    print(f"Data type: {args.data_type}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Sparsity levels: {SPARSITY_LEVELS}")
    print(f"Hidden dimensions: {N_HIDDEN}")
    print(f"Input features: {N_FEATURES}")
    print(f"Robust features: {N_ROBUST}")
    print(f"Training samples: {N_SAMPLES}")
    print(f"Feature drop threshold: {FEATURE_DROP_THRESHOLD}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}\n")
    
    # ====================== Run Experiments ======================
    results = {}

    for sparsity in SPARSITY_LEVELS:
        print(f"\n{'='*60}")
        print(f"Running experiment for sparsity = {sparsity}")
        print(f"{'='*60}")
        
        # Create data
        if args.data_type == "structured":
            train_input, train_target, robust_idx, non_robust_idx = structured_data(
                sparsity=sparsity,
                num_samples=N_SAMPLES,
                n_features=N_FEATURES,
                n_robust=N_ROBUST
            )
            has_feature_types = True
        else:  # normal
            train_input, train_target = normal_data(
                sparsity=sparsity,
                num_samples=N_SAMPLES,
                n_features=N_FEATURES
            )
            robust_idx = torch.tensor([])
            non_robust_idx = torch.tensor([])
            has_feature_types = False
        
        train_dataset = TensorDataset(train_input, train_target)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Initialize models
        model_clean = Model(n=N_HIDDEN, m=N_FEATURES)
        model_adv = Model(n=N_HIDDEN, m=N_FEATURES)
        model_clean.to(device)
        model_adv.to(device)
        
        optimizer_clean = torch.optim.AdamW(model_clean.parameters(), lr=LEARNING_RATE)
        optimizer_adv = torch.optim.AdamW(model_adv.parameters(), lr=LEARNING_RATE)
        
        # Train models
        for epoch in range(EPOCHS + 1):
            if epoch % 1000 == 0:
                print(f"  Epoch: {epoch}/{EPOCHS}")
            
            clean_train(train_dataloader, criterion, optimizer_clean, model_clean, device)
            adv_train(train_dataloader, criterion, optimizer_adv, model_adv, device)
        
        print(f"  Training complete.")
        
        # Compute metrics
        density_clean = superposition(model_clean)
        density_adv = superposition(model_adv)
        rep_clean = rep_power(model_clean)
        rep_adv = rep_power(model_adv)
        interf_clean = compute_interference(model_clean)
        interf_adv = compute_interference(model_adv)
        
        dropped_clean, dropped_idx_clean, col_norms_clean = count_dropped_features(model_clean, FEATURE_DROP_THRESHOLD)
        dropped_adv, dropped_idx_adv, col_norms_adv = count_dropped_features(model_adv, FEATURE_DROP_THRESHOLD)
        
        # Store results
        results[sparsity] = {
            "interference_clean": interf_clean,
            "interference_adv": interf_adv,
            "superposition_clean": density_clean,
            "superposition_adv": density_adv,
            "rep_power_clean": rep_clean,
            "rep_power_adv": rep_adv,
            "dropped_features_clean": dropped_clean,
            "dropped_features_adv": dropped_adv,
            "dropped_idx_clean": dropped_idx_clean,
            "dropped_idx_adv": dropped_idx_adv,
            "col_norms_clean": col_norms_clean,
            "col_norms_adv": col_norms_adv,
            "feature_matrix_clean": model_clean.w.detach().cpu(),
            "feature_matrix_adv": model_adv.w.detach().cpu(),
            "robust_idx": robust_idx.cpu(),
            "non_robust_idx": non_robust_idx.cpu()
        }
        
        # Cleanup
        del model_clean, model_adv, optimizer_clean, optimizer_adv
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()


    # ====================== Analysis ======================
    print(f"\n{'='*60}")
    print("ANALYSIS: Feature Dropping and Feature Type")
    print(f"{'='*60}\n")

    for sparsity in SPARSITY_LEVELS:
        print(f"\nSparsity = {sparsity}")
        print("-" * 60)
        
        w_clean = results[sparsity]['feature_matrix_clean']
        w_adv = results[sparsity]['feature_matrix_adv']
        robust_idx = results[sparsity]['robust_idx']
        non_robust_idx = results[sparsity]['non_robust_idx']
        
        col_norms_clean = results[sparsity]['col_norms_clean']
        col_norms_adv = results[sparsity]['col_norms_adv']
        
        # Summary statistics
        print(f"Clean - mean col norm: {col_norms_clean.mean():.4f}, "
              f"std: {col_norms_clean.std():.4f}, "
              f"num dropped (<{FEATURE_DROP_THRESHOLD}): {(col_norms_clean < FEATURE_DROP_THRESHOLD).sum().item()}")
        
        print(f"Adv   - mean col norm: {col_norms_adv.mean():.4f}, "
              f"std: {col_norms_adv.std():.4f}, "
              f"num dropped (<{FEATURE_DROP_THRESHOLD}): {(col_norms_adv < FEATURE_DROP_THRESHOLD).sum().item()}")
        
        # Feature type analysis (only for structured data)
        if has_feature_types and len(results[sparsity]['robust_idx']) > 0:
            print(f"\nFeature breakdown:")
            print(f"  Robust features: {robust_idx.tolist()}")
            print(f"  Non-robust features: {non_robust_idx.tolist()}")
            
            # Check which features are dropped in adversarial model
            dropped_adv = results[sparsity]['dropped_idx_adv']
            
            # Count overlap with non-robust features
            overlap_non_robust = len(set(dropped_adv.tolist()) & set(non_robust_idx.tolist()))
            overlap_robust = len(set(dropped_adv.tolist()) & set(robust_idx.tolist()))
            
            print(f"\nAdversarial model drops:")
            print(f"  {overlap_non_robust}/{len(non_robust_idx)} non-robust features")
            print(f"  {overlap_robust}/{len(robust_idx)} robust features")
            
            if len(dropped_adv) > 0:
                print(f"  Dropped feature indices: {dropped_adv.tolist()}")

    print(f"\n{'='*60}")
    print("Experiment complete!")
    print(f"{'='*60}")
    
    # Save results for later analysis
    import pickle
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to results.pkl")
    print(f"Run 'python analysis.py' for detailed interference analysis")
