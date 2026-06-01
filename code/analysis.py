import torch
import argparse
import pickle


def compute_per_feature_interference(W):
    """
    Compute per-feature interference (sum of absolute off-diagonal Gram matrix entries per row).
    
    Args:
        W: Weight matrix of shape (n_hidden, n_features)
        
    Returns:
        interference: Per-feature interference of shape (n_features,)
    """
    W_norm = W / (W.norm(dim=0, keepdim=True) + 1e-8)
    gram = W_norm.T @ W_norm  # (n_features, n_features)
    off_diag = gram - torch.diag(torch.diag(gram))
    interference = off_diag.sum(dim=1)  # Sum per row
    
    return interference


def analyze_interference(results, sparsity_levels, feature_drop_threshold=0.01):
    """
    Analyze kept vs dropped features based on interference.
    
    Args:
        results: Dictionary of results from main.py
        sparsity_levels: List of sparsity levels to analyze
        feature_drop_threshold: Threshold for dropping features
    """
    
    print(f"\n{'='*70}")
    print("DETAILED INTERFERENCE ANALYSIS: Kept vs Dropped Features")
    print(f"{'='*70}\n")
    
    for sparsity in sparsity_levels:
        print(f"\n{'='*70}")
        print(f"Sparsity = {sparsity}")
        print(f"{'='*70}")
        
        # ===== Clean Model =====
        print(f"\n--- CLEAN MODEL ---")
        
        w_clean = results[sparsity]['feature_matrix_clean']
        interference_clean = compute_per_feature_interference(w_clean)
        
        # Identify kept vs dropped
        col_norms_clean = (w_clean ** 2).sum(dim=0)
        mask_clean = col_norms_clean > feature_drop_threshold
        
        kept_interference_clean = interference_clean[mask_clean]
        dropped_interference_clean = interference_clean[~mask_clean]
        
        print(f"\nKept features ({len(kept_interference_clean)} features):")
        print(f"   Mean      = {kept_interference_clean.mean():.4f}")
        print(f"   Median    = {kept_interference_clean.median():.4f}")
        print(f"   Std       = {kept_interference_clean.std():.4f}")
        print(f"   Min       = {kept_interference_clean.min():.4f}")
        print(f"   Max       = {kept_interference_clean.max():.4f}")
        
        print(f"\nDropped features ({len(dropped_interference_clean)} features):")
        print(f"   Mean      = {dropped_interference_clean.mean():.4f}")
        print(f"   Median    = {dropped_interference_clean.median():.4f}")
        print(f"   Std       = {dropped_interference_clean.std():.4f}")
        print(f"   Min       = {dropped_interference_clean.min():.4f}")
        print(f"   Max       = {dropped_interference_clean.max():.4f}")
        
        # ===== Adversarial Model =====
        print(f"\n--- ADVERSARIAL MODEL ---")
        
        w_adv = results[sparsity]['feature_matrix_adv']
        interference_adv = compute_per_feature_interference(w_adv)
        
        # Identify kept vs dropped
        col_norms_adv = (w_adv ** 2).sum(dim=0)
        mask_adv = col_norms_adv > feature_drop_threshold
        
        kept_interference_adv = interference_adv[mask_adv]
        dropped_interference_adv = interference_adv[~mask_adv]
        
        print(f"\nKept features ({len(kept_interference_adv)} features):")
        print(f"   Mean      = {kept_interference_adv.mean():.4f}")
        print(f"   Median    = {kept_interference_adv.median():.4f}")
        print(f"   Std       = {kept_interference_adv.std():.4f}")
        print(f"   Min       = {kept_interference_adv.min():.4f}")
        print(f"   Max       = {kept_interference_adv.max():.4f}")
        
        print(f"\nDropped features ({len(dropped_interference_adv)} features):")
        print(f"   Mean      = {dropped_interference_adv.mean():.4f}")
        print(f"   Median    = {dropped_interference_adv.median():.4f}")
        print(f"   Std       = {dropped_interference_adv.std():.4f}")
        print(f"   Min       = {dropped_interference_adv.min():.4f}")
        print(f"   Max       = {dropped_interference_adv.max():.4f}")
        
        # ===== Comparison =====
        print(f"\n--- COMPARISON (Adv - Clean) ---")
        print(f"\nKept features:")
        print(f"   Mean interference change = {kept_interference_adv.mean() - kept_interference_clean.mean():.4f}")
        print(f"   Median interference change = {kept_interference_adv.median() - kept_interference_clean.median():.4f}")
        
        print(f"\nDropped features:")
        print(f"   Mean interference change = {dropped_interference_adv.mean() - dropped_interference_clean.mean():.4f}")
        print(f"   Median interference change = {dropped_interference_adv.median() - dropped_interference_clean.median():.4f}")
        
        # ===== Feature Type Breakdown =====
        print(f"\n--- FEATURE TYPE BREAKDOWN (Adversarial Model) ---")
        
        robust_idx = results[sparsity]['robust_idx']
        non_robust_idx = results[sparsity]['non_robust_idx']
        
        interference_robust = interference_adv[robust_idx]
        interference_non_robust = interference_adv[non_robust_idx]
        
        print(f"\nRobust features ({len(robust_idx)} features):")
        print(f"   Mean interference = {interference_robust.mean():.4f}")
        print(f"   Median interference = {interference_robust.median():.4f}")
        print(f"   Std = {interference_robust.std():.4f}")
        
        print(f"\nNon-robust features ({len(non_robust_idx)} features):")
        print(f"   Mean interference = {interference_non_robust.mean():.4f}")
        print(f"   Median interference = {interference_non_robust.median():.4f}")
        print(f"   Std = {interference_non_robust.std():.4f}")
        
        # Dropped indices
        dropped_idx_adv = results[sparsity]['dropped_idx_adv']
        
        overlap_non_robust = len(set(dropped_idx_adv.tolist()) & set(non_robust_idx.tolist()))
        overlap_robust = len(set(dropped_idx_adv.tolist()) & set(robust_idx.tolist()))
        
        print(f"\nDropped feature type breakdown:")
        print(f"   {overlap_non_robust}/{len(non_robust_idx)} non-robust features dropped")
        print(f"   {overlap_robust}/{len(robust_idx)} robust features dropped")
        
        if len(dropped_idx_adv) > 0:
            # Interference of dropped features by type
            dropped_robust = [i for i in dropped_idx_adv.tolist() if i in robust_idx.tolist()]
            dropped_non_robust = [i for i in dropped_idx_adv.tolist() if i in non_robust_idx.tolist()]
            
            if dropped_robust:
                print(f"\n   Dropped robust features: {dropped_robust}")
                print(f"      Mean interference: {interference_adv[dropped_robust].mean():.4f}")
            
            if dropped_non_robust:
                print(f"\n   Dropped non-robust features: {dropped_non_robust}")
                print(f"      Mean interference: {interference_adv[dropped_non_robust].mean():.4f}")
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detailed interference analysis of feature dropping"
    )
    
    parser.add_argument(
        "--results-file",
        type=str,
        default="results.pkl",
        help="Path to saved results pickle file (default: results.pkl)"
    )
    
    parser.add_argument(
        "--sparsity",
        type=float,
        nargs="+",
        default=[0.78, 0.80, 0.88, 0.90],
        help="Sparsity levels to analyze (default: 0.78 0.80 0.88 0.90)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Feature dropping threshold (default: 0.01)"
    )
    
    args = parser.parse_args()
    
    # Load results
    try:
        with open(args.results_file, 'rb') as f:
            results = pickle.load(f)
        print(f"Loaded results from {args.results_file}")
    except FileNotFoundError:
        print(f"Error: Could not find {args.results_file}")
        print("Make sure to run main.py first to generate results.")
        exit(1)
    
    # Run analysis
    analyze_interference(results, args.sparsity, args.threshold)
