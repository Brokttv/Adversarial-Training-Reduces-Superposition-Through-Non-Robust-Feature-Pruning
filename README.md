
## Setup
```bash
pip install -r requirements.txt
```


## Requirements

```
torch
numpy
```

### Default run (all defaults from paper):
```bash
python main.py
```

### Custom configuration:
```bash
python main.py --epochs 500 --batch-size 32 --sparsity 0.8 0.85 0.9
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 1920 | Number of training epochs |
| `--batch-size` | 64 | Batch size for training |
| `--data-type` | structured | Data type: 'normal' or 'structured' (with known robust/non-robust) |
| `--sparsity` | [0.78, 0.80, 0.88, 0.90] | Sparsity levels to sweep (space-separated) |
| `--n-hidden` | 20 | Number of hidden dimensions |
| `--n-features` | 100 | Number of input features |
| `--n-robust` | 70 | Number of robust features (only for structured data) |
| `--n-samples` | 5000 | Number of training samples |
| `--threshold` | 0.01 | Feature dropping threshold |
| `--seed` | 42 | Random seed for reproducibility |
| `--device` | auto | Device: 'cpu' or 'cuda' (auto-detects if not specified) |
| `--lr` | 1e-3 | Learning rate for optimizers |

## Usage Examples

### Structured data (with known robust/non-robust features):
```bash
python main.py --data-type structured
```

### Normal data (random features, no feature type labeling):
```bash
python main.py --data-type normal
```

### Fast experiment (fewer epochs, smaller dataset):
```bash
python main.py --epochs 100 --n-samples 1000 --sparsity 0.8 0.88
```

### GPU-accelerated with custom learning rate:
```bash
python main.py --device cuda --lr 5e-3 --epochs 2000
```

### Single sparsity level with custom metrics threshold:
```bash
python main.py --sparsity 0.85 --threshold 0.005 --epochs 500
```

### Full reproducible run:
```bash
python main.py --seed 42 --device cuda --epochs 1920 --sparsity 0.78 0.80 0.88 0.90
```

## Output

The script prints:
1. **Configuration summary** - All CLI arguments used
2. **Training progress** - Epoch updates every 1000 steps per sparsity level
3. **Analysis results** - Feature dropping statistics and robust/non-robust breakdown for each sparsity level

# Key metrics computed and displayed:
- Superposition (Φ = ||W||_F / n)
- Representational power (P = ||W||_F²)
- Interference (mean off-diagonal Gram matrix entries)
- Dropped features (column norms < threshold)
- Robust vs. non-robust feature overlap

**Results are saved to `results.pkl`** for later analysis.

## Detailed Interference Analysis

**After running `main.py`, use `analysis.py` for further investigation:**

```bash
# Default analysis
python analysis.py

# Custom threshold and sparsity levels
python analysis.py --threshold 0.005 --sparsity 0.8 0.85

# Custom results file
python analysis.py --results-file my_results.pkl
```

**Analysis output includes:**
- Per-feature interference (sum of Gram matrix off-diagonal entries)
- Kept vs dropped feature interference statistics
- Robust vs non-robust feature interference breakdown
- Feature type analysis (which features are dropped)



