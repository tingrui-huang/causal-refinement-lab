"""
Step 3: Neural LP Training

This script trains a Neural LP model to refine the poor FCI results.

The model learns to:
1. Orient undirected edges correctly (X->Y, Y->Z)
2. Remove spurious edges (X-Z)
3. Discover the true causal structure through data

Training objective: Predict Z from X and Y using multi-hop reasoning.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from neural_lp import (
    NeuralLP,
    train_neural_lp,
    print_adjacency_comparison,
    threshold_adjacency
)


def load_data():
    """Load generated data and matrices."""
    print("=" * 70)
    print("STEP 3: NEURAL LP TRAINING")
    print("=" * 70)
    
    print("\nLoading data...")
    
    # Load CSV data
    data_df = pd.read_csv('data/ground_truth_data.csv')
    data = torch.FloatTensor(data_df.values)
    
    # Load mask matrix
    mask = np.load('data/mask_matrix.npy')
    
    # Load ground truth
    gt_adjacency = np.load('data/ground_truth_adjacency.npy')
    gt_weights = np.load('data/ground_truth_weights.npy')
    
    print(f"  Data shape: {data.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Trainable edges: {int(mask.sum())}")
    
    return data, mask, gt_adjacency, gt_weights


def print_setup(mask, gt_adjacency):
    """Print training setup information."""
    print("\n" + "=" * 70)
    print("TRAINING SETUP")
    print("=" * 70)
    
    print("\nGround Truth: X -> Y -> Z")
    print("  Y = 2.0 * X + noise")
    print("  Z = -3.0 * Y + noise")
    
    print("\nFCI Result (Poor):")
    print("  X - Y (undirected)")
    print("  Y - Z (undirected)")
    print("  X - Z (spurious)")
    
    print("\nMask Matrix (Allowed Edges):")
    var_names = ['X', 'Y', 'Z']
    print("      " + "  ".join([f"{v:>6}" for v in var_names]))
    for i, var in enumerate(var_names):
        row_str = "  ".join([f"{int(mask[i, j]):6d}" for j in range(len(var_names))])
        print(f"  {var}   {row_str}")
    
    print("\nTraining Task:")
    print("  Predict Z from X and Y using multi-hop reasoning")
    print("  Expected: Model learns X->Y->Z path is most predictive")


def visualize_training(history, output_path='results/training_loss.png'):
    """Visualize training loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Total loss
    axes[0].plot(history['total_loss'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].grid(True, alpha=0.3)
    
    # MSE loss
    axes[1].plot(history['mse_loss'], color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE Loss')
    axes[1].set_title('Prediction Error (MSE)')
    axes[1].grid(True, alpha=0.3)
    
    # L1 loss
    axes[2].plot(history['l1_loss'], color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('L1 Loss')
    axes[2].set_title('Sparsity Regularization (L1)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved training curves to {output_path}")
    plt.close()


def analyze_learned_structure(learned_adj, gt_adj, var_names=['X', 'Y', 'Z']):
    """Analyze what the model learned."""
    print("\n" + "=" * 70)
    print("ANALYSIS: WHAT DID THE MODEL LEARN?")
    print("=" * 70)
    
    print("\nExpected High Weights (True Edges):")
    print(f"  X -> Y: {learned_adj[0, 1]:.4f} (expected: high, ~2.0)")
    print(f"  Y -> Z: {learned_adj[1, 2]:.4f} (expected: high, ~-3.0)")
    
    print("\nExpected Low Weights (Reverse Edges):")
    print(f"  Y -> X: {learned_adj[1, 0]:.4f} (expected: low, ~0.0)")
    print(f"  Z -> Y: {learned_adj[2, 1]:.4f} (expected: low, ~0.0)")
    
    print("\nExpected Low Weights (Spurious Edges):")
    print(f"  X -> Z: {learned_adj[0, 2]:.4f} (expected: low, ~0.0)")
    print(f"  Z -> X: {learned_adj[2, 0]:.4f} (expected: low, ~0.0)")
    
    # Check if refinement worked
    print("\n" + "=" * 70)
    print("REFINEMENT SUCCESS CHECK")
    print("=" * 70)
    
    checks = []
    
    # Check 1: X -> Y should be strong
    if abs(learned_adj[0, 1]) > 0.5:
        checks.append(("X -> Y is strong", True))
    else:
        checks.append(("X -> Y is strong", False))
    
    # Check 2: Y -> Z should be strong
    if abs(learned_adj[1, 2]) > 0.5:
        checks.append(("Y -> Z is strong", True))
    else:
        checks.append(("Y -> Z is strong", False))
    
    # Check 3: Reverse edges should be weak
    if abs(learned_adj[1, 0]) < 0.3 and abs(learned_adj[2, 1]) < 0.3:
        checks.append(("Reverse edges are weak", True))
    else:
        checks.append(("Reverse edges are weak", False))
    
    # Check 4: Spurious edges should be weak
    if abs(learned_adj[0, 2]) < 0.3 and abs(learned_adj[2, 0]) < 0.3:
        checks.append(("Spurious edges are weak", True))
    else:
        checks.append(("Spurious edges are weak", False))
    
    for check_name, passed in checks:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {check_name}")
    
    success_rate = sum(1 for _, passed in checks if passed) / len(checks)
    print(f"\nOverall Success Rate: {success_rate:.1%}")
    
    if success_rate >= 0.75:
        print("\n*** REFINEMENT SUCCESSFUL! ***")
        print("Neural LP successfully refined the poor FCI result!")
    else:
        print("\n*** REFINEMENT NEEDS TUNING ***")
        print("Try adjusting hyperparameters (learning rate, L1 lambda, epochs)")


def main():
    # Load data
    data, mask, gt_adjacency, gt_weights = load_data()
    
    # Print setup
    print_setup(mask, gt_adjacency)
    
    # Initialize model
    print("\n" + "=" * 70)
    print("INITIALIZING NEURAL LP MODEL")
    print("=" * 70)
    
    n_vars = 3
    max_hops = 2  # Allow paths up to length 2 (X -> Y -> Z)
    
    model = NeuralLP(
        n_vars=n_vars,
        mask=mask,
        max_hops=max_hops,
        init_value=0.1
    )
    
    print(f"\nModel Configuration:")
    print(f"  Variables: {n_vars} (X, Y, Z)")
    print(f"  Max hops: {max_hops}")
    print(f"  Trainable edges: {int(mask.sum())}")
    print(f"  Target variable: Z (index 2)")
    
    # Train model
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    target_idx = 2  # Predict Z
    
    history = train_neural_lp(
        model=model,
        data=data,
        target_idx=target_idx,
        n_epochs=1000,
        learning_rate=0.01,
        l1_lambda=0.01,
        print_every=100,
        verbose=True
    )
    
    # Get learned adjacency matrix
    learned_adj = model.get_adjacency_matrix()
    
    # Visualize training
    try:
        visualize_training(history)
    except Exception as e:
        print(f"\n[WARNING] Could not create training plots: {e}")
    
    # Print comparison
    print_adjacency_comparison(
        learned=learned_adj,
        ground_truth=gt_adjacency,
        var_names=['X', 'Y', 'Z'],
        threshold=0.5
    )
    
    # Analyze results
    analyze_learned_structure(learned_adj, gt_adjacency)
    
    # Save learned adjacency
    np.save('results/learned_adjacency.npy', learned_adj)
    print(f"\nSaved learned adjacency to results/learned_adjacency.npy")
    
    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE!")
    print("=" * 70)
    print("\nNext Steps:")
    print("  - Analyze learned weights vs ground truth")
    print("  - Test on more complex structures")
    print("  - Integrate with LLM pipeline")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()

