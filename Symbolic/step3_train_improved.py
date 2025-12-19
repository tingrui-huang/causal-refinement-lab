"""
Step 3: Neural LP Training (Improved Version)

This improved version uses:
1. Stronger L1 regularization to encourage sparsity
2. More training epochs
3. Learning rate scheduling
4. Better initialization strategy
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from neural_lp import (
    NeuralLP,
    print_adjacency_comparison,
    compute_loss
)


def train_neural_lp_improved(
    model,
    data,
    target_idx,
    n_epochs=3000,
    learning_rate=0.05,
    l1_lambda=0.05,
    print_every=200,
    verbose=True
):
    """
    Improved training with learning rate scheduling.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200, verbose=False
    )
    
    history = {
        'total_loss': [],
        'mse_loss': [],
        'l1_loss': []
    }
    
    targets = data[:, target_idx]
    
    if verbose:
        print(f"\nTraining Neural LP to predict variable {target_idx}")
        print(f"Epochs: {n_epochs}, Initial LR: {learning_rate}, L1: {l1_lambda}")
        print("-" * 70)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # Forward pass
        predictions = model.predict_target(data, target_idx)
        
        # Compute loss
        adjacency = model.adjacency()
        loss, loss_dict = compute_loss(
            predictions, targets, adjacency, l1_lambda
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Learning rate scheduling
        scheduler.step(loss)
        
        # Record history
        history['total_loss'].append(loss_dict['total'])
        history['mse_loss'].append(loss_dict['mse'])
        history['l1_loss'].append(loss_dict['l1'])
        
        # Print progress
        if verbose and (epoch + 1) % print_every == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:4d} | "
                  f"Loss: {loss_dict['total']:.4f} | "
                  f"MSE: {loss_dict['mse']:.4f} | "
                  f"L1: {loss_dict['l1']:.4f} | "
                  f"LR: {current_lr:.5f}")
        
        # Early stopping check
        if loss_dict['total'] < best_loss:
            best_loss = loss_dict['total']
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Stop if no improvement for 500 epochs
        if patience_counter > 500:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    if verbose:
        print("-" * 70)
        print(f"Training completed! Best loss: {best_loss:.4f}")
    
    return history


def main():
    print("=" * 70)
    print("STEP 3: NEURAL LP TRAINING (IMPROVED)")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    data_df = pd.read_csv('data/ground_truth_data.csv')
    data = torch.FloatTensor(data_df.values)
    mask = np.load('data/mask_matrix.npy')
    gt_adjacency = np.load('data/ground_truth_adjacency.npy')
    
    print(f"  Data shape: {data.shape}")
    print(f"  Trainable edges: {int(mask.sum())}")
    
    # Initialize model with better settings
    print("\n" + "=" * 70)
    print("INITIALIZING MODEL")
    print("=" * 70)
    
    model = NeuralLP(
        n_vars=3,
        mask=mask,
        max_hops=2,
        init_value=0.05  # Smaller initialization
    )
    
    print("\nModel Configuration:")
    print("  Variables: 3 (X, Y, Z)")
    print("  Max hops: 2")
    print("  Target: Z (index 2)")
    print("  Initialization: Small random (0.05)")
    
    # Train with improved settings
    print("\n" + "=" * 70)
    print("TRAINING (IMPROVED HYPERPARAMETERS)")
    print("=" * 70)
    print("\nImprovements:")
    print("  [+] Stronger L1 regularization (0.05 vs 0.01)")
    print("  [+] More epochs (3000 vs 1000)")
    print("  [+] Higher initial learning rate (0.05 vs 0.01)")
    print("  [+] Learning rate scheduling")
    print("  [+] Early stopping")
    
    history = train_neural_lp_improved(
        model=model,
        data=data,
        target_idx=2,
        n_epochs=3000,
        learning_rate=0.05,
        l1_lambda=0.05,
        print_every=200,
        verbose=True
    )
    
    # Get results
    learned_adj = model.get_adjacency_matrix()
    
    # Visualize
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(history['total_loss'])
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Total Loss')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history['mse_loss'], color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE Loss')
        axes[1].set_title('Prediction Error (MSE)')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(history['l1_loss'], color='green')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('L1 Loss')
        axes[2].set_title('Sparsity Regularization (L1)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/training_loss_improved.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved training curves to results/training_loss_improved.png")
        plt.close()
    except Exception as e:
        print(f"\n[WARNING] Could not create plots: {e}")
    
    # Print results
    print_adjacency_comparison(
        learned=learned_adj,
        ground_truth=gt_adjacency,
        var_names=['X', 'Y', 'Z'],
        threshold=0.3  # Lower threshold
    )
    
    # Detailed analysis
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS")
    print("=" * 70)
    
    print("\nTrue Edges (should be strong):")
    print(f"  X -> Y: {learned_adj[0, 1]:.4f} (truth: 2.0)")
    print(f"  Y -> Z: {learned_adj[1, 2]:.4f} (truth: -3.0)")
    
    print("\nReverse Edges (should be weak):")
    print(f"  Y -> X: {learned_adj[1, 0]:.4f} (truth: 0.0)")
    print(f"  Z -> Y: {learned_adj[2, 1]:.4f} (truth: 0.0)")
    
    print("\nSpurious Edges (should be weak):")
    print(f"  X -> Z: {learned_adj[0, 2]:.4f} (truth: 0.0)")
    print(f"  Z -> X: {learned_adj[2, 0]:.4f} (truth: 0.0)")
    
    # Success metrics
    print("\n" + "=" * 70)
    print("SUCCESS METRICS")
    print("=" * 70)
    
    true_edges_strong = (abs(learned_adj[0, 1]) > 0.3 and 
                         abs(learned_adj[1, 2]) > 0.3)
    reverse_weak = (abs(learned_adj[1, 0]) < 0.2 and 
                    abs(learned_adj[2, 1]) < 0.2)
    spurious_weak = (abs(learned_adj[0, 2]) < 0.2 and 
                     abs(learned_adj[2, 0]) < 0.2)
    
    print(f"  True edges strong: {'[OK]' if true_edges_strong else '[FAIL]'}")
    print(f"  Reverse edges weak: {'[OK]' if reverse_weak else '[FAIL]'}")
    print(f"  Spurious edges weak: {'[OK]' if spurious_weak else '[FAIL]'}")
    
    success = true_edges_strong and reverse_weak and spurious_weak
    
    if success:
        print("\n*** REFINEMENT SUCCESSFUL! ***")
        print("Neural LP successfully refined the poor FCI result!")
        print("The model learned the correct causal structure X -> Y -> Z")
    else:
        print("\n*** PARTIAL SUCCESS ***")
        print("Model learned some structure but may need further tuning.")
    
    # Save results
    np.save('results/learned_adjacency_improved.npy', learned_adj)
    print(f"\nSaved results to results/learned_adjacency_improved.npy")
    
    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()





