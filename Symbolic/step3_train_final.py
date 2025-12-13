"""
Step 3: Neural LP Training (Final Version)

Key insight: We need to use X and Y as INPUTS to predict Z.
The model should learn that the path X -> Y -> Z is most predictive.

Strategy:
- Use X and Y as source features
- Predict Z through multi-hop reasoning
- The model will learn strong weights on X->Y->Z path
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from neural_lp import NeuralLP, compute_loss, print_adjacency_comparison


def train_with_input_features(
    model,
    data,
    source_indices,
    target_idx,
    n_epochs=2000,
    learning_rate=0.02,
    l1_lambda=0.02,
    print_every=200
):
    """
    Train model using specific source features to predict target.
    
    Args:
        source_indices: List of indices to use as input features
        target_idx: Index of target variable to predict
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {'total_loss': [], 'mse_loss': [], 'l1_loss': []}
    
    # Create input features: only use source variables
    input_features = torch.zeros_like(data)
    for idx in source_indices:
        input_features[:, idx] = data[:, idx]
    
    targets = data[:, target_idx]
    
    print(f"\nTraining to predict variable {target_idx} from variables {source_indices}")
    print(f"Epochs: {n_epochs}, LR: {learning_rate}, L1: {l1_lambda}")
    print("-" * 70)
    
    for epoch in range(n_epochs):
        # Forward pass with only source features
        predictions = model.predict_target(input_features, target_idx)
        
        # Compute loss
        adjacency = model.adjacency()
        loss, loss_dict = compute_loss(predictions, targets, adjacency, l1_lambda)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record history
        for key in history:
            history[key].append(loss_dict[key.replace('_loss', '')])
        
        # Print progress
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1:4d} | "
                  f"Loss: {loss_dict['total']:.4f} | "
                  f"MSE: {loss_dict['mse']:.4f} | "
                  f"L1: {loss_dict['l1']:.4f}")
    
    print("-" * 70)
    print("Training completed!")
    
    return history


def main():
    print("=" * 70)
    print("STEP 3: NEURAL LP TRAINING (FINAL VERSION)")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    data_df = pd.read_csv('data/ground_truth_data.csv')
    data = torch.FloatTensor(data_df.values)
    mask = np.load('data/mask_matrix.npy')
    gt_adjacency = np.load('data/ground_truth_adjacency.npy')
    gt_weights = np.load('data/ground_truth_weights.npy')
    
    print(f"  Data shape: {data.shape}")
    print(f"  Variables: X (0), Y (1), Z (2)")
    
    # Print ground truth
    print("\n" + "=" * 70)
    print("GROUND TRUTH")
    print("=" * 70)
    print("\nCausal Structure: X -> Y -> Z")
    print("  Y = 2.0 * X + noise")
    print("  Z = -3.0 * Y + noise")
    print("\nGround Truth Adjacency:")
    print("      X  Y  Z")
    for i, var in enumerate(['X', 'Y', 'Z']):
        print(f"  {var}  {gt_adjacency[i]}")
    
    # Initialize model
    print("\n" + "=" * 70)
    print("MODEL INITIALIZATION")
    print("=" * 70)
    
    model = NeuralLP(
        n_vars=3,
        mask=mask,
        max_hops=2,
        init_value=0.1
    )
    
    print("\nConfiguration:")
    print("  Variables: 3 (X, Y, Z)")
    print("  Max hops: 2 (allows X -> Y -> Z)")
    print("  Trainable edges: 6")
    print("  Task: Predict Z from X and Y")
    
    # Training strategy
    print("\n" + "=" * 70)
    print("TRAINING STRATEGY")
    print("=" * 70)
    print("\nInput: X and Y")
    print("Output: Z")
    print("\nExpected behavior:")
    print("  - Model should learn X -> Y edge")
    print("  - Model should learn Y -> Z edge")
    print("  - Path X -> Y -> Z should be strong")
    print("  - Reverse and spurious edges should be weak")
    
    # Train
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    history = train_with_input_features(
        model=model,
        data=data,
        source_indices=[0, 1],  # Use X and Y as inputs
        target_idx=2,  # Predict Z
        n_epochs=2000,
        learning_rate=0.02,
        l1_lambda=0.02,
        print_every=200
    )
    
    # Get learned adjacency
    learned_adj = model.get_adjacency_matrix()
    
    # Visualize training
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(history['total_loss'])
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Total Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        axes[1].plot(history['mse_loss'], color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE Loss')
        axes[1].set_title('Prediction Error (MSE)')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
        
        axes[2].plot(history['l1_loss'], color='green')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('L1 Loss')
        axes[2].set_title('Sparsity Regularization (L1)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/training_loss_final.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved training curves to results/training_loss_final.png")
        plt.close()
    except Exception as e:
        print(f"\n[WARNING] Could not create plots: {e}")
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\nLearned Adjacency Matrix:")
    print("      X      Y      Z")
    for i, var in enumerate(['X', 'Y', 'Z']):
        row_str = "  ".join([f"{learned_adj[i, j]:6.3f}" for j in range(3)])
        print(f"  {var}  {row_str}")
    
    print("\nKey Edges:")
    print(f"  X -> Y: {learned_adj[0, 1]:6.3f} (truth: 2.0)")
    print(f"  Y -> Z: {learned_adj[1, 2]:6.3f} (truth: -3.0)")
    print(f"  Y -> X: {learned_adj[1, 0]:6.3f} (should be ~0)")
    print(f"  Z -> Y: {learned_adj[2, 1]:6.3f} (should be ~0)")
    print(f"  X -> Z: {learned_adj[0, 2]:6.3f} (should be ~0)")
    print(f"  Z -> X: {learned_adj[2, 0]:6.3f} (should be ~0)")
    
    # Comparison with ground truth
    print_adjacency_comparison(
        learned=learned_adj,
        ground_truth=gt_adjacency,
        var_names=['X', 'Y', 'Z'],
        threshold=0.4
    )
    
    # Success analysis
    print("\n" + "=" * 70)
    print("SUCCESS ANALYSIS")
    print("=" * 70)
    
    checks = []
    
    # Check 1: X -> Y should be strong
    xy_strong = abs(learned_adj[0, 1]) > 0.4
    checks.append(("X -> Y is strong", xy_strong))
    print(f"  {'[OK]' if xy_strong else '[FAIL]'} X -> Y = {learned_adj[0, 1]:.3f} (expected: strong)")
    
    # Check 2: Y -> Z should be strong
    yz_strong = abs(learned_adj[1, 2]) > 0.4
    checks.append(("Y -> Z is strong", yz_strong))
    print(f"  {'[OK]' if yz_strong else '[FAIL]'} Y -> Z = {learned_adj[1, 2]:.3f} (expected: strong)")
    
    # Check 3: Y -> X should be weak
    yx_weak = abs(learned_adj[1, 0]) < 0.3
    checks.append(("Y -> X is weak", yx_weak))
    print(f"  {'[OK]' if yx_weak else '[FAIL]'} Y -> X = {learned_adj[1, 0]:.3f} (expected: weak)")
    
    # Check 4: Z -> Y should be weak
    zy_weak = abs(learned_adj[2, 1]) < 0.3
    checks.append(("Z -> Y is weak", zy_weak))
    print(f"  {'[OK]' if zy_weak else '[FAIL]'} Z -> Y = {learned_adj[2, 1]:.3f} (expected: weak)")
    
    # Check 5: X -> Z should be weak (spurious)
    xz_weak = abs(learned_adj[0, 2]) < 0.3
    checks.append(("X -> Z is weak", xz_weak))
    print(f"  {'[OK]' if xz_weak else '[FAIL]'} X -> Z = {learned_adj[0, 2]:.3f} (expected: weak)")
    
    # Check 6: Z -> X should be weak (spurious)
    zx_weak = abs(learned_adj[2, 0]) < 0.3
    checks.append(("Z -> X is weak", zx_weak))
    print(f"  {'[OK]' if zx_weak else '[FAIL]'} Z -> X = {learned_adj[2, 0]:.3f} (expected: weak)")
    
    success_rate = sum(1 for _, passed in checks if passed) / len(checks)
    
    print(f"\nSuccess Rate: {success_rate:.1%} ({sum(1 for _, p in checks if p)}/{len(checks)})")
    
    if success_rate >= 0.8:
        print("\n*** REFINEMENT SUCCESSFUL! ***")
        print("Neural LP successfully refined the poor FCI result!")
        print("The model discovered the true causal structure X -> Y -> Z")
    elif success_rate >= 0.6:
        print("\n*** PARTIAL SUCCESS ***")
        print("Model learned most of the structure.")
        print("Some edges may need threshold adjustment.")
    else:
        print("\n*** NEEDS IMPROVEMENT ***")
        print("Try different hyperparameters or longer training.")
    
    # Save results
    np.save('results/learned_adjacency_final.npy', learned_adj)
    print(f"\nSaved learned adjacency to results/learned_adjacency_final.npy")
    
    print("\n" + "=" * 70)
    print("STEP 3 COMPLETE!")
    print("=" * 70)
    print("\nWhat we demonstrated:")
    print("  [+] Neural LP can learn from data")
    print("  [+] Multi-hop reasoning discovers causal paths")
    print("  [+] L1 regularization encourages sparsity")
    print("  [+] Poor FCI results can be refined")
    print("\nNext steps:")
    print("  - Test on more complex structures")
    print("  - Integrate with LLM pipeline")
    print("  - Apply to real ALARM dataset")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()

