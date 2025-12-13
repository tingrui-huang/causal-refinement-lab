"""
Step 4: Final Refinement with Two-Phase Training

Key insight from v1 and v2: Strong regularization suppresses ALL weights!

Solution: Two-phase training approach
1. Phase 1: Learn strong weights with MINIMAL regularization
   - Focus on prediction accuracy
   - Allow model to learn true causal effects (X->Y, Y->Z)
   
2. Phase 2: Apply DAG + sparsity constraints to prune weak edges
   - Now that strong edges are established, regularization only affects weak edges
   - Removes spurious and reverse edges while preserving true edges

This mimics the principle: "First learn the signal, then remove the noise"
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from neural_lp import NeuralLP
from typing import Dict


def compute_dag_constraint(W: torch.Tensor) -> torch.Tensor:
    """Compute NOTEARS DAG constraint: tr(e^{W o W}) - d"""
    n_vars = W.shape[0]
    W_squared = W * W
    M = W_squared
    expm = torch.eye(n_vars, device=W.device)
    M_power = torch.eye(n_vars, device=W.device)
    factorial = 1.0
    for i in range(1, 7):
        factorial *= i
        M_power = torch.matmul(M_power, M)
        expm = expm + M_power / factorial
    h = torch.trace(expm) - n_vars
    return h


def train_phase1_learn_signal(
    model: NeuralLP,
    data: torch.Tensor,
    target_idx: int,
    n_epochs: int = 1500,
    learning_rate: float = 0.02,
    l1_lambda: float = 0.001,  # MINIMAL regularization
    print_every: int = 300
) -> Dict:
    """
    Phase 1: Learn true causal effects with minimal regularization.
    
    Goal: Establish strong weights for X->Y and Y->Z paths.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    targets = data[:, target_idx]
    
    history = {'total_loss': [], 'mse_loss': [], 'l1_loss': []}
    
    print("\n" + "=" * 70)
    print("PHASE 1: LEARNING SIGNAL (Minimal Regularization)")
    print("=" * 70)
    print(f"Goal: Establish strong causal paths X->Y->Z")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"L1 lambda: {l1_lambda} (VERY WEAK - focus on prediction)")
    print("-" * 70)
    
    for epoch in range(n_epochs):
        predictions = model.predict_target(data, target_idx)
        adjacency = model.adjacency()
        
        # MSE + minimal L1 (just to prevent explosion)
        mse_loss = nn.functional.mse_loss(predictions, targets)
        l1_loss = torch.sum(torch.abs(adjacency))
        total_loss = mse_loss + l1_lambda * l1_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        history['total_loss'].append(total_loss.item())
        history['mse_loss'].append(mse_loss.item())
        history['l1_loss'].append(l1_loss.item())
        
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1:4d} | "
                  f"Loss: {total_loss.item():.4f} | "
                  f"MSE: {mse_loss.item():.4f} | "
                  f"L1: {l1_loss.item():.4f}")
    
    print("-" * 70)
    print("Phase 1 complete!")
    return history


def train_phase2_prune_noise(
    model: NeuralLP,
    data: torch.Tensor,
    target_idx: int,
    n_epochs: int = 2000,
    learning_rate: float = 0.005,
    l1_lambda: float = 0.08,
    dag_lambda: float = 2.0,
    print_every: int = 400
) -> Dict:
    """
    Phase 2: Apply constraints to prune weak/spurious edges.
    
    Goal: Remove Y->X (cycle) and X->Z (shortcut) while preserving X->Y, Y->Z.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    targets = data[:, target_idx]
    
    history = {'total_loss': [], 'mse_loss': [], 'l1_loss': [], 'dag_loss': []}
    
    print("\n" + "=" * 70)
    print("PHASE 2: PRUNING NOISE (Strong Regularization)")
    print("=" * 70)
    print(f"Goal: Remove spurious edges while preserving true paths")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate} (lower for stability)")
    print(f"L1 lambda: {l1_lambda} (prune weak edges)")
    print(f"DAG lambda: {dag_lambda} (break cycles)")
    print("-" * 70)
    
    for epoch in range(n_epochs):
        predictions = model.predict_target(data, target_idx)
        adjacency = model.adjacency()
        
        # MSE + strong L1 + DAG constraint
        mse_loss = nn.functional.mse_loss(predictions, targets)
        l1_loss = torch.sum(torch.abs(adjacency))
        dag_constraint = compute_dag_constraint(adjacency)
        total_loss = mse_loss + l1_lambda * l1_loss + dag_lambda * dag_constraint
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        history['total_loss'].append(total_loss.item())
        history['mse_loss'].append(mse_loss.item())
        history['l1_loss'].append(l1_loss.item())
        history['dag_loss'].append(dag_constraint.item())
        
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1:4d} | "
                  f"Loss: {total_loss.item():.4f} | "
                  f"MSE: {mse_loss.item():.4f} | "
                  f"L1: {l1_loss.item():.4f} | "
                  f"DAG: {dag_constraint.item():.4f}")
    
    print("-" * 70)
    print("Phase 2 complete!")
    return history


def visualize_two_phase_training(
    phase1_history: Dict,
    phase2_history: Dict,
    output_path: str = 'results/training_two_phase.png'
):
    """Visualize two-phase training progress."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    n_phase1 = len(phase1_history['total_loss'])
    n_phase2 = len(phase2_history['total_loss'])
    epochs_phase1 = np.arange(n_phase1)
    epochs_phase2 = np.arange(n_phase1, n_phase1 + n_phase2)
    
    # Total loss
    axes[0, 0].plot(epochs_phase1, phase1_history['total_loss'], 
                    label='Phase 1', color='blue', alpha=0.7)
    axes[0, 0].plot(epochs_phase2, phase2_history['total_loss'], 
                    label='Phase 2', color='red', alpha=0.7)
    axes[0, 0].axvline(n_phase1, color='black', linestyle='--', alpha=0.3, label='Phase Boundary')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss (Two-Phase Training)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MSE loss
    axes[0, 1].plot(epochs_phase1, phase1_history['mse_loss'], 
                    label='Phase 1', color='blue', alpha=0.7)
    axes[0, 1].plot(epochs_phase2, phase2_history['mse_loss'], 
                    label='Phase 2', color='red', alpha=0.7)
    axes[0, 1].axvline(n_phase1, color='black', linestyle='--', alpha=0.3)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].set_title('Prediction Error (MSE)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # L1 loss
    axes[1, 0].plot(epochs_phase1, phase1_history['l1_loss'], 
                    label='Phase 1', color='blue', alpha=0.7)
    axes[1, 0].plot(epochs_phase2, phase2_history['l1_loss'], 
                    label='Phase 2', color='red', alpha=0.7)
    axes[1, 0].axvline(n_phase1, color='black', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('L1 Loss')
    axes[1, 0].set_title('Sparsity (L1)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # DAG constraint (Phase 2 only)
    axes[1, 1].plot(epochs_phase2, phase2_history['dag_loss'], 
                    label='Phase 2 only', color='red', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('DAG Constraint')
    axes[1, 1].set_title('Acyclicity (NOTEARS) - Phase 2')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved two-phase training curves to {output_path}")
    plt.close()


def analyze_final_results(
    learned_adj: np.ndarray,
    gt_adj: np.ndarray,
    gt_weights: np.ndarray,
    var_names: list = ['X', 'Y', 'Z']
):
    """Comprehensive analysis of final results."""
    print("\n" + "=" * 70)
    print("FINAL RESULTS ANALYSIS")
    print("=" * 70)
    
    print("\nLearned Adjacency Matrix:")
    print("      " + "  ".join([f"{v:>8}" for v in var_names]))
    for i, var in enumerate(var_names):
        row_str = "  ".join([f"{learned_adj[i, j]:8.4f}" for j in range(len(var_names))])
        print(f"  {var}   {row_str}")
    
    print("\nGround Truth Weights:")
    print("      " + "  ".join([f"{v:>8}" for v in var_names]))
    for i, var in enumerate(var_names):
        row_str = "  ".join([f"{gt_weights[i, j]:8.1f}" for j in range(len(var_names))])
        print(f"  {var}   {row_str}")
    
    print("\n" + "=" * 70)
    print("EDGE-BY-EDGE ANALYSIS")
    print("=" * 70)
    
    checks = []
    
    # True edges
    print("\nTRUE EDGES (should be strong):")
    x_y = learned_adj[0, 1]
    print(f"  X -> Y: {x_y:7.4f} (ground truth: {gt_weights[0, 1]:5.1f})")
    if x_y > 1.0:
        checks.append(("X -> Y strong", True))
    else:
        checks.append(("X -> Y strong", False))
    
    y_z = learned_adj[1, 2]
    print(f"  Y -> Z: {y_z:7.4f} (ground truth: {gt_weights[1, 2]:5.1f})")
    if y_z < -1.0:
        checks.append(("Y -> Z strong", True))
    else:
        checks.append(("Y -> Z strong", False))
    
    # Reverse edges (should be weak)
    print("\nREVERSE EDGES (should be ~0, DAG constraint):")
    y_x = learned_adj[1, 0]
    print(f"  Y -> X: {y_x:7.4f} (ground truth: {gt_weights[1, 0]:5.1f})")
    if abs(y_x) < 0.3:
        checks.append(("Y -> X weak", True))
    else:
        checks.append(("Y -> X weak", False))
    
    z_y = learned_adj[2, 1]
    print(f"  Z -> Y: {z_y:7.4f} (ground truth: {gt_weights[2, 1]:5.1f})")
    if abs(z_y) < 0.3:
        checks.append(("Z -> Y weak", True))
    else:
        checks.append(("Z -> Y weak", False))
    
    # Spurious edges (should be weak)
    print("\nSPURIOUS EDGES (should be ~0, sparsity constraint):")
    x_z = learned_adj[0, 2]
    print(f"  X -> Z: {x_z:7.4f} (ground truth: {gt_weights[0, 2]:5.1f})")
    if abs(x_z) < 0.3:
        checks.append(("X -> Z weak", True))
    else:
        checks.append(("X -> Z weak", False))
    
    z_x = learned_adj[2, 0]
    print(f"  Z -> X: {z_x:7.4f} (ground truth: {gt_weights[2, 0]:5.1f})")
    if abs(z_x) < 0.3:
        checks.append(("Z -> X weak", True))
    else:
        checks.append(("Z -> X weak", False))
    
    # Overall success
    print("\n" + "=" * 70)
    print("SUCCESS CHECKLIST")
    print("=" * 70)
    for i, (check_name, passed) in enumerate(checks, 1):
        status = "[OK]" if passed else "[X]"
        print(f"{i}. {status} {check_name}")
    
    success_rate = sum(1 for _, passed in checks if passed) / len(checks)
    print(f"\nOverall Success Rate: {success_rate:.1%}")
    
    if success_rate >= 0.83:  # 5/6
        print("\n*** SUCCESS! ***")
        print("Two-phase training successfully learned the causal structure!")
        print("The model distinguished correlation from causation.")
    elif success_rate >= 0.67:  # 4/6
        print("\n*** GOOD PROGRESS! ***")
        print("Two-phase training made significant improvements.")
    else:
        print("\n*** NEEDS MORE TUNING ***")


def main():
    print("=" * 70)
    print("STEP 4: FINAL TWO-PHASE REFINEMENT")
    print("=" * 70)
    
    print("\nStrategy: 'First learn the signal, then remove the noise'")
    print("\nPhase 1: Learn strong causal effects (minimal regularization)")
    print("         -> Establish X->Y and Y->Z with correct magnitudes")
    print("\nPhase 2: Prune weak edges (strong regularization)")
    print("         -> Remove Y->X (reverse) and X->Z (spurious)")
    print("         -> Preserve X->Y and Y->Z (already strong)")
    
    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    data_df = pd.read_csv('data/ground_truth_data.csv')
    data = torch.FloatTensor(data_df.values)
    mask = np.load('data/mask_matrix.npy')
    gt_adjacency = np.load('data/ground_truth_adjacency.npy')
    gt_weights = np.load('data/ground_truth_weights.npy')
    
    print(f"Data shape: {data.shape}")
    print(f"Ground truth: X -> Y (coef=2.0), Y -> Z (coef=-3.0)")
    
    # Initialize model
    n_vars = 3
    max_hops = 2
    model = NeuralLP(n_vars=n_vars, mask=mask, max_hops=max_hops, init_value=0.1)
    target_idx = 2  # Predict Z
    
    # Phase 1: Learn signal
    phase1_history = train_phase1_learn_signal(
        model=model,
        data=data,
        target_idx=target_idx,
        n_epochs=1500,
        learning_rate=0.02,
        l1_lambda=0.001,
        print_every=300
    )
    
    print("\nPhase 1 learned weights:")
    phase1_adj = model.get_adjacency_matrix()
    print("      " + "  ".join([f"{v:>8}" for v in ['X', 'Y', 'Z']]))
    for i, var in enumerate(['X', 'Y', 'Z']):
        row_str = "  ".join([f"{phase1_adj[i, j]:8.4f}" for j in range(3)])
        print(f"  {var}   {row_str}")
    
    # Phase 2: Prune noise
    phase2_history = train_phase2_prune_noise(
        model=model,
        data=data,
        target_idx=target_idx,
        n_epochs=2000,
        learning_rate=0.005,
        l1_lambda=0.08,
        dag_lambda=2.0,
        print_every=400
    )
    
    # Get final results
    learned_adj = model.get_adjacency_matrix()
    
    # Visualize
    try:
        visualize_two_phase_training(phase1_history, phase2_history)
    except Exception as e:
        print(f"\n[WARNING] Could not create plots: {e}")
    
    # Analyze
    analyze_final_results(learned_adj, gt_adjacency, gt_weights)
    
    # Save
    np.save('results/learned_adjacency_final.npy', learned_adj)
    print(f"\nSaved final adjacency to results/learned_adjacency_final.npy")
    
    print("\n" + "=" * 70)
    print("TWO-PHASE TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
