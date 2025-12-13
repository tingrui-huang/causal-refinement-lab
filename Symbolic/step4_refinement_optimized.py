"""
Step 4: Optimized Refinement - Multi-Target Training

Key insight: The model only predicts Z, so it learns Y->Z strongly but not X->Y!

Solution: Train on MULTIPLE targets simultaneously
1. Predict Y from X (learns X->Y)
2. Predict Z from X and Y (learns Y->Z, validates X->Y->Z path)

This forces the model to learn the ENTIRE causal chain, not just the last edge.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from neural_lp import NeuralLP
from typing import Dict, Tuple


def compute_dag_constraint(W: torch.Tensor) -> torch.Tensor:
    """Compute NOTEARS DAG constraint."""
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


def compute_multi_target_loss(
    model: NeuralLP,
    data: torch.Tensor,
    l1_lambda: float = 0.0,
    dag_lambda: float = 0.0
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute loss for multiple prediction targets.
    
    Targets:
    - Y (index 1): Predict from X
    - Z (index 2): Predict from X and Y
    
    This ensures the entire causal chain X->Y->Z is learned.
    """
    adjacency = model.adjacency()
    
    # Predict Y (target 1) and Z (target 2)
    pred_Y = model.predict_target(data, target_idx=1)
    pred_Z = model.predict_target(data, target_idx=2)
    
    true_Y = data[:, 1]
    true_Z = data[:, 2]
    
    # MSE for both targets
    mse_Y = nn.functional.mse_loss(pred_Y, true_Y)
    mse_Z = nn.functional.mse_loss(pred_Z, true_Z)
    mse_loss = mse_Y + mse_Z
    
    # Regularization
    l1_loss = torch.sum(torch.abs(adjacency))
    dag_loss = compute_dag_constraint(adjacency)
    
    total_loss = mse_loss + l1_lambda * l1_loss + dag_lambda * dag_loss
    
    return total_loss, {
        'total': total_loss.item(),
        'mse_total': mse_loss.item(),
        'mse_Y': mse_Y.item(),
        'mse_Z': mse_Z.item(),
        'l1': l1_loss.item(),
        'dag': dag_loss.item()
    }


def train_phase1_multi_target(
    model: NeuralLP,
    data: torch.Tensor,
    n_epochs: int = 2000,
    learning_rate: float = 0.02,
    l1_lambda: float = 0.002,
    print_every: int = 400
) -> Dict:
    """Phase 1: Learn all edges by predicting multiple targets."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'total_loss': [], 
        'mse_Y': [], 
        'mse_Z': [], 
        'l1_loss': []
    }
    
    print("\n" + "=" * 70)
    print("PHASE 1: MULTI-TARGET LEARNING")
    print("=" * 70)
    print(f"Goal: Learn ENTIRE causal chain by predicting Y AND Z")
    print(f"  - Predict Y from X -> learns X->Y weight")
    print(f"  - Predict Z from X,Y -> learns Y->Z weight")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"L1 lambda: {l1_lambda} (minimal)")
    print("-" * 70)
    
    for epoch in range(n_epochs):
        loss, loss_dict = compute_multi_target_loss(
            model, data, l1_lambda=l1_lambda, dag_lambda=0.0
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        history['total_loss'].append(loss_dict['total'])
        history['mse_Y'].append(loss_dict['mse_Y'])
        history['mse_Z'].append(loss_dict['mse_Z'])
        history['l1_loss'].append(loss_dict['l1'])
        
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1:4d} | "
                  f"Loss: {loss_dict['total']:.4f} | "
                  f"MSE(Y): {loss_dict['mse_Y']:.4f} | "
                  f"MSE(Z): {loss_dict['mse_Z']:.4f} | "
                  f"L1: {loss_dict['l1']:.4f}")
    
    print("-" * 70)
    print("Phase 1 complete - Full causal chain established!")
    return history


def train_phase2_constrained(
    model: NeuralLP,
    data: torch.Tensor,
    n_epochs: int = 2000,
    learning_rate: float = 0.005,
    l1_lambda: float = 0.06,
    dag_lambda: float = 2.5,
    print_every: int = 400
) -> Dict:
    """Phase 2: Apply constraints to prune spurious edges."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'total_loss': [],
        'mse_Y': [],
        'mse_Z': [],
        'l1_loss': [],
        'dag_loss': []
    }
    
    print("\n" + "=" * 70)
    print("PHASE 2: CONSTRAINED REFINEMENT")
    print("=" * 70)
    print(f"Goal: Prune weak/spurious edges while preserving true edges")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"L1 lambda: {l1_lambda} (strong)")
    print(f"DAG lambda: {dag_lambda} (strong)")
    print("-" * 70)
    
    for epoch in range(n_epochs):
        loss, loss_dict = compute_multi_target_loss(
            model, data, l1_lambda=l1_lambda, dag_lambda=dag_lambda
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        history['total_loss'].append(loss_dict['total'])
        history['mse_Y'].append(loss_dict['mse_Y'])
        history['mse_Z'].append(loss_dict['mse_Z'])
        history['l1_loss'].append(loss_dict['l1'])
        history['dag_loss'].append(loss_dict['dag'])
        
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1:4d} | "
                  f"Loss: {loss_dict['total']:.4f} | "
                  f"MSE(Y): {loss_dict['mse_Y']:.4f} | "
                  f"MSE(Z): {loss_dict['mse_Z']:.4f} | "
                  f"L1: {loss_dict['l1']:.4f} | "
                  f"DAG: {loss_dict['dag']:.4f}")
    
    print("-" * 70)
    print("Phase 2 complete - Spurious edges pruned!")
    return history


def analyze_results_detailed(
    learned_adj: np.ndarray,
    gt_weights: np.ndarray,
    var_names: list = ['X', 'Y', 'Z']
):
    """Detailed analysis with weight accuracy."""
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    print("\nLearned vs Ground Truth Comparison:")
    print(f"{'Edge':<10} {'Learned':>10} {'Truth':>10} {'Error':>10} {'Status'}")
    print("-" * 55)
    
    checks = []
    
    # X -> Y (should be ~2.0)
    learned_xy = learned_adj[0, 1]
    truth_xy = gt_weights[0, 1]
    error_xy = abs(learned_xy - truth_xy)
    status_xy = "OK" if learned_xy > 1.0 else "WEAK"
    print(f"{'X -> Y':<10} {learned_xy:10.4f} {truth_xy:10.1f} {error_xy:10.4f} {status_xy}")
    checks.append(("X->Y strong (>1.0)", learned_xy > 1.0))
    
    # Y -> Z (should be ~-3.0)
    learned_yz = learned_adj[1, 2]
    truth_yz = gt_weights[1, 2]
    error_yz = abs(learned_yz - truth_yz)
    status_yz = "OK" if learned_yz < -1.0 else "WEAK"
    print(f"{'Y -> Z':<10} {learned_yz:10.4f} {truth_yz:10.1f} {error_yz:10.4f} {status_yz}")
    checks.append(("Y->Z strong (<-1.0)", learned_yz < -1.0))
    
    # Y -> X (should be ~0)
    learned_yx = learned_adj[1, 0]
    status_yx = "OK" if abs(learned_yx) < 0.3 else "TOO STRONG"
    print(f"{'Y -> X':<10} {learned_yx:10.4f} {0.0:10.1f} {abs(learned_yx):10.4f} {status_yx}")
    checks.append(("Y->X weak (<0.3)", abs(learned_yx) < 0.3))
    
    # Z -> Y (should be ~0)
    learned_zy = learned_adj[2, 1]
    status_zy = "OK" if abs(learned_zy) < 0.3 else "TOO STRONG"
    print(f"{'Z -> Y':<10} {learned_zy:10.4f} {0.0:10.1f} {abs(learned_zy):10.4f} {status_zy}")
    checks.append(("Z->Y weak (<0.3)", abs(learned_zy) < 0.3))
    
    # X -> Z (should be ~0)
    learned_xz = learned_adj[0, 2]
    status_xz = "OK" if abs(learned_xz) < 0.3 else "TOO STRONG"
    print(f"{'X -> Z':<10} {learned_xz:10.4f} {0.0:10.1f} {abs(learned_xz):10.4f} {status_xz}")
    checks.append(("X->Z weak (<0.3)", abs(learned_xz) < 0.3))
    
    # Z -> X (should be ~0)
    learned_zx = learned_adj[2, 0]
    status_zx = "OK" if abs(learned_zx) < 0.3 else "TOO STRONG"
    print(f"{'Z -> X':<10} {learned_zx:10.4f} {0.0:10.1f} {abs(learned_zx):10.4f} {status_zx}")
    checks.append(("Z->X weak (<0.3)", abs(learned_zx) < 0.3))
    
    print("\n" + "=" * 70)
    print("SUCCESS EVALUATION")
    print("=" * 70)
    
    for i, (check_name, passed) in enumerate(checks, 1):
        status = "[OK]" if passed else "[X]"
        print(f"{i}. {status} {check_name}")
    
    success_rate = sum(1 for _, passed in checks if passed) / len(checks)
    print(f"\nOverall Success Rate: {success_rate:.1%} ({sum(1 for _, p in checks if p)}/{len(checks)} checks passed)")
    
    if success_rate == 1.0:
        print("\n*** PERFECT! ***")
        print("All edges learned correctly!")
    elif success_rate >= 0.83:
        print("\n*** EXCELLENT! ***")
        print("Successfully distinguished causation from correlation!")
    elif success_rate >= 0.67:
        print("\n*** GOOD! ***")
        print("Major progress, minor tuning needed.")
    else:
        print("\n*** NEEDS WORK ***")


def main():
    print("=" * 70)
    print("STEP 4: OPTIMIZED MULTI-TARGET REFINEMENT")
    print("=" * 70)
    
    print("\nKey Innovation: Train on MULTIPLE targets")
    print("  Problem: Single-target (Z) doesn't force learning X->Y")
    print("  Solution: Multi-target (Y and Z) forces learning full chain")
    print("\nPhase 1: Learn full chain X->Y->Z (weak regularization)")
    print("Phase 2: Prune spurious edges (strong constraints)")
    
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
    
    # Phase 1: Multi-target learning
    phase1_history = train_phase1_multi_target(
        model=model,
        data=data,
        n_epochs=2000,
        learning_rate=0.02,
        l1_lambda=0.002,
        print_every=400
    )
    
    print("\nPhase 1 results:")
    phase1_adj = model.get_adjacency_matrix()
    print("      " + "  ".join([f"{v:>8}" for v in ['X', 'Y', 'Z']]))
    for i, var in enumerate(['X', 'Y', 'Z']):
        row_str = "  ".join([f"{phase1_adj[i, j]:8.4f}" for j in range(3)])
        print(f"  {var}   {row_str}")
    
    # Phase 2: Constrained refinement
    phase2_history = train_phase2_constrained(
        model=model,
        data=data,
        n_epochs=2000,
        learning_rate=0.005,
        l1_lambda=0.06,
        dag_lambda=2.5,
        print_every=400
    )
    
    # Final results
    learned_adj = model.get_adjacency_matrix()
    analyze_results_detailed(learned_adj, gt_weights)
    
    # Save
    np.save('results/learned_adjacency_optimized.npy', learned_adj)
    print(f"\nSaved optimized adjacency to results/learned_adjacency_optimized.npy")
    
    print("\n" + "=" * 70)
    print("OPTIMIZED MULTI-TARGET TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
