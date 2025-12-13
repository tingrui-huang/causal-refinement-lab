"""
Step 4: Refinement with DAG Constraints and Stronger Sparsity (Version 2)

This version uses more aggressive hyperparameters based on initial results:
- Increased DAG lambda: 1.0 -> 3.0 (stronger acyclicity)
- Increased L1 lambda: 0.05 -> 0.1 (stronger sparsity)
- More epochs: 2000 -> 3000 (better convergence)
- Adaptive learning rate with decay

Expected outcome: All weights should converge to correct values.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from neural_lp import NeuralLP, MaskedAdjacencyMatrix
from typing import Tuple, Dict


def compute_dag_constraint(W: torch.Tensor) -> torch.Tensor:
    """
    Compute NOTEARS DAG constraint: tr(e^{W o W}) - d
    
    This constraint is zero if and only if the graph is acyclic.
    The exponential ensures the constraint is differentiable.
    
    Args:
        W: Adjacency matrix [n_vars, n_vars]
    
    Returns:
        DAG constraint value (0 = acyclic, >0 = has cycles)
    """
    n_vars = W.shape[0]
    
    # Compute W o W (element-wise square)
    W_squared = W * W
    
    # Compute matrix exponential: e^{W o W}
    # Use Taylor series approximation for stability
    # exp(M) = I + M + M^2/2! + M^3/3! + ...
    M = W_squared
    expm = torch.eye(n_vars, device=W.device)
    
    # Add terms up to power 6 (sufficient for small graphs)
    M_power = torch.eye(n_vars, device=W.device)
    factorial = 1.0
    for i in range(1, 7):
        factorial *= i
        M_power = torch.matmul(M_power, M)
        expm = expm + M_power / factorial
    
    # Compute trace
    h = torch.trace(expm) - n_vars
    
    return h


def compute_refined_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    adjacency_matrix: torch.Tensor,
    l1_lambda: float = 0.1,
    dag_lambda: float = 3.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute refined loss with DAG constraint and stronger sparsity.
    
    Loss = MSE + l1_lambda * L1 + dag_lambda * DAG_constraint
    
    Args:
        predictions: Predicted values
        targets: True values
        adjacency_matrix: Current adjacency matrix
        l1_lambda: L1 regularization strength (increased to 0.1)
        dag_lambda: DAG constraint strength (increased to 3.0)
    
    Returns:
        Tuple of (total_loss, loss_dict)
    """
    # MSE loss for predictions
    mse_loss = nn.functional.mse_loss(predictions, targets)
    
    # L1 regularization for sparsity (MUCH STRONGER)
    l1_loss = torch.sum(torch.abs(adjacency_matrix))
    
    # DAG constraint to prevent cycles (MUCH STRONGER)
    dag_constraint = compute_dag_constraint(adjacency_matrix)
    
    # Total loss
    total_loss = mse_loss + l1_lambda * l1_loss + dag_lambda * dag_constraint
    
    loss_dict = {
        'total': total_loss.item(),
        'mse': mse_loss.item(),
        'l1': l1_loss.item(),
        'dag': dag_constraint.item()
    }
    
    return total_loss, loss_dict


def train_neural_lp_refined_v2(
    model: NeuralLP,
    data: torch.Tensor,
    target_idx: int,
    n_epochs: int = 3000,
    learning_rate: float = 0.01,
    l1_lambda: float = 0.1,
    dag_lambda: float = 3.0,
    lr_decay_steps: int = 1000,
    lr_decay_rate: float = 0.5,
    print_every: int = 300,
    verbose: bool = True
) -> Dict:
    """
    Train Neural LP with stronger constraints and adaptive learning rate.
    
    Args:
        model: NeuralLP model to train
        data: Training data [n_samples, n_vars]
        target_idx: Index of target variable to predict
        n_epochs: Number of training epochs (increased to 3000)
        learning_rate: Initial learning rate
        l1_lambda: L1 regularization strength (STRONGER: 0.1)
        dag_lambda: DAG constraint strength (STRONGER: 3.0)
        lr_decay_steps: Decay learning rate every N steps
        lr_decay_rate: Learning rate decay multiplier
        print_every: Print progress every N epochs
        verbose: Whether to print training progress
    
    Returns:
        Dictionary with training history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_decay_steps, gamma=lr_decay_rate
    )
    
    history = {
        'total_loss': [],
        'mse_loss': [],
        'l1_loss': [],
        'dag_loss': []
    }
    
    # Extract target values
    targets = data[:, target_idx]
    
    if verbose:
        print(f"\nTraining Neural LP with STRONGER constraints (v2)")
        print(f"Target variable: {target_idx}")
        print(f"Epochs: {n_epochs}")
        print(f"Learning rate: {learning_rate} (with decay)")
        print(f"L1 lambda (sparsity): {l1_lambda} [v1: 0.05, v0: 0.01]")
        print(f"DAG lambda (acyclicity): {dag_lambda} [v1: 1.0]")
        print("-" * 70)
    
    for epoch in range(n_epochs):
        # Forward pass
        predictions = model.predict_target(data, target_idx)
        
        # Compute refined loss with DAG constraint
        adjacency = model.adjacency()
        loss, loss_dict = compute_refined_loss(
            predictions, targets, adjacency, l1_lambda, dag_lambda
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        scheduler.step()
        
        # Record history
        history['total_loss'].append(loss_dict['total'])
        history['mse_loss'].append(loss_dict['mse'])
        history['l1_loss'].append(loss_dict['l1'])
        history['dag_loss'].append(loss_dict['dag'])
        
        # Print progress
        if verbose and (epoch + 1) % print_every == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:4d} | "
                  f"Loss: {loss_dict['total']:.4f} | "
                  f"MSE: {loss_dict['mse']:.4f} | "
                  f"L1: {loss_dict['l1']:.4f} | "
                  f"DAG: {loss_dict['dag']:.4f} | "
                  f"LR: {current_lr:.4f}")
    
    if verbose:
        print("-" * 70)
        print(f"Training completed!")
    
    return history


def visualize_refined_training(
    history: Dict, 
    output_path: str = 'results/training_loss_refined_v2.png'
):
    """Visualize training loss curves with DAG constraint."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Total loss
    axes[0, 0].plot(history['total_loss'])
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # MSE loss
    axes[0, 1].plot(history['mse_loss'], color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].set_title('Prediction Error (MSE)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # L1 loss
    axes[1, 0].plot(history['l1_loss'], color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('L1 Loss')
    axes[1, 0].set_title('Sparsity Regularization (L1)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # DAG constraint
    axes[1, 1].plot(history['dag_loss'], color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('DAG Constraint')
    axes[1, 1].set_title('Acyclicity Constraint (NOTEARS)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved refined training curves to {output_path}")
    plt.close()


def compare_all_versions(
    step3_adj: np.ndarray,
    step4v1_adj: np.ndarray,
    step4v2_adj: np.ndarray,
    gt_adj: np.ndarray,
    var_names: list = ['X', 'Y', 'Z']
):
    """
    Compare all versions: Step 3 vs Step 4 v1 vs Step 4 v2.
    
    Args:
        step3_adj: Adjacency matrix from Step 3 (no constraints)
        step4v1_adj: Adjacency matrix from Step 4 v1 (weak constraints)
        step4v2_adj: Adjacency matrix from Step 4 v2 (strong constraints)
        gt_adj: Ground truth adjacency matrix
        var_names: Variable names
    """
    print("\n" + "=" * 70)
    print("COMPARISON: STEP 3 vs STEP 4 v1 vs STEP 4 v2")
    print("=" * 70)
    
    print("\nStep 3 (No constraints):")
    print("      " + "  ".join([f"{v:>8}" for v in var_names]))
    for i, var in enumerate(var_names):
        row_str = "  ".join([f"{step3_adj[i, j]:8.4f}" for j in range(len(var_names))])
        print(f"  {var}   {row_str}")
    
    print("\nStep 4 v1 (L1=0.05, DAG=1.0):")
    print("      " + "  ".join([f"{v:>8}" for v in var_names]))
    for i, var in enumerate(var_names):
        row_str = "  ".join([f"{step4v1_adj[i, j]:8.4f}" for j in range(len(var_names))])
        print(f"  {var}   {row_str}")
    
    print("\nStep 4 v2 (L1=0.1, DAG=3.0):")
    print("      " + "  ".join([f"{v:>8}" for v in var_names]))
    for i, var in enumerate(var_names):
        row_str = "  ".join([f"{step4v2_adj[i, j]:8.4f}" for j in range(len(var_names))])
        print(f"  {var}   {row_str}")
    
    print("\nGround Truth:")
    print("      " + "  ".join([f"{v:>8}" for v in var_names]))
    for i, var in enumerate(var_names):
        row_str = "  ".join([f"{int(gt_adj[i, j]):8d}" for j in range(len(var_names))])
        print(f"  {var}   {row_str}")


def analyze_refinement_success(
    learned_adj: np.ndarray,
    gt_adj: np.ndarray,
    var_names: list = ['X', 'Y', 'Z'],
    version: str = "v2"
):
    """
    Analyze if refinement successfully fixed the issues.
    
    Args:
        learned_adj: Learned adjacency matrix
        gt_adj: Ground truth adjacency matrix
        var_names: Variable names
        version: Version string for display
    """
    print("\n" + "=" * 70)
    print(f"REFINEMENT SUCCESS ANALYSIS ({version})")
    print("=" * 70)
    
    checks = []
    
    # Check 1: X -> Y should be strong and positive
    x_y = learned_adj[0, 1]
    if x_y > 1.0:
        checks.append(("X -> Y is strong and positive (~2.0)", True, x_y))
    else:
        checks.append(("X -> Y is strong and positive (~2.0)", False, x_y))
    
    # Check 2: Y -> Z should be strong and negative
    y_z = learned_adj[1, 2]
    if y_z < -1.0:
        checks.append(("Y -> Z is strong and negative (~-3.0)", True, y_z))
    else:
        checks.append(("Y -> Z is strong and negative (~-3.0)", False, y_z))
    
    # Check 3: Y -> X should be weak (FIXED)
    y_x = learned_adj[1, 0]
    if abs(y_x) < 0.2:
        checks.append(("Y -> X is weak (cycle broken)", True, y_x))
    else:
        checks.append(("Y -> X is weak (cycle broken)", False, y_x))
    
    # Check 4: Z -> Y should be weak
    z_y = learned_adj[2, 1]
    if abs(z_y) < 0.2:
        checks.append(("Z -> Y is weak (cycle broken)", True, z_y))
    else:
        checks.append(("Z -> Y is weak (cycle broken)", False, z_y))
    
    # Check 5: X -> Z should be weak (FIXED)
    x_z = learned_adj[0, 2]
    if abs(x_z) < 0.2:
        checks.append(("X -> Z is weak (spurious edge removed)", True, x_z))
    else:
        checks.append(("X -> Z is weak (spurious edge removed)", False, x_z))
    
    # Check 6: Z -> X should be weak
    z_x = learned_adj[2, 0]
    if abs(z_x) < 0.2:
        checks.append(("Z -> X is weak (spurious edge removed)", True, z_x))
    else:
        checks.append(("Z -> X is weak (spurious edge removed)", False, z_x))
    
    print("\nChecklist:")
    for i, (check_name, passed, value) in enumerate(checks, 1):
        status = "[OK]" if passed else "[X]"
        print(f"{i}. {status} {check_name}")
        print(f"         Actual value: {value:.4f}")
    
    success_rate = sum(1 for _, passed, _ in checks if passed) / len(checks)
    print(f"\n{'=' * 70}")
    print(f"Overall Success Rate: {success_rate:.1%}")
    
    improvement_threshold = 0.83  # 5/6 checks
    if success_rate >= improvement_threshold:
        print("\n*** REFINEMENT SUCCESSFUL! ***")
        print("DAG constraint + stronger sparsity fixed the issues!")
        print("- Reverse edges (Y->X) suppressed")
        print("- Spurious edges (X->Z) suppressed")
        print("- True causal structure recovered")
    else:
        print("\n*** REFINEMENT PARTIALLY SUCCESSFUL ***")
        print("Some issues remain. Consider:")
        print("- Increasing dag_lambda further (e.g., 5.0)")
        print("- Increasing l1_lambda further (e.g., 0.15)")
        print("- Training for more epochs (e.g., 5000)")
        print("- Adjusting thresholds for success criteria")


def main():
    print("=" * 70)
    print("STEP 4 v2: REFINEMENT WITH STRONGER CONSTRAINTS")
    print("=" * 70)
    
    print("\nImproving on v1 (which achieved 16.7% success):")
    print("Changes from v1 to v2:")
    print("  - L1 lambda: 0.05 -> 0.1 (2x stronger sparsity)")
    print("  - DAG lambda: 1.0 -> 3.0 (3x stronger acyclicity)")
    print("  - Epochs: 2000 -> 3000 (better convergence)")
    print("  - Added learning rate decay for stability")
    
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
    print(f"Trainable edges: {int(mask.sum())}")
    
    # Load previous results for comparison
    try:
        step3_adjacency = np.load('results/learned_adjacency.npy')
        step4v1_adjacency = np.load('results/learned_adjacency_refined.npy')
        have_previous = True
        print("Loaded Step 3 and Step 4 v1 results for comparison")
    except FileNotFoundError:
        step3_adjacency = None
        step4v1_adjacency = None
        have_previous = False
        print("Previous results not found (will skip comparison)")
    
    # Initialize model
    print("\n" + "=" * 70)
    print("INITIALIZING MODEL")
    print("=" * 70)
    
    n_vars = 3
    max_hops = 2
    
    model = NeuralLP(
        n_vars=n_vars,
        mask=mask,
        max_hops=max_hops,
        init_value=0.1
    )
    
    print(f"Model: Neural LP with STRONGER DAG constraints")
    print(f"Variables: {n_vars} (X, Y, Z)")
    print(f"Max hops: {max_hops}")
    
    # Train model with stronger refinements
    print("\n" + "=" * 70)
    print("TRAINING WITH STRONGER REFINEMENTS")
    print("=" * 70)
    
    target_idx = 2  # Predict Z
    
    history = train_neural_lp_refined_v2(
        model=model,
        data=data,
        target_idx=target_idx,
        n_epochs=3000,
        learning_rate=0.01,
        l1_lambda=0.1,    # STRONGER than v1 (was 0.05)
        dag_lambda=3.0,   # STRONGER than v1 (was 1.0)
        lr_decay_steps=1000,
        lr_decay_rate=0.5,
        print_every=300,
        verbose=True
    )
    
    # Get learned adjacency
    learned_adj = model.get_adjacency_matrix()
    
    # Visualize training
    try:
        visualize_refined_training(history)
    except Exception as e:
        print(f"\n[WARNING] Could not create training plots: {e}")
    
    # Compare all versions if available
    if have_previous:
        compare_all_versions(
            step3_adjacency, step4v1_adjacency, learned_adj, gt_adjacency
        )
    
    # Analyze success
    analyze_refinement_success(learned_adj, gt_adjacency, version="v2")
    
    # Save results
    np.save('results/learned_adjacency_refined_v2.npy', learned_adj)
    print(f"\nSaved refined adjacency (v2) to results/learned_adjacency_refined_v2.npy")
    
    print("\n" + "=" * 70)
    print("STEP 4 v2 COMPLETE!")
    print("=" * 70)
    
    print("\nExpected Improvements over v1:")
    print("  Y -> X weight: should be closer to 0 (stronger DAG)")
    print("  X -> Z weight: should be closer to 0 (stronger sparsity)")
    print("  X -> Y weight: should remain strong")
    print("  Y -> Z weight: should remain strong")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()
