"""
Step 5: Neural LP Training on ALARM Network

This script applies the two-phase training approach (from Step 4) to the ALARM network.

Pipeline:
1. Load or generate ALARM observational data (37 variables)
2. Load the hybrid FCI-LLM mask matrix
3. Train Neural LP with two-phase approach:
   - Phase 1: Learn signal (100 epochs, minimal regularization)
   - Phase 2: Prune noise (100 epochs, DAG + strong L1)
4. Analyze learned structure:
   - Top-10 strongest edges
   - Compare with initial mask (which edges were pruned)
   - Orientation of bidirectional edges

Note: For now, we generate synthetic ALARM data. Replace with real data when available.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path to import neural_lp
sys.path.append(str(Path(__file__).parent))

from neural_lp import NeuralLP


def generate_synthetic_alarm_data(n_samples=1000, random_seed=42):
    """
    Generate synthetic ALARM network data.
    
    This is a simplified version for testing. In production, use real ALARM data.
    We generate data with some causal structure to test the training pipeline.
    
    Args:
        n_samples: Number of samples to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with 36 variables (ALARM network)
    """
    print("=" * 70)
    print("GENERATING SYNTHETIC ALARM DATA")
    print("=" * 70)
    print("\n[WARNING] Using synthetic data for testing.")
    print("Replace with real ALARM observational data for actual experiments.\n")
    
    np.random.seed(random_seed)
    
    # Load variable names
    var_file = Path('data/alarm_variables_37.txt')
    with open(var_file, 'r') as f:
        var_names = [line.strip().split('\t')[1] for line in f if line.strip()]
    
    n_vars = len(var_names)
    
    print(f"Variables: {n_vars}")
    print(f"Samples: {n_samples}")
    
    # Initialize data matrix
    data = np.zeros((n_samples, n_vars))
    
    # Generate data with some dependencies (simplified ALARM structure)
    # This is just for testing - real ALARM has complex conditional dependencies
    
    # Root causes (exogenous variables)
    root_vars = ['HYPOVOLEMIA', 'LVFAILURE', 'ANAPHYLAXIS', 'INTUBATION', 
                 'PULMEMBOLUS', 'MINVOLSET', 'FIO2', 'VENTTUBE']
    
    for var in root_vars:
        if var in var_names:
            idx = var_names.index(var)
            data[:, idx] = np.random.randn(n_samples) * 0.5
    
    # Generate other variables with some dependencies
    # Note: This is highly simplified - real ALARM has specific causal mechanisms
    
    # Cardiovascular chain (simplified)
    if 'HYPOVOLEMIA' in var_names and 'STROKEVOLUME' in var_names:
        hypo_idx = var_names.index('HYPOVOLEMIA')
        stroke_idx = var_names.index('STROKEVOLUME')
        data[:, stroke_idx] = -0.8 * data[:, hypo_idx] + np.random.randn(n_samples) * 0.3
    
    if 'STROKEVOLUME' in var_names and 'CO' in var_names:
        stroke_idx = var_names.index('STROKEVOLUME')
        co_idx = var_names.index('CO')
        data[:, co_idx] = 1.2 * data[:, stroke_idx] + np.random.randn(n_samples) * 0.3
    
    if 'CO' in var_names and 'BP' in var_names:
        co_idx = var_names.index('CO')
        bp_idx = var_names.index('BP')
        data[:, bp_idx] = 0.9 * data[:, co_idx] + np.random.randn(n_samples) * 0.3
    
    # HR pathway
    if 'HR' in var_names:
        hr_idx = var_names.index('HR')
        data[:, hr_idx] = np.random.randn(n_samples) * 0.5
        
        if 'CO' in var_names:
            co_idx = var_names.index('CO')
            data[:, co_idx] += 0.7 * data[:, hr_idx]
    
    # Respiratory chain (simplified)
    if 'INTUBATION' in var_names and 'VENTLUNG' in var_names:
        intub_idx = var_names.index('INTUBATION')
        vent_idx = var_names.index('VENTLUNG')
        data[:, vent_idx] = 0.8 * data[:, intub_idx] + np.random.randn(n_samples) * 0.3
    
    # Fill remaining variables with noise + small dependencies
    for i in range(n_vars):
        if np.all(data[:, i] == 0):
            data[:, i] = np.random.randn(n_samples) * 0.5
    
    # Standardize
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    
    df = pd.DataFrame(data, columns=var_names)
    
    print(f"\nGenerated data shape: {df.shape}")
    print(f"Sample statistics:")
    print(df.describe())
    
    return df


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


def train_phase1_signal(
    model: NeuralLP,
    data: torch.Tensor,
    target_idx: int,
    n_epochs: int = 100,
    learning_rate: float = 0.02,
    l1_lambda: float = 0.001,
    print_every: int = 20
):
    """Phase 1: Learn signal with minimal regularization."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    targets = data[:, target_idx]
    
    history = {'total_loss': [], 'mse_loss': [], 'l1_loss': []}
    
    print("\n" + "=" * 70)
    print("PHASE 1: LEARNING SIGNAL")
    print("=" * 70)
    print(f"Goal: Establish strong causal paths")
    print(f"Target variable: {target_idx}")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"L1 lambda: {l1_lambda} (minimal)")
    print("-" * 70)
    
    for epoch in range(n_epochs):
        predictions = model.predict_target(data, target_idx)
        adjacency = model.adjacency()
        
        # MSE + minimal L1
        mse_loss = torch.nn.functional.mse_loss(predictions, targets)
        l1_loss = torch.sum(torch.abs(adjacency))
        total_loss = mse_loss + l1_lambda * l1_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        history['total_loss'].append(total_loss.item())
        history['mse_loss'].append(mse_loss.item())
        history['l1_loss'].append(l1_loss.item())
        
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {total_loss.item():.4f} | "
                  f"MSE: {mse_loss.item():.4f} | L1: {l1_loss.item():.4f}")
    
    print("-" * 70)
    print("Phase 1 complete!")
    return history


def train_phase2_prune(
    model: NeuralLP,
    data: torch.Tensor,
    target_idx: int,
    n_epochs: int = 100,
    learning_rate: float = 0.005,
    l1_lambda: float = 0.08,
    dag_lambda: float = 2.0,
    print_every: int = 20
):
    """Phase 2: Prune noise with DAG constraints and strong sparsity."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    targets = data[:, target_idx]
    
    history = {'total_loss': [], 'mse_loss': [], 'l1_loss': [], 'dag_loss': []}
    
    print("\n" + "=" * 70)
    print("PHASE 2: PRUNING NOISE")
    print("=" * 70)
    print(f"Goal: Remove spurious/reverse edges while preserving true paths")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate} (lower for stability)")
    print(f"L1 lambda: {l1_lambda} (strong sparsity)")
    print(f"DAG lambda: {dag_lambda} (acyclicity constraint)")
    print("-" * 70)
    
    for epoch in range(n_epochs):
        predictions = model.predict_target(data, target_idx)
        adjacency = model.adjacency()
        
        # MSE + strong L1 + DAG constraint
        mse_loss = torch.nn.functional.mse_loss(predictions, targets)
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
            print(f"Epoch {epoch+1:3d} | Loss: {total_loss.item():.4f} | "
                  f"MSE: {mse_loss.item():.4f} | L1: {l1_loss.item():.4f} | "
                  f"DAG: {dag_constraint.item():.4f}")
    
    print("-" * 70)
    print("Phase 2 complete!")
    return history


def analyze_learned_structure(learned_adj, mask, var_names, top_k=10):
    """
    Analyze the learned adjacency matrix.
    
    Args:
        learned_adj: Learned adjacency matrix (n_vars x n_vars)
        mask: Original mask matrix (n_vars x n_vars)
        var_names: List of variable names
        top_k: Number of top edges to display
    """
    print("\n" + "=" * 70)
    print("LEARNED STRUCTURE ANALYSIS")
    print("=" * 70)
    
    n_vars = len(var_names)
    
    # Overall statistics
    print("\nOverall Statistics:")
    print(f"  Variables: {n_vars}")
    print(f"  Trainable edges (mask): {mask.sum()}")
    print(f"  Non-zero weights (|w| > 0.1): {(np.abs(learned_adj) > 0.1).sum()}")
    print(f"  Strong edges (|w| > 0.5): {(np.abs(learned_adj) > 0.5).sum()}")
    
    # Top-K strongest edges
    print("\n" + "-" * 70)
    print(f"TOP {top_k} STRONGEST LEARNED EDGES")
    print("-" * 70)
    
    # Get all edges with their weights
    edges = []
    for i in range(n_vars):
        for j in range(n_vars):
            if mask[i, j] == 1 and i != j:  # Only consider trainable edges
                weight = learned_adj[i, j]
                edges.append((i, j, weight, abs(weight)))
    
    # Sort by absolute weight
    edges.sort(key=lambda x: x[3], reverse=True)
    
    print(f"\n{'Rank':<6} {'Source':<20} {'Target':<20} {'Weight':<10} {'|Weight|':<10}")
    print("-" * 70)
    for rank, (i, j, weight, abs_weight) in enumerate(edges[:top_k], 1):
        print(f"{rank:<6} {var_names[i]:<20} {var_names[j]:<20} "
              f"{weight:9.4f} {abs_weight:9.4f}")
    
    # Pruned edges (mask allowed but weight is near zero)
    print("\n" + "-" * 70)
    print("PRUNED EDGES (mask allowed but learned weight ~0)")
    print("-" * 70)
    
    threshold = 0.05
    pruned = []
    for i in range(n_vars):
        for j in range(n_vars):
            if mask[i, j] == 1 and abs(learned_adj[i, j]) < threshold:
                pruned.append((i, j))
    
    print(f"\nEdges pruned (|w| < {threshold}): {len(pruned)}/{mask.sum()}")
    if pruned:
        print("\nSample pruned edges (first 20):")
        for i, j in pruned[:20]:
            print(f"  {var_names[i]:<20} -> {var_names[j]:<20} (w={learned_adj[i, j]:.4f})")
        if len(pruned) > 20:
            print(f"  ... and {len(pruned) - 20} more")
    
    # Bidirectional edge resolution
    print("\n" + "-" * 70)
    print("BIDIRECTIONAL EDGE RESOLUTION")
    print("-" * 70)
    
    bidirectional = []
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            if mask[i, j] == 1 and mask[j, i] == 1:
                w_ij = learned_adj[i, j]
                w_ji = learned_adj[j, i]
                bidirectional.append((i, j, w_ij, w_ji))
    
    print(f"\nBidirectional edges in mask: {len(bidirectional)}")
    if bidirectional:
        print(f"\n{'Source':<20} {'Target':<20} {'W(i->j)':<10} {'W(j->i)':<10} {'Selected'}")
        print("-" * 70)
        for i, j, w_ij, w_ji in bidirectional:
            if abs(w_ij) > abs(w_ji):
                selected = f"{var_names[i]} -> {var_names[j]}"
            else:
                selected = f"{var_names[j]} -> {var_names[i]}"
            print(f"{var_names[i]:<20} {var_names[j]:<20} "
                  f"{w_ij:9.4f} {w_ji:9.4f} {selected}")
    
    # Degree distribution
    print("\n" + "-" * 70)
    print("DEGREE DISTRIBUTION (|weight| > 0.1)")
    print("-" * 70)
    
    binary_adj = (np.abs(learned_adj) > 0.1).astype(int)
    out_degrees = binary_adj.sum(axis=1)
    in_degrees = binary_adj.sum(axis=0)
    
    print(f"\nMost connected variables (out-degree):")
    top_out_idx = np.argsort(out_degrees)[-5:][::-1]
    for idx in top_out_idx:
        if out_degrees[idx] > 0:
            print(f"  {var_names[idx]:<20}: {out_degrees[idx]:2d} outgoing edges")
    
    print(f"\nMost connected variables (in-degree):")
    top_in_idx = np.argsort(in_degrees)[-5:][::-1]
    for idx in top_in_idx:
        if in_degrees[idx] > 0:
            print(f"  {var_names[idx]:<20}: {in_degrees[idx]:2d} incoming edges")


def visualize_results(phase1_history, phase2_history, learned_adj, var_names):
    """Create visualizations of training and results."""
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Training curves
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
    axes[0, 0].axvline(n_phase1, color='black', linestyle='--', alpha=0.3)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss (ALARM Two-Phase Training)')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # MSE loss
    axes[0, 1].plot(epochs_phase1, phase1_history['mse_loss'], 
                    label='Phase 1', color='blue', alpha=0.7)
    axes[0, 1].plot(epochs_phase2, phase2_history['mse_loss'], 
                    label='Phase 2', color='red', alpha=0.7)
    axes[0, 1].axvline(n_phase1, color='black', linestyle='--', alpha=0.3)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].set_title('Prediction Error')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # L1 loss
    axes[1, 0].plot(epochs_phase1, phase1_history['l1_loss'], 
                    label='Phase 1', color='blue', alpha=0.7)
    axes[1, 0].plot(epochs_phase2, phase2_history['l1_loss'], 
                    label='Phase 2', color='red', alpha=0.7)
    axes[1, 0].axvline(n_phase1, color='black', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('L1 Loss')
    axes[1, 0].set_title('Sparsity Regularization')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # DAG constraint
    axes[1, 1].plot(epochs_phase2, phase2_history['dag_loss'], 
                    label='Phase 2', color='red', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('DAG Constraint')
    axes[1, 1].set_title('Acyclicity Constraint (NOTEARS)')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'alarm_training_curves.png', dpi=150, bbox_inches='tight')
    print("\nSaved training curves to: results/alarm_training_curves.png")
    plt.close()
    
    # Weight distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of weights
    weights_flat = learned_adj.flatten()
    weights_nonzero = weights_flat[np.abs(weights_flat) > 0.01]
    
    axes[0].hist(weights_nonzero, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Weight Value')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Weight Distribution (|w| > 0.01)\nn={len(weights_nonzero)} edges')
    axes[0].grid(alpha=0.3)
    
    # Heatmap (sampled for visibility)
    sample_size = min(20, len(var_names))
    sample_idx = np.linspace(0, len(var_names)-1, sample_size, dtype=int)
    sample_adj = learned_adj[np.ix_(sample_idx, sample_idx)]
    sample_names = [var_names[i] for i in sample_idx]
    
    im = axes[1].imshow(sample_adj, cmap='RdBu_r', aspect='auto', 
                        vmin=-np.max(np.abs(learned_adj)), 
                        vmax=np.max(np.abs(learned_adj)))
    axes[1].set_xticks(range(sample_size))
    axes[1].set_yticks(range(sample_size))
    axes[1].set_xticklabels(sample_names, rotation=90, fontsize=8)
    axes[1].set_yticklabels(sample_names, fontsize=8)
    axes[1].set_title(f'Learned Adjacency (Sample: {sample_size}x{sample_size})')
    plt.colorbar(im, ax=axes[1], label='Weight')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'alarm_learned_weights.png', dpi=150, bbox_inches='tight')
    print("Saved weight visualizations to: results/alarm_learned_weights.png")
    plt.close()


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("STEP 5: NEURAL LP TRAINING ON ALARM NETWORK")
    print("=" * 70)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load or generate data
    data_file = Path('data/alarm_data.csv')
    if data_file.exists():
        print("\nLoading ALARM data from file...")
        df = pd.read_csv(data_file)
        print(f"Loaded data shape: {df.shape}")
    else:
        print("\nALARM data file not found. Generating synthetic data...")
        df = generate_synthetic_alarm_data(n_samples=1000)
        # Save for future use
        df.to_csv(data_file, index=False)
        print(f"Saved synthetic data to: {data_file}")
    
    data = torch.FloatTensor(df.values)
    var_names = list(df.columns)
    n_vars = len(var_names)
    
    # Load mask
    print("\n" + "=" * 70)
    print("LOADING MASK MATRIX")
    print("=" * 70)
    
    mask_file = Path('data/alarm_mask_37x37.npy')
    mask = np.load(mask_file)
    
    print(f"\nMask shape: {mask.shape}")
    print(f"Trainable edges: {mask.sum()}")
    print(f"Sparsity: {(1 - mask.sum() / (mask.shape[0] * mask.shape[1])):.2%}")
    
    # Initialize Neural LP model
    print("\n" + "=" * 70)
    print("INITIALIZING NEURAL LP MODEL")
    print("=" * 70)
    
    model = NeuralLP(
        n_vars=n_vars,
        mask=mask,
        max_hops=2,
        init_value=0.1
    )
    
    print(f"\nModel configuration:")
    print(f"  Variables: {n_vars}")
    print(f"  Max hops: 2")
    print(f"  Trainable edges: {mask.sum()}")
    
    # Choose target variable (e.g., BP - blood pressure)
    target_var = 'BP' if 'BP' in var_names else var_names[-1]
    target_idx = var_names.index(target_var)
    
    print(f"  Target variable: {target_var} (index {target_idx})")
    
    # Phase 1: Learn signal
    phase1_history = train_phase1_signal(
        model=model,
        data=data,
        target_idx=target_idx,
        n_epochs=100,
        learning_rate=0.02,
        l1_lambda=0.001,
        print_every=20
    )
    
    print("\nPhase 1 learned weights (sample):")
    phase1_adj = model.get_adjacency_matrix()
    strong_edges_phase1 = (np.abs(phase1_adj) > 0.5).sum()
    print(f"  Strong edges (|w| > 0.5): {strong_edges_phase1}")
    
    # Phase 2: Prune noise
    phase2_history = train_phase2_prune(
        model=model,
        data=data,
        target_idx=target_idx,
        n_epochs=100,
        learning_rate=0.005,
        l1_lambda=0.00000001,
        dag_lambda=2.0,
        print_every=20
    )
    
    # Get final learned adjacency
    learned_adj = model.get_adjacency_matrix()
    
    # Analyze results
    analyze_learned_structure(learned_adj, mask, var_names, top_k=10)
    
    # Visualize
    try:
        visualize_results(phase1_history, phase2_history, learned_adj, var_names)
    except Exception as e:
        print(f"\n[WARNING] Could not create visualizations: {e}")
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / 'alarm_learned_adjacency.npy', learned_adj)
    print(f"\nSaved learned adjacency to: results/alarm_learned_adjacency.npy")
    
    # Save edge list
    edge_list = []
    for i in range(n_vars):
        for j in range(n_vars):
            if abs(learned_adj[i, j]) > 0.1:  # Threshold for "real" edges
                edge_list.append({
                    'source': var_names[i],
                    'target': var_names[j],
                    'weight': float(learned_adj[i, j]),
                    'abs_weight': float(abs(learned_adj[i, j]))
                })
    
    edge_df = pd.DataFrame(edge_list)
    edge_df = edge_df.sort_values('abs_weight', ascending=False)
    edge_df.to_csv(output_dir / 'alarm_learned_edges.csv', index=False)
    print(f"Saved edge list to: results/alarm_learned_edges.csv")
    
    print("\n" + "=" * 70)
    print("STEP 5 COMPLETE!")
    print("=" * 70)
    
    print("\nSummary:")
    print(f"  Trainable edges (mask): {mask.sum()}")
    print(f"  Learned edges (|w| > 0.1): {(np.abs(learned_adj) > 0.1).sum()}")
    print(f"  Strong edges (|w| > 0.5): {(np.abs(learned_adj) > 0.5).sum()}")
    print(f"  Pruning rate: {(1 - (np.abs(learned_adj) > 0.1).sum() / mask.sum()):.1%}")
    
    print("\nNext steps:")
    print("  1. Compare learned edges with ALARM ground truth structure")
    print("  2. Validate LLM-resolved edges against learned weights")
    print("  3. Analyze oriented bidirectional edges")
    print("  4. Test on real ALARM observational data (when available)")


if __name__ == "__main__":
    main()
