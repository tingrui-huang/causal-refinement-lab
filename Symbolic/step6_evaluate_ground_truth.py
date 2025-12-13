"""
Step 6: Evaluate Learned Structure Against ALARM Ground Truth

This script performs comprehensive evaluation of the Neural LP learned structure
against the ALARM Bayesian network ground truth from the BIF file.

Metrics computed:
- Structural Hamming Distance (SHD)
- Precision, Recall, F1 Score
- Orientation Accuracy (for correctly predicted edges)
- False Discovery Rate (FDR)
- True Positive Rate (TPR)

The script handles:
1. BIF file parsing to extract true DAG structure
2. Variable ordering alignment (BIF order != training order)
3. Thresholding of learned weights to binary adjacency
4. Comprehensive metric computation
5. Visualization of results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from typing import Dict, Tuple, List


def parse_bif_structure(bif_path: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Parse BIF file to extract DAG structure.
    
    Args:
        bif_path: Path to .bif file
        
    Returns:
        Tuple of (edges_dict, all_variables)
        edges_dict: {child: [parent1, parent2, ...]}
        all_variables: List of all variable names in BIF order
    """
    print("=" * 70)
    print("PARSING BIF FILE")
    print("=" * 70)
    
    with open(bif_path, 'r') as f:
        content = f.read()
    
    # Extract all variable names
    var_pattern = r'variable\s+(\w+)\s+\{'
    variables = re.findall(var_pattern, content)
    
    print(f"\nFound {len(variables)} variables in BIF file")
    
    # Extract parent relationships from probability statements
    # Pattern: probability ( CHILD | PARENT1, PARENT2, ... )
    prob_pattern = r'probability\s*\(\s*(\w+)\s*(?:\|\s*([^\)]+))?\s*\)'
    
    edges = {}  # child: [parents]
    
    for match in re.finditer(prob_pattern, content):
        child = match.group(1)
        parents_str = match.group(2)
        
        if parents_str:
            # Parse parent list
            parents = [p.strip() for p in parents_str.split(',')]
            edges[child] = parents
        else:
            # No parents (root node)
            edges[child] = []
    
    print(f"Found {len(edges)} probability statements")
    
    # Count total edges
    total_edges = sum(len(parents) for parents in edges.values())
    print(f"Total edges in ground truth: {total_edges}")
    
    return edges, variables


def create_ground_truth_adjacency(
    edges: Dict[str, List[str]], 
    var_order: List[str]
) -> np.ndarray:
    """
    Create adjacency matrix from edge dictionary.
    
    Args:
        edges: {child: [parents]} dictionary
        var_order: Ordered list of variable names (defines row/col indices)
        
    Returns:
        Binary adjacency matrix A where A[i,j]=1 means var_i -> var_j
    """
    n_vars = len(var_order)
    var_to_idx = {var: idx for idx, var in enumerate(var_order)}
    
    adj = np.zeros((n_vars, n_vars), dtype=int)
    
    for child, parents in edges.items():
        if child not in var_to_idx:
            continue
        j = var_to_idx[child]  # child is target (column)
        
        for parent in parents:
            if parent not in var_to_idx:
                continue
            i = var_to_idx[parent]  # parent is source (row)
            adj[i, j] = 1
    
    return adj


def align_ground_truth_to_training_order(
    gt_adj: np.ndarray,
    bif_variables: List[str],
    training_variables: List[str]
) -> np.ndarray:
    """
    Reorder ground truth adjacency matrix to match training variable order.
    
    Args:
        gt_adj: Ground truth adjacency (BIF order)
        bif_variables: Variable names in BIF order
        training_variables: Variable names in training order
        
    Returns:
        Reordered adjacency matrix matching training order
    """
    print("\n" + "=" * 70)
    print("ALIGNING VARIABLE ORDERS")
    print("=" * 70)
    
    # Create mapping
    bif_to_idx = {var: idx for idx, var in enumerate(bif_variables)}
    
    # Check for missing variables
    missing_in_bif = set(training_variables) - set(bif_variables)
    missing_in_training = set(bif_variables) - set(training_variables)
    
    if missing_in_bif:
        print(f"\n[WARNING] Variables in training but not in BIF: {missing_in_bif}")
    if missing_in_training:
        print(f"\n[WARNING] Variables in BIF but not in training: {missing_in_training}")
    
    # Create reordered matrix
    n_train = len(training_variables)
    aligned_adj = np.zeros((n_train, n_train), dtype=int)
    
    for i, var_i in enumerate(training_variables):
        if var_i not in bif_to_idx:
            continue
        bif_i = bif_to_idx[var_i]
        
        for j, var_j in enumerate(training_variables):
            if var_j not in bif_to_idx:
                continue
            bif_j = bif_to_idx[var_j]
            
            aligned_adj[i, j] = gt_adj[bif_i, bif_j]
    
    print(f"\nAlignment complete:")
    print(f"  Training variables: {n_train}")
    print(f"  Edges in aligned GT: {aligned_adj.sum()}")
    
    return aligned_adj


def threshold_learned_adjacency(
    learned_adj: np.ndarray, 
    threshold: float = 0.3
) -> np.ndarray:
    """
    Threshold continuous learned weights to binary adjacency.
    
    Args:
        learned_adj: Continuous weight matrix
        threshold: Absolute value threshold
        
    Returns:
        Binary adjacency matrix
    """
    binary_adj = (np.abs(learned_adj) > threshold).astype(int)
    return binary_adj


def compute_structural_hamming_distance(
    gt_adj: np.ndarray, 
    pred_adj: np.ndarray
) -> Tuple[int, Dict[str, int]]:
    """
    Compute Structural Hamming Distance (SHD) and its components.
    
    SHD = # missing edges + # extra edges + # reversed edges
    
    Args:
        gt_adj: Ground truth adjacency
        pred_adj: Predicted adjacency
        
    Returns:
        Tuple of (total_shd, components_dict)
    """
    n = gt_adj.shape[0]
    
    # Missing edges: in GT but not in pred
    missing = 0
    # Extra edges: in pred but not in GT (and not reversed)
    extra = 0
    # Reversed edges: i->j in GT but j->i in pred
    reversed_edges = 0
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            gt_ij = gt_adj[i, j]
            pred_ij = pred_adj[i, j]
            gt_ji = gt_adj[j, i]
            pred_ji = pred_adj[j, i]
            
            if gt_ij == 1:
                # True edge i -> j
                if pred_ij == 0:
                    if pred_ji == 1:
                        # Reversed: predicted j -> i instead
                        reversed_edges += 1
                    else:
                        # Missing: not predicted at all
                        missing += 1
            else:
                # No true edge i -> j
                if pred_ij == 1:
                    # Check if this is counted as reversed
                    if gt_ji == 1 and pred_ji == 0:
                        # This is the reverse of j -> i, already counted
                        pass
                    else:
                        # Extra edge
                        extra += 1
    
    # To avoid double counting reversed edges
    reversed_edges = reversed_edges // 2
    
    shd = missing + extra + reversed_edges
    
    components = {
        'missing': missing,
        'extra': extra,
        'reversed': reversed_edges
    }
    
    return shd, components


def compute_precision_recall_f1(
    gt_adj: np.ndarray, 
    pred_adj: np.ndarray
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score for edge prediction.
    
    Args:
        gt_adj: Ground truth adjacency
        pred_adj: Predicted adjacency
        
    Returns:
        Dictionary with precision, recall, f1
    """
    # True positives: edges in both GT and pred (ignoring direction for now)
    # For directed edges, we consider i->j and j->i as different
    
    n = gt_adj.shape[0]
    
    tp = 0  # True positive: correctly predicted edges
    fp = 0  # False positive: predicted but not in GT
    fn = 0  # False negative: in GT but not predicted
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            gt_ij = gt_adj[i, j]
            pred_ij = pred_adj[i, j]
            
            if gt_ij == 1 and pred_ij == 1:
                tp += 1
            elif gt_ij == 0 and pred_ij == 1:
                fp += 1
            elif gt_ij == 1 and pred_ij == 0:
                fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0  # False Discovery Rate
    tpr = recall  # True Positive Rate = Recall
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fdr': fdr,
        'tpr': tpr,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def compute_orientation_accuracy(
    gt_adj: np.ndarray, 
    pred_adj: np.ndarray
) -> Dict[str, float]:
    """
    Compute orientation accuracy: proportion of correctly oriented edges.
    
    Only considers edges where there is an edge between i and j in both GT and pred.
    Checks if the direction is correct.
    
    Args:
        gt_adj: Ground truth adjacency
        pred_adj: Predicted adjacency
        
    Returns:
        Dictionary with orientation metrics
    """
    n = gt_adj.shape[0]
    
    # Find pairs (i,j) where there's an edge in both GT and pred
    correct_orientation = 0
    incorrect_orientation = 0
    
    processed_pairs = set()
    
    for i in range(n):
        for j in range(n):
            if i >= j:  # Only process each pair once
                continue
            
            pair = (min(i, j), max(i, j))
            if pair in processed_pairs:
                continue
            
            # Check if there's an edge between i and j in both
            gt_has_edge = (gt_adj[i, j] == 1) or (gt_adj[j, i] == 1)
            pred_has_edge = (pred_adj[i, j] == 1) or (pred_adj[j, i] == 1)
            
            if gt_has_edge and pred_has_edge:
                processed_pairs.add(pair)
                
                # Check if directions match
                if gt_adj[i, j] == pred_adj[i, j] and gt_adj[j, i] == pred_adj[j, i]:
                    correct_orientation += 1
                else:
                    incorrect_orientation += 1
    
    total = correct_orientation + incorrect_orientation
    orientation_accuracy = correct_orientation / total if total > 0 else 0.0
    
    return {
        'orientation_accuracy': orientation_accuracy,
        'correct_orientations': correct_orientation,
        'incorrect_orientations': incorrect_orientation,
        'total_comparable_edges': total
    }


def print_metrics_table(metrics: Dict, shd: int, shd_components: Dict):
    """Print a professional metrics table."""
    print("\n" + "=" * 70)
    print("EVALUATION METRICS")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("STRUCTURAL METRICS")
    print("-" * 70)
    print(f"Structural Hamming Distance (SHD):    {shd:6d}")
    print(f"  - Missing edges:                     {shd_components['missing']:6d}")
    print(f"  - Extra edges:                       {shd_components['extra']:6d}")
    print(f"  - Reversed edges:                    {shd_components['reversed']:6d}")
    
    print("\n" + "-" * 70)
    print("EDGE PREDICTION METRICS")
    print("-" * 70)
    print(f"True Positives (TP):                   {metrics['tp']:6d}")
    print(f"False Positives (FP):                  {metrics['fp']:6d}")
    print(f"False Negatives (FN):                  {metrics['fn']:6d}")
    print(f"")
    print(f"Precision:                             {metrics['precision']:6.2%}")
    print(f"Recall (TPR):                          {metrics['recall']:6.2%}")
    print(f"F1 Score:                              {metrics['f1']:6.2%}")
    print(f"False Discovery Rate (FDR):            {metrics['fdr']:6.2%}")
    
    print("\n" + "-" * 70)
    print("ORIENTATION ACCURACY *** KEY METRIC ***")
    print("-" * 70)
    print(f"Correctly oriented edges:              {metrics['correct_orientations']:6d}")
    print(f"Incorrectly oriented edges:            {metrics['incorrect_orientations']:6d}")
    print(f"Total comparable edges:                {metrics['total_comparable_edges']:6d}")
    print(f"")
    print(f"ORIENTATION ACCURACY:                  {metrics['orientation_accuracy']:6.2%}")
    print("=" * 70)


def visualize_comparison(
    gt_adj: np.ndarray,
    pred_adj: np.ndarray,
    var_names: List[str],
    learned_weights: np.ndarray,
    threshold: float,
    shd: int,
    shd_components: Dict,
    output_path: str = 'results/final_evaluation_metrics.png'
):
    """
    Create comprehensive visualization comparing ground truth and learned structure.
    
    Args:
        gt_adj: Ground truth binary adjacency
        pred_adj: Predicted binary adjacency
        var_names: Variable names
        learned_weights: Continuous learned weights
        threshold: Threshold used for binarization
        output_path: Output file path
    """
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Ground Truth Heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(gt_adj, cmap='Greys', aspect='auto', interpolation='nearest')
    ax1.set_title(f'Ground Truth (ALARM BIF)\n{gt_adj.sum()} edges', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Target Variable (column)')
    ax1.set_ylabel('Source Variable (row)')
    plt.colorbar(im1, ax=ax1, label='Edge (1) or No edge (0)')
    
    # 2. Learned Structure (Thresholded)
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(pred_adj, cmap='Greys', aspect='auto', interpolation='nearest')
    ax2.set_title(f'Learned Structure (|w| > {threshold})\n{pred_adj.sum()} edges', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Target Variable (column)')
    ax2.set_ylabel('Source Variable (row)')
    plt.colorbar(im2, ax=ax2, label='Edge (1) or No edge (0)')
    
    # 3. Difference Heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    diff = pred_adj.astype(float) - gt_adj.astype(float)
    # -1 = missing (in GT but not pred)
    #  0 = match
    # +1 = extra (in pred but not GT)
    im3 = ax3.imshow(diff, cmap='RdBu_r', aspect='auto', 
                     interpolation='nearest', vmin=-1, vmax=1)
    ax3.set_title('Difference (Learned - Ground Truth)\nRed=Extra, Blue=Missing', 
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('Target Variable (column)')
    ax3.set_ylabel('Source Variable (row)')
    cbar3 = plt.colorbar(im3, ax=ax3, ticks=[-1, 0, 1])
    cbar3.set_ticklabels(['Missing', 'Match', 'Extra'])
    
    # 4. Learned Weights (Continuous)
    ax4 = fig.add_subplot(gs[1, 0])
    max_abs_weight = np.max(np.abs(learned_weights))
    im4 = ax4.imshow(learned_weights, cmap='RdBu_r', aspect='auto',
                     vmin=-max_abs_weight, vmax=max_abs_weight)
    ax4.set_title(f'Learned Weights (Continuous)\nMax |w| = {max_abs_weight:.3f}', 
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('Target Variable (column)')
    ax4.set_ylabel('Source Variable (row)')
    plt.colorbar(im4, ax=ax4, label='Weight')
    
    # 5. Weight Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    weights_flat = learned_weights.flatten()
    weights_nonzero = weights_flat[np.abs(weights_flat) > 0.01]
    if len(weights_nonzero) > 0:
        ax5.hist(weights_nonzero, bins=30, edgecolor='black', alpha=0.7)
        ax5.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold}')
        ax5.axvline(-threshold, color='red', linestyle='--', linewidth=2)
        ax5.set_xlabel('Weight Value')
        ax5.set_ylabel('Count')
        ax5.set_title(f'Weight Distribution (|w| > 0.01)\nn={len(weights_nonzero)}', 
                      fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No weights > 0.01', ha='center', va='center', fontsize=14)
        ax5.set_title('Weight Distribution', fontsize=12, fontweight='bold')
    
    # 6. Confusion Matrix Style View
    ax6 = fig.add_subplot(gs[1, 2])
    tp = np.sum((gt_adj == 1) & (pred_adj == 1))
    fp = np.sum((gt_adj == 0) & (pred_adj == 1))
    fn = np.sum((gt_adj == 1) & (pred_adj == 0))
    tn = np.sum((gt_adj == 0) & (pred_adj == 0))
    
    confusion = np.array([[tn, fp], [fn, tp]])
    im6 = ax6.imshow(confusion, cmap='Blues', aspect='auto')
    ax6.set_xticks([0, 1])
    ax6.set_yticks([0, 1])
    ax6.set_xticklabels(['Pred: No Edge', 'Pred: Edge'])
    ax6.set_yticklabels(['GT: No Edge', 'GT: Edge'])
    ax6.set_title('Edge Prediction Confusion Matrix', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax6.text(j, i, confusion[i, j],
                           ha="center", va="center", color="black", fontsize=16, fontweight='bold')
    
    plt.colorbar(im6, ax=ax6)
    
    # 7. Degree Distribution Comparison
    ax7 = fig.add_subplot(gs[2, 0])
    gt_out_deg = gt_adj.sum(axis=1)
    pred_out_deg = pred_adj.sum(axis=1)
    
    x = np.arange(len(var_names))
    width = 0.35
    
    # Only show variables with at least one edge
    mask = (gt_out_deg > 0) | (pred_out_deg > 0)
    x_filtered = np.where(mask)[0]
    
    if len(x_filtered) > 0:
        sample_size = min(20, len(x_filtered))
        sample_idx = x_filtered[:sample_size]
        
        ax7.bar(np.arange(len(sample_idx)) - width/2, gt_out_deg[sample_idx], 
                width, label='Ground Truth', alpha=0.7)
        ax7.bar(np.arange(len(sample_idx)) + width/2, pred_out_deg[sample_idx], 
                width, label='Learned', alpha=0.7)
        
        ax7.set_xlabel('Variable (sample)')
        ax7.set_ylabel('Out-degree')
        ax7.set_title(f'Out-Degree Comparison (Top {sample_size} variables)', 
                      fontsize=12, fontweight='bold')
        ax7.legend()
        ax7.grid(axis='y', alpha=0.3)
    
    # 8. Metrics Summary Text
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')
    
    # Recompute metrics for display
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics_text = f"""
    EVALUATION SUMMARY
    
    Structural Hamming Distance (SHD): {shd}
      • Missing: {shd_components['missing']}
      • Extra: {shd_components['extra']}
      • Reversed: {shd_components['reversed']}
    
    Edge Prediction:
      • Precision: {precision:.2%}
      • Recall:    {recall:.2%}
      • F1 Score:  {f1:.2%}
    
    Counts:
      • True Positives:  {tp}
      • False Positives: {fp}
      • False Negatives: {fn}
      • True Negatives:  {tn}
    
    Ground Truth: {gt_adj.sum()} edges
    Predicted:    {pred_adj.sum()} edges
    Threshold:    |w| > {threshold}
    """
    
    ax8.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('ALARM Network: Ground Truth vs Learned Structure Evaluation', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    plt.close()


def main():
    """Main evaluation pipeline."""
    print("=" * 70)
    print("STEP 6: EVALUATE AGAINST ALARM GROUND TRUTH")
    print("=" * 70)
    
    # Paths
    bif_path = Path('../alarm.bif')
    if not bif_path.exists():
        bif_path = Path('alarm.bif')
    if not bif_path.exists():
        bif_path = Path('data/alarm.bif')
    
    learned_adj_path = Path('results/alarm_learned_adjacency.npy')
    var_names_path = Path('data/alarm_variables_37.txt')
    
    # Check files exist
    if not bif_path.exists():
        print(f"\n[ERROR] BIF file not found: {bif_path}")
        return
    
    if not learned_adj_path.exists():
        print(f"\n[ERROR] Learned adjacency not found: {learned_adj_path}")
        print("Run step5_alarm_training.py first!")
        return
    
    # 1. Parse BIF file
    edges_dict, bif_variables = parse_bif_structure(str(bif_path))
    
    # 2. Create ground truth adjacency in BIF order
    gt_adj_bif_order = create_ground_truth_adjacency(edges_dict, bif_variables)
    
    print(f"\nGround truth adjacency (BIF order):")
    print(f"  Shape: {gt_adj_bif_order.shape}")
    print(f"  Total edges: {gt_adj_bif_order.sum()}")
    
    # 3. Load training variable order
    with open(var_names_path, 'r') as f:
        training_variables = [line.strip().split('\t')[1] for line in f if line.strip()]
    
    print(f"\nTraining variables: {len(training_variables)}")
    
    # 4. Align ground truth to training order
    gt_adj_aligned = align_ground_truth_to_training_order(
        gt_adj_bif_order, bif_variables, training_variables
    )
    
    # 5. Load learned adjacency
    learned_adj = np.load(learned_adj_path)
    
    print(f"\nLearned adjacency:")
    print(f"  Shape: {learned_adj.shape}")
    print(f"  Max |weight|: {np.max(np.abs(learned_adj)):.4f}")
    print(f"  Non-zero weights (|w| > 0.01): {(np.abs(learned_adj) > 0.01).sum()}")
    
    # 6. Threshold learned adjacency
    threshold = 0.0000001
    pred_adj = threshold_learned_adjacency(learned_adj, threshold)
    
    print(f"\nThresholded learned structure (|w| > {threshold}):")
    print(f"  Predicted edges: {pred_adj.sum()}")
    
    # 7. Compute metrics
    print("\n" + "=" * 70)
    print("COMPUTING METRICS")
    print("=" * 70)
    
    # SHD
    shd, shd_components = compute_structural_hamming_distance(gt_adj_aligned, pred_adj)
    
    # Precision, Recall, F1
    edge_metrics = compute_precision_recall_f1(gt_adj_aligned, pred_adj)
    
    # Orientation Accuracy
    orientation_metrics = compute_orientation_accuracy(gt_adj_aligned, pred_adj)
    
    # Combine all metrics
    all_metrics = {**edge_metrics, **orientation_metrics}
    
    # 8. Print metrics
    print_metrics_table(all_metrics, shd, shd_components)
    
    # 9. Visualize
    visualize_comparison(
        gt_adj_aligned, pred_adj, training_variables, 
        learned_adj, threshold, shd, shd_components,
        output_path='results/final_evaluation_metrics.png'
    )
    
    # 10. Save detailed results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([{
        'metric': 'SHD',
        'value': shd,
        'description': 'Structural Hamming Distance'
    }, {
        'metric': 'Missing Edges',
        'value': shd_components['missing'],
        'description': 'Edges in GT but not predicted'
    }, {
        'metric': 'Extra Edges',
        'value': shd_components['extra'],
        'description': 'Edges predicted but not in GT'
    }, {
        'metric': 'Reversed Edges',
        'value': shd_components['reversed'],
        'description': 'Edges with wrong direction'
    }, {
        'metric': 'Precision',
        'value': f"{all_metrics['precision']:.4f}",
        'description': 'TP / (TP + FP)'
    }, {
        'metric': 'Recall',
        'value': f"{all_metrics['recall']:.4f}",
        'description': 'TP / (TP + FN)'
    }, {
        'metric': 'F1 Score',
        'value': f"{all_metrics['f1']:.4f}",
        'description': 'Harmonic mean of precision and recall'
    }, {
        'metric': 'Orientation Accuracy',
        'value': f"{all_metrics['orientation_accuracy']:.4f}",
        'description': 'Correctly oriented edges / total comparable edges'
    }])
    
    metrics_df.to_csv(output_dir / 'evaluation_metrics.csv', index=False)
    print(f"\nSaved metrics to: results/evaluation_metrics.csv")
    
    # Save aligned ground truth for reference
    np.save(output_dir / 'alarm_ground_truth_aligned.npy', gt_adj_aligned)
    print(f"Saved aligned ground truth to: results/alarm_ground_truth_aligned.npy")
    
    print("\n" + "=" * 70)
    print("STEP 6 COMPLETE!")
    print("=" * 70)
    
    print("\nKey Findings:")
    print(f"  - Ground truth edges: {gt_adj_aligned.sum()}")
    print(f"  - Predicted edges: {pred_adj.sum()}")
    print(f"  - Orientation accuracy: {all_metrics['orientation_accuracy']:.1%}")
    print(f"  - F1 Score: {all_metrics['f1']:.1%}")


if __name__ == "__main__":
    main()
