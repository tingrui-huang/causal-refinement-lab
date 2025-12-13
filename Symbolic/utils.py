"""
Utility functions for loading and working with generated data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


def load_ground_truth_data() -> pd.DataFrame:
    """
    Load the generated ground truth data.
    
    Returns:
        DataFrame with columns ['X', 'Y', 'Z']
    """
    return pd.read_csv('data/ground_truth_data.csv')


def load_ground_truth_structure() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ground truth adjacency matrix and weights.
    
    Returns:
        Tuple of (adjacency_matrix, weight_matrix)
    """
    adjacency = np.load('data/ground_truth_adjacency.npy')
    weights = np.load('data/ground_truth_weights.npy')
    return adjacency, weights


def load_fci_results() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load simulated FCI skeleton and mask matrix.
    
    Returns:
        Tuple of (skeleton, mask_matrix)
    """
    skeleton = np.load('data/fci_skeleton.npy')
    mask = np.load('data/mask_matrix.npy')
    return skeleton, mask


def get_all_data() -> Dict[str, np.ndarray]:
    """
    Load all generated data into a dictionary.
    
    Returns:
        Dictionary with keys:
        - 'data': DataFrame with X, Y, Z columns
        - 'gt_adjacency': Ground truth adjacency matrix
        - 'gt_weights': Ground truth weight matrix
        - 'fci_skeleton': Simulated FCI skeleton
        - 'mask': Mask matrix for training
    """
    data = load_ground_truth_data()
    gt_adj, gt_weights = load_ground_truth_structure()
    skeleton, mask = load_fci_results()
    
    return {
        'data': data,
        'gt_adjacency': gt_adj,
        'gt_weights': gt_weights,
        'fci_skeleton': skeleton,
        'mask': mask
    }


def print_matrix(matrix: np.ndarray, name: str, var_names: list = None):
    """
    Pretty print a matrix with variable names.
    
    Args:
        matrix: Matrix to print
        name: Name of the matrix
        var_names: List of variable names (default: ['X', 'Y', 'Z'])
    """
    if var_names is None:
        var_names = ['X', 'Y', 'Z']
    
    print(f"\n{name}:")
    print("    " + "  ".join(var_names))
    for i, var in enumerate(var_names):
        row_str = "  ".join([f"{val:3.0f}" if abs(val) < 10 else f"{val:3.1f}" 
                             for val in matrix[i]])
        print(f" {var}  {row_str}")


def compare_adjacency_matrices(predicted: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """
    Compare predicted adjacency matrix with ground truth.
    
    Args:
        predicted: Predicted adjacency matrix (can be continuous weights)
        ground_truth: Ground truth binary adjacency matrix
    
    Returns:
        Dictionary with metrics:
        - 'accuracy': Overall accuracy
        - 'true_positives': Number of correctly identified edges
        - 'false_positives': Number of incorrectly added edges
        - 'false_negatives': Number of missed edges
        - 'precision': Precision score
        - 'recall': Recall score
    """
    # Threshold predicted matrix if it contains continuous values
    pred_binary = (predicted > 0.5).astype(int)
    gt_binary = ground_truth.astype(int)
    
    # Calculate metrics
    tp = np.sum((pred_binary == 1) & (gt_binary == 1))
    fp = np.sum((pred_binary == 1) & (gt_binary == 0))
    fn = np.sum((pred_binary == 0) & (gt_binary == 1))
    tn = np.sum((pred_binary == 0) & (gt_binary == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn,
        'precision': precision,
        'recall': recall
    }


def print_comparison_metrics(metrics: Dict[str, float]):
    """
    Pretty print comparison metrics.
    
    Args:
        metrics: Dictionary from compare_adjacency_matrices
    """
    print("\n=== Adjacency Matrix Comparison ===")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print(f"  True Negatives:  {metrics['true_negatives']}")


if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("Loading all generated data...")
    print("=" * 60)
    
    data_dict = get_all_data()
    
    print(f"\nData shape: {data_dict['data'].shape}")
    print(f"First 5 rows:")
    print(data_dict['data'].head())
    
    print_matrix(data_dict['gt_adjacency'], "Ground Truth Adjacency")
    print_matrix(data_dict['gt_weights'], "Ground Truth Weights")
    print_matrix(data_dict['fci_skeleton'], "FCI Skeleton")
    print_matrix(data_dict['mask'], "Mask Matrix")
    
    # Compare FCI skeleton with ground truth
    print("\n" + "=" * 60)
    print("FCI Skeleton vs Ground Truth")
    print("=" * 60)
    metrics = compare_adjacency_matrices(
        data_dict['fci_skeleton'], 
        data_dict['gt_adjacency']
    )
    print_comparison_metrics(metrics)
    
    print("\nNote: FCI skeleton is symmetric (undirected), so it has")
    print("many false positives compared to the directed ground truth.")

