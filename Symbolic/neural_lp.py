"""
Neural LP module for causal structure refinement.

Implements differentiable multi-hop reasoning over causal graphs.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class MaskedAdjacencyMatrix(nn.Module):
    """
    Learnable adjacency matrix with mask constraints.
    
    The mask determines which edges are allowed to be learned.
    Masked positions are always zero and not trainable.
    """
    
    def __init__(self, n_vars: int, mask: np.ndarray, init_value: float = 0.1):
        """
        Initialize masked adjacency matrix.
        
        Args:
            n_vars: Number of variables
            mask: Binary mask matrix (1 = trainable, 0 = forbidden)
            init_value: Initial value for trainable weights
        """
        super().__init__()
        
        self.n_vars = n_vars
        self.mask = torch.FloatTensor(mask)
        
        # Initialize weights with small random values
        weights = torch.randn(n_vars, n_vars) * init_value
        
        # Apply mask: only trainable positions get non-zero values
        weights = weights * self.mask
        
        # Make it a learnable parameter
        self.weights = nn.Parameter(weights)
        
    def forward(self) -> torch.Tensor:
        """
        Get the masked adjacency matrix.
        
        Returns:
            Adjacency matrix with mask applied
        """
        # Apply mask to ensure forbidden edges stay zero
        return self.weights * self.mask
    
    def get_adjacency(self) -> torch.Tensor:
        """
        Get the current adjacency matrix (alias for forward).
        
        Returns:
            Masked adjacency matrix
        """
        return self.forward()


class MultiHopReasoning(nn.Module):
    """
    Multi-hop reasoning module using matrix powers.
    
    Computes predictions by following paths through the causal graph.
    Path length is controlled by the number of hops (matrix powers).
    """
    
    def __init__(self, adjacency_matrix: MaskedAdjacencyMatrix, max_hops: int = 2):
        """
        Initialize multi-hop reasoning.
        
        Args:
            adjacency_matrix: Learnable masked adjacency matrix
            max_hops: Maximum path length to consider
        """
        super().__init__()
        
        self.adjacency_matrix = adjacency_matrix
        self.max_hops = max_hops
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Perform multi-hop reasoning.
        
        Args:
            features: Input features [batch_size, n_vars]
        
        Returns:
            Predictions after multi-hop reasoning [batch_size, n_vars]
        """
        W = self.adjacency_matrix()
        
        # Start with input features
        current = features
        predictions = torch.zeros_like(features)
        
        # Accumulate predictions from each hop
        for hop in range(1, self.max_hops + 1):
            # Apply adjacency matrix: current @ W
            # This follows edges in the graph
            current = torch.matmul(current, W)
            predictions = predictions + current
        
        return predictions


class NeuralLP(nn.Module):
    """
    Neural Logic Programming model for causal structure learning.
    
    Combines masked adjacency matrix with multi-hop reasoning
    to learn causal structures from data.
    """
    
    def __init__(
        self, 
        n_vars: int, 
        mask: np.ndarray, 
        max_hops: int = 2,
        init_value: float = 0.1
    ):
        """
        Initialize Neural LP model.
        
        Args:
            n_vars: Number of variables
            mask: Binary mask matrix for allowed edges
            max_hops: Maximum path length for reasoning
            init_value: Initial value for weights
        """
        super().__init__()
        
        self.n_vars = n_vars
        self.max_hops = max_hops
        
        # Learnable adjacency matrix (Operator)
        self.adjacency = MaskedAdjacencyMatrix(n_vars, mask, init_value)
        
        # Multi-hop reasoning (Controller)
        self.reasoning = MultiHopReasoning(self.adjacency, max_hops)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict variables using multi-hop reasoning.
        
        Args:
            features: Input features [batch_size, n_vars]
        
        Returns:
            Predictions [batch_size, n_vars]
        """
        return self.reasoning(features)
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Get the current learned adjacency matrix.
        
        Returns:
            Adjacency matrix as numpy array
        """
        with torch.no_grad():
            return self.adjacency().cpu().numpy()
    
    def predict_target(
        self, 
        features: torch.Tensor, 
        target_idx: int
    ) -> torch.Tensor:
        """
        Predict a specific target variable.
        
        Args:
            features: Input features [batch_size, n_vars]
            target_idx: Index of target variable to predict
        
        Returns:
            Predictions for target variable [batch_size]
        """
        predictions = self.forward(features)
        return predictions[:, target_idx]


def compute_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    adjacency_matrix: torch.Tensor,
    l1_lambda: float = 0.01
) -> Tuple[torch.Tensor, dict]:
    """
    Compute loss with MSE and L1 regularization.
    
    Args:
        predictions: Predicted values
        targets: True values
        adjacency_matrix: Current adjacency matrix
        l1_lambda: L1 regularization strength
    
    Returns:
        Tuple of (total_loss, loss_dict)
    """
    # MSE loss for predictions
    mse_loss = nn.functional.mse_loss(predictions, targets)
    
    # L1 regularization for sparsity
    l1_loss = torch.sum(torch.abs(adjacency_matrix))
    
    # Total loss
    total_loss = mse_loss + l1_lambda * l1_loss
    
    loss_dict = {
        'total': total_loss.item(),
        'mse': mse_loss.item(),
        'l1': l1_loss.item()
    }
    
    return total_loss, loss_dict


def train_neural_lp(
    model: NeuralLP,
    data: torch.Tensor,
    target_idx: int,
    n_epochs: int = 1000,
    learning_rate: float = 0.01,
    l1_lambda: float = 0.01,
    print_every: int = 100,
    verbose: bool = True
) -> dict:
    """
    Train Neural LP model.
    
    Args:
        model: NeuralLP model to train
        data: Training data [n_samples, n_vars]
        target_idx: Index of target variable to predict
        n_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        l1_lambda: L1 regularization strength
        print_every: Print progress every N epochs
        verbose: Whether to print training progress
    
    Returns:
        Dictionary with training history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'total_loss': [],
        'mse_loss': [],
        'l1_loss': []
    }
    
    # Extract target values
    targets = data[:, target_idx]
    
    if verbose:
        print(f"\nTraining Neural LP to predict variable {target_idx}")
        print(f"Epochs: {n_epochs}, LR: {learning_rate}, L1: {l1_lambda}")
        print("-" * 60)
    
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
        
        # Record history
        history['total_loss'].append(loss_dict['total'])
        history['mse_loss'].append(loss_dict['mse'])
        history['l1_loss'].append(loss_dict['l1'])
        
        # Print progress
        if verbose and (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1:4d} | "
                  f"Loss: {loss_dict['total']:.4f} | "
                  f"MSE: {loss_dict['mse']:.4f} | "
                  f"L1: {loss_dict['l1']:.4f}")
    
    if verbose:
        print("-" * 60)
        print(f"Training completed!")
    
    return history


def threshold_adjacency(
    adjacency: np.ndarray, 
    threshold: float = 0.5
) -> np.ndarray:
    """
    Threshold adjacency matrix to binary.
    
    Args:
        adjacency: Continuous adjacency matrix
        threshold: Threshold value
    
    Returns:
        Binary adjacency matrix
    """
    return (np.abs(adjacency) > threshold).astype(int)


def print_adjacency_comparison(
    learned: np.ndarray,
    ground_truth: np.ndarray,
    var_names: list = None,
    threshold: float = 0.5
):
    """
    Print comparison between learned and ground truth adjacency.
    
    Args:
        learned: Learned adjacency matrix
        ground_truth: Ground truth adjacency matrix
        var_names: Variable names
        threshold: Threshold for binarization
    """
    if var_names is None:
        var_names = [f"V{i}" for i in range(len(learned))]
    
    print("\n" + "=" * 70)
    print("LEARNED ADJACENCY MATRIX (Continuous)")
    print("=" * 70)
    print("      " + "  ".join([f"{v:>6}" for v in var_names]))
    for i, var in enumerate(var_names):
        row_str = "  ".join([f"{learned[i, j]:6.3f}" for j in range(len(var_names))])
        print(f"  {var}   {row_str}")
    
    print("\n" + "=" * 70)
    print(f"LEARNED ADJACENCY MATRIX (Binary, threshold={threshold})")
    print("=" * 70)
    binary_learned = threshold_adjacency(learned, threshold)
    print("      " + "  ".join([f"{v:>6}" for v in var_names]))
    for i, var in enumerate(var_names):
        row_str = "  ".join([f"{binary_learned[i, j]:6d}" for j in range(len(var_names))])
        print(f"  {var}   {row_str}")
    
    print("\n" + "=" * 70)
    print("GROUND TRUTH ADJACENCY MATRIX")
    print("=" * 70)
    print("      " + "  ".join([f"{v:>6}" for v in var_names]))
    for i, var in enumerate(var_names):
        row_str = "  ".join([f"{int(ground_truth[i, j]):6d}" for j in range(len(var_names))])
        print(f"  {var}   {row_str}")
    
    # Compute accuracy
    correct = np.sum(binary_learned == ground_truth)
    total = ground_truth.size
    accuracy = correct / total
    
    print("\n" + "=" * 70)
    print("COMPARISON METRICS")
    print("=" * 70)
    print(f"Accuracy: {accuracy:.3f} ({correct}/{total} edges correct)")
    
    # Edge-wise comparison
    print("\nEdge-by-edge comparison:")
    for i, from_var in enumerate(var_names):
        for j, to_var in enumerate(var_names):
            if i == j:
                continue
            
            learned_val = learned[i, j]
            binary_val = binary_learned[i, j]
            gt_val = ground_truth[i, j]
            
            status = "[OK]" if binary_val == gt_val else "[X]"
            if binary_val == 1 or gt_val == 1:
                print(f"  {status} {from_var}->{to_var}: "
                      f"learned={learned_val:.3f}, "
                      f"binary={binary_val}, "
                      f"truth={int(gt_val)}")

