"""
Causal Discovery Model Module

CRITICAL CHANGE from previous version:
- Sparse rule list -> Dense 105x105 masked matrix
- This enables:
  1. Exploration of new rules within FCI skeleton
  2. Efficient Group Lasso computation on blocks
  3. Standard matrix operations
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class CausalDiscoveryModel(nn.Module):
    """
    Dense matrix-based causal discovery model
    
    Core architecture:
    - Learnable: raw_adj (105, 105) - Dense weight matrix
    - Fixed: skeleton_mask (105, 105) - FCI constraint
    - Output: adjacency (105, 105) - Masked, activated weights
    
    Key difference from previous Neural LP:
    - OLD: self.rule_weights = Parameter(vector of length 54)
    - NEW: self.raw_adj = Parameter(matrix of size 105x105)
    """
    
    def __init__(self, 
                 n_states: int,
                 skeleton_mask: torch.Tensor,
                 direction_prior: torch.Tensor):
        """
        Args:
            n_states: Number of states (105)
            skeleton_mask: Binary mask from FCI (105, 105)
            direction_prior: Initial weights from LLM (105, 105)
        """
        super().__init__()
        
        self.n_states = n_states
        
        # CRITICAL: Dense learnable matrix (not sparse rule list!)
        # Initialize from direction_prior using logit
        # This ensures initial values match LLM suggestions
        init_logits = torch.logit(direction_prior.clamp(0.01, 0.99))
        self.raw_adj = nn.Parameter(init_logits)
        
        # Fixed skeleton mask (FCI constraint)
        self.register_buffer('skeleton_mask', skeleton_mask)
        
        print("=" * 70)
        print("CAUSAL DISCOVERY MODEL INITIALIZED")
        print("=" * 70)
        print(f"State space: {n_states}")
        print(f"Parameter matrix: {self.raw_adj.shape}")
        print(f"Total parameters: {self.raw_adj.numel()}")
        print(f"Skeleton constraint: {int(skeleton_mask.sum().item())} / {skeleton_mask.numel()} allowed")
    
    def get_adjacency(self) -> torch.Tensor:
        """
        Get current adjacency matrix
        
        Steps:
        1. Apply sigmoid activation to raw weights
        2. Apply skeleton mask (FCI constraint)
        
        Returns:
            Adjacency matrix (105, 105) with values in [0, 1]
        """
        # Sigmoid: convert logits to probabilities
        weights = torch.sigmoid(self.raw_adj)
        
        # Apply skeleton mask: zero out forbidden connections
        adjacency = weights * self.skeleton_mask
        
        return adjacency
    
    def forward(self, observations: torch.Tensor, n_hops: int = 1) -> torch.Tensor:
        """
        Multi-hop causal reasoning
        
        Args:
            observations: (batch, 105) binary observation matrix
            n_hops: Number of reasoning hops (default: 1)
        
        Returns:
            predictions: (batch, 105) predicted state probabilities
            
        Logic:
            For each hop:
            1. Apply adjacency matrix: next = current @ adjacency
            2. Preserve initial observations: next = max(next, observations)
            3. Clamp to [0, 1]
        """
        adjacency = self.get_adjacency()
        
        current = observations
        
        for hop in range(n_hops):
            # Single hop reasoning: matrix multiplication
            next_state = torch.matmul(current, adjacency)
            
            # Residual connection: preserve initial observations
            # This is critical to prevent observed facts from disappearing
            next_state = torch.max(next_state, observations)
            
            # Clamp to valid probability range
            next_state = torch.clamp(next_state, 0.0, 1.0)
            
            current = next_state
        
        return current
    
    def get_block_weights(self, row_indices: list, col_indices: list) -> torch.Tensor:
        """
        Extract a specific block from the adjacency matrix
        
        Args:
            row_indices: Row indices for the block
            col_indices: Column indices for the block
        
        Returns:
            Block submatrix
        """
        adjacency = self.get_adjacency()
        
        # Convert to tensors if needed
        if not isinstance(row_indices, torch.Tensor):
            row_indices = torch.tensor(row_indices, dtype=torch.long)
        if not isinstance(col_indices, torch.Tensor):
            col_indices = torch.tensor(col_indices, dtype=torch.long)
        
        # Extract block using advanced indexing
        block = adjacency[row_indices][:, col_indices]
        
        return block
    
    def get_variable_level_edges(self, var_structure: Dict, threshold: float = 0.3) -> set:
        """
        Extract variable-level causal edges from learned adjacency
        
        Args:
            var_structure: Variable structure from DataLoader
            threshold: Threshold for considering a block as having an edge
        
        Returns:
            Set of (var_a, var_b) tuples representing causal edges
        """
        adjacency = self.get_adjacency()
        edges = set()
        
        for var_a in var_structure['variable_names']:
            for var_b in var_structure['variable_names']:
                if var_a == var_b:
                    continue
                
                # Get block for this variable pair
                states_a = var_structure['var_to_states'][var_a]
                states_b = var_structure['var_to_states'][var_b]
                
                # Extract block
                block = adjacency[states_a][:, states_b]
                
                # Compute block strength (mean or max)
                block_strength = block.mean().item()
                
                # If block strength exceeds threshold, add edge
                if block_strength > threshold:
                    edges.add((var_a, var_b))
        
        return edges


# LossComputer moved to modules/loss.py in Phase 2


if __name__ == "__main__":
    # Test the model
    import sys
    sys.path.append('..')
    from modules.data_loader import CausalDataLoader
    from modules.prior_builder import PriorBuilder
    
    # Load data
    loader = CausalDataLoader(
        data_path='data/alarm_data_10000.csv',
        metadata_path='output/knowledge_graph_metadata.json'
    )
    data = loader.load_data()
    var_structure = loader.get_variable_structure()
    
    # Build priors
    prior_builder = PriorBuilder(var_structure)
    priors = prior_builder.get_all_priors(
        fci_csv_path='data/edges_Hybrid_FCI_LLM_20251207_230956.csv',
        llm_rules_path='llm_prior_rules'
    )
    
    # Initialize model
    model = CausalDiscoveryModel(
        n_states=var_structure['n_states'],
        skeleton_mask=priors['skeleton_mask'],
        direction_prior=priors['direction_prior']
    )
    
    # Test forward pass
    print("\n" + "=" * 70)
    print("TESTING FORWARD PASS")
    print("=" * 70)
    
    batch = data[:10]  # First 10 samples
    predictions = model(batch, n_hops=1)
    
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Input sum: {batch.sum().item()}")
    print(f"Output sum: {predictions.sum().item()}")
    
    # Test loss
    loss_computer = LossComputer()
    losses = loss_computer.compute_total_loss(predictions, batch)
    
    print(f"\nLoss: {losses['total'].item():.4f}")

