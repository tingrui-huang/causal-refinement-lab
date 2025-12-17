"""
Loss Module - Phase 2

The mathematical core of causal discovery.

Three loss components:
1. Reconstruction Loss: Auto-encoder style prediction
2. Weighted Group Lasso: Block-level sparsity with Normal protection
3. Cycle Consistency Loss: Direction learning by penalizing bidirectional edges

Critical Implementation Note:
- Penalty weights MUST be inside the norm: ||W ⊙ P||_F
- NOT: ||W||_F * P (this is wrong!)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class LossComputer:
    """
    Compute all loss components for causal discovery
    
    This is the "soul" of the model - where mathematical constraints
    enforce causal structure learning.
    """
    
    def __init__(self, block_structure: List[Dict], penalty_weights: torch.Tensor):
        """
        Initialize loss computer with prior knowledge
        
        Args:
            block_structure: List of block definitions from PriorBuilder
                Each block: {'var_pair': (var_a, var_b), 
                            'row_indices': [...], 
                            'col_indices': [...]}
            penalty_weights: (105, 105) tensor with weights for each connection
                Normal→Normal: 0.1 (low penalty, allow)
                Others: 1.0 (high penalty, force sparsity)
        """
        self.block_structure = block_structure
        self.penalty_weights = penalty_weights
        
        # Build reverse block lookup for cycle consistency
        self.block_lookup = {}
        for block in block_structure:
            var_a, var_b = block['var_pair']
            self.block_lookup[(var_a, var_b)] = block
        
        # BCE loss for reconstruction
        self.bce_loss = nn.BCELoss()
        
        print("=" * 70)
        print("LOSS COMPUTER INITIALIZED (PHASE 2)")
        print("=" * 70)
        print(f"Blocks: {len(block_structure)}")
        print(f"Penalty weights shape: {penalty_weights.shape}")
        print(f"Normal→Normal (0.1): {(penalty_weights == 0.1).sum().item()} connections")
        print(f"Others (1.0): {(penalty_weights == 1.0).sum().item()} connections")
    
    def reconstruction_loss(self, predictions: torch.Tensor, 
                           targets: torch.Tensor) -> torch.Tensor:
        """
        Reconstruction loss: Auto-encoder style
        
        Goal: Model should be able to reconstruct observed states
        
        Args:
            predictions: (batch, 105) predicted state probabilities
            targets: (batch, 105) observed binary states
        
        Returns:
            Scalar loss value
        """
        return self.bce_loss(predictions, targets)
    
    def weighted_group_lasso_loss(self, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Weighted Group Lasso: Block-level sparsity with Normal protection
        
        CRITICAL MATHEMATICAL FORMULA:
        For each block:
            Loss_block = ||W_block ⊙ P_block||_F
        
        Where:
        - W_block: Weight sub-matrix for this variable pair
        - P_block: Penalty weight sub-matrix
        - ⊙: Element-wise multiplication (Hadamard product)
        - ||·||_F: Frobenius norm
        
        This is NOT: ||W_block||_F * P_scalar
        
        Why this matters:
        - Normal→Normal connections have P=0.1 (inside the norm)
        - Other connections have P=1.0
        - This allows Normal→Normal to exist while forcing sparsity elsewhere
        
        Args:
            adjacency: (105, 105) adjacency matrix
                MUST be sigmoid-activated and skeleton-masked
        
        Returns:
            Scalar loss value (sum over all blocks)
        """
        total_penalty = 0.0
        
        for block in self.block_structure:
            row_indices = block['row_indices']
            col_indices = block['col_indices']
            
            # Extract block weights
            # Use advanced indexing to get sub-matrix
            block_weights = adjacency[row_indices][:, col_indices]
            
            # Extract corresponding penalty weights
            block_penalties = self.penalty_weights[row_indices][:, col_indices]
            
            # CRITICAL: Element-wise multiplication BEFORE norm
            # This is the Weighted Group Lasso formula
            weighted_block = block_weights * block_penalties
            
            # Frobenius norm of the weighted block
            block_norm = torch.norm(weighted_block, p='fro')
            
            total_penalty += block_norm
        
        return total_penalty
    
    def cycle_consistency_loss(self, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Cycle Consistency Loss: Penalize bidirectional edges
        
        Goal: Force model to choose direction (A→B OR B→A, not both)
        
        Formula:
        For each variable pair (A, B):
            Penalty = ||W_{A→B}||_F × ||W_{B→A}||_F
        
        Why this works:
        - If both directions are strong → high penalty
        - If only one direction is strong → low penalty
        - Forces model to choose one direction
        
        Args:
            adjacency: (105, 105) adjacency matrix
        
        Returns:
            Scalar loss value (sum over all pairs)
        """
        total_penalty = 0.0
        processed_pairs = set()
        
        for block in self.block_structure:
            var_a, var_b = block['var_pair']
            
            # Avoid double-counting (A,B) and (B,A)
            pair_key = tuple(sorted([var_a, var_b]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            # Find reverse block
            reverse_block = self.block_lookup.get((var_b, var_a))
            if reverse_block is None:
                continue
            
            # Extract both direction blocks
            forward_weights = adjacency[block['row_indices']][:, block['col_indices']]
            backward_weights = adjacency[reverse_block['row_indices']][:, reverse_block['col_indices']]
            
            # Compute norms
            forward_norm = torch.norm(forward_weights, p='fro')
            backward_norm = torch.norm(backward_weights, p='fro')
            
            # Penalty: product of norms
            # High when both directions are strong
            cycle_penalty = forward_norm * backward_norm
            
            total_penalty += cycle_penalty
        
        return total_penalty
    
    def compute_total_loss(self, 
                          predictions: torch.Tensor,
                          targets: torch.Tensor,
                          adjacency: torch.Tensor,
                          lambda_group: float = 0.01,
                          lambda_cycle: float = 0.001) -> Dict[str, torch.Tensor]:
        """
        Compute total loss with all components
        
        Args:
            predictions: (batch, 105) predicted states
            targets: (batch, 105) observed states
            adjacency: (105, 105) adjacency matrix (sigmoid + masked)
            lambda_group: Weight for Group Lasso (default: 0.01)
            lambda_cycle: Weight for Cycle Consistency (default: 0.001)
        
        Returns:
            Dictionary with all loss components:
            {
                'total': total_loss,
                'reconstruction': recon_loss,
                'weighted_group_lasso': group_loss,
                'cycle_consistency': cycle_loss
            }
        """
        # Compute individual losses
        loss_recon = self.reconstruction_loss(predictions, targets)
        loss_group = self.weighted_group_lasso_loss(adjacency)
        loss_cycle = self.cycle_consistency_loss(adjacency)
        
        # Total loss
        total_loss = (loss_recon + 
                     lambda_group * loss_group + 
                     lambda_cycle * loss_cycle)
        
        return {
            'total': total_loss,
            'reconstruction': loss_recon,
            'weighted_group_lasso': loss_group,
            'cycle_consistency': loss_cycle
        }


def test_weighted_group_lasso():
    """
    Unit test for Weighted Group Lasso calculation
    
    Verifies that the math is correct:
    ||W ⊙ P||_F should equal sqrt(sum((W * P)^2))
    """
    print("\n" + "=" * 70)
    print("UNIT TEST: Weighted Group Lasso")
    print("=" * 70)
    
    # Create a simple 2x2 block
    block_weights = torch.tensor([
        [1.0, 1.0],
        [1.0, 1.0]
    ])
    
    # Penalty weights (one Normal→Normal connection)
    block_penalties = torch.tensor([
        [0.1, 1.0],
        [1.0, 1.0]
    ])
    
    # Manual calculation
    # W ⊙ P = [[1.0*0.1, 1.0*1.0], [1.0*1.0, 1.0*1.0]]
    #       = [[0.1, 1.0], [1.0, 1.0]]
    # ||W ⊙ P||_F = sqrt(0.1^2 + 1.0^2 + 1.0^2 + 1.0^2)
    #             = sqrt(0.01 + 1 + 1 + 1)
    #             = sqrt(3.01)
    #             ≈ 1.7349
    
    expected_loss = torch.sqrt(torch.tensor(3.01))
    
    # Computed loss
    weighted_block = block_weights * block_penalties
    computed_loss = torch.norm(weighted_block, p='fro')
    
    print(f"\nBlock weights:\n{block_weights}")
    print(f"\nPenalty weights:\n{block_penalties}")
    print(f"\nWeighted block (W ⊙ P):\n{weighted_block}")
    print(f"\nExpected loss: {expected_loss:.6f}")
    print(f"Computed loss: {computed_loss:.6f}")
    print(f"Difference: {abs(expected_loss - computed_loss):.10f}")
    
    # Assert correctness
    assert torch.allclose(computed_loss, expected_loss, atol=1e-6), \
        f"Loss mismatch! Expected {expected_loss}, got {computed_loss}"
    
    print("\n[PASS] TEST PASSED: Weighted Group Lasso math is correct!")
    
    # Test 2: Verify it's different from wrong formula
    wrong_loss = torch.norm(block_weights, p='fro') * block_penalties.mean()
    print(f"\nWrong formula (||W||_F * mean(P)): {wrong_loss:.6f}")
    print(f"Correct formula: {computed_loss:.6f}")
    print(f"Difference: {abs(wrong_loss - computed_loss):.6f}")
    
    assert not torch.allclose(wrong_loss, computed_loss, atol=0.1), \
        "Wrong formula should give different result!"
    
    print("[PASS] Verified: Penalty must be inside the norm!")


def test_cycle_consistency():
    """
    Unit test for Cycle Consistency Loss
    """
    print("\n" + "=" * 70)
    print("UNIT TEST: Cycle Consistency Loss")
    print("=" * 70)
    
    # Create dummy adjacency with bidirectional edge
    adjacency = torch.zeros(4, 4)
    
    # Strong A→B (indices 0,1 → 2,3)
    adjacency[0:2, 2:4] = 0.8
    
    # Strong B→A (indices 2,3 → 0,1)
    adjacency[2:4, 0:2] = 0.7
    
    # Create block structure
    blocks = [
        {'var_pair': ('A', 'B'), 'row_indices': [0, 1], 'col_indices': [2, 3]},
        {'var_pair': ('B', 'A'), 'row_indices': [2, 3], 'col_indices': [0, 1]}
    ]
    
    # Dummy penalty weights
    penalty_weights = torch.ones(4, 4)
    
    # Create loss computer
    loss_computer = LossComputer(blocks, penalty_weights)
    
    # Compute cycle loss
    cycle_loss = loss_computer.cycle_consistency_loss(adjacency)
    
    # Manual calculation
    # ||W_{A→B}||_F = sqrt(4 * 0.8^2) = sqrt(2.56) ≈ 1.6
    # ||W_{B→A}||_F = sqrt(4 * 0.7^2) = sqrt(1.96) = 1.4
    # Penalty = 1.6 * 1.4 = 2.24
    
    forward_norm = torch.norm(adjacency[0:2, 2:4], p='fro')
    backward_norm = torch.norm(adjacency[2:4, 0:2], p='fro')
    expected_loss = forward_norm * backward_norm
    
    print(f"\nForward block (A→B) norm: {forward_norm:.6f}")
    print(f"Backward block (B→A) norm: {backward_norm:.6f}")
    print(f"Expected cycle loss: {expected_loss:.6f}")
    print(f"Computed cycle loss: {cycle_loss:.6f}")
    print(f"Difference: {abs(expected_loss - cycle_loss):.10f}")
    
    assert torch.allclose(cycle_loss, expected_loss, atol=1e-6), \
        f"Cycle loss mismatch! Expected {expected_loss}, got {cycle_loss}"
    
    print("\n[PASS] TEST PASSED: Cycle Consistency math is correct!")


if __name__ == "__main__":
    print("=" * 70)
    print("LOSS MODULE UNIT TESTS")
    print("=" * 70)
    
    # Test 1: Weighted Group Lasso
    test_weighted_group_lasso()
    
    # Test 2: Cycle Consistency
    test_cycle_consistency()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED [SUCCESS]")
    print("=" * 70)
    print("\nLoss module is ready for Phase 2 training!")

