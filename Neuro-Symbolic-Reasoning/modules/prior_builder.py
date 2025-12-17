"""
Prior Builder Module

Integrates prior knowledge from:
1. FCI: Skeleton mask (which variable pairs are connected)
2. LLM: Direction prior (initial weights for specific rules)
3. Domain knowledge: Normal state handling
"""

import torch
import pandas as pd
from typing import Dict, List, Tuple, Set
from pathlib import Path


class PriorBuilder:
    """
    Build prior knowledge structures for causal discovery
    
    Key responsibilities:
    1. FCI skeleton mask: (105, 105) binary mask
    2. LLM direction prior: (105, 105) initial weights
    3. Normal penalty weights: (105, 105) for weighted Group Lasso
    4. Block structure: List of blocks for Group Lasso
    """
    
    def __init__(self, var_structure: Dict):
        """
        Args:
            var_structure: Variable structure from DataLoader
        """
        self.var_structure = var_structure
        self.n_states = var_structure['n_states']
        
        print("=" * 70)
        print("PRIOR BUILDER INITIALIZED")
        print("=" * 70)
    
    def build_skeleton_mask_from_fci(self, fci_csv_path: str) -> torch.Tensor:
        """
        Build skeleton mask from FCI results
        
        Args:
            fci_csv_path: Path to FCI edges CSV (variable-level)
        
        Returns:
            Binary mask (105, 105) where 1 = allowed, 0 = forbidden
            
        Logic:
            - If FCI says A-B connected (any direction) -> entire block A->B is 1
            - Otherwise -> block is 0
        """
        print("\n" + "=" * 70)
        print("BUILDING SKELETON MASK FROM FCI")
        print("=" * 70)
        
        # Initialize mask to zeros (nothing allowed)
        skeleton_mask = torch.zeros(self.n_states, self.n_states)
        
        # Load FCI edges
        df_fci = pd.read_csv(fci_csv_path)
        print(f"Loaded {len(df_fci)} edges from FCI")
        
        # Determine column names
        if 'Source' in df_fci.columns and 'Target' in df_fci.columns:
            source_col, target_col = 'Source', 'Target'
        elif 'source' in df_fci.columns and 'target' in df_fci.columns:
            source_col, target_col = 'source', 'target'
        else:
            # Use first two columns
            source_col, target_col = df_fci.columns[0], df_fci.columns[1]
        
        print(f"Using columns: {source_col} -> {target_col}")
        
        # Count edges
        edge_count = 0
        
        # For each FCI edge, enable the entire block
        for _, row in df_fci.iterrows():
            var_a = row[source_col]
            var_b = row[target_col]
            
            # Check if variables exist
            if var_a not in self.var_structure['var_to_states']:
                print(f"Warning: Variable {var_a} not found in structure")
                continue
            if var_b not in self.var_structure['var_to_states']:
                print(f"Warning: Variable {var_b} not found in structure")
                continue
            
            # Get state indices for both variables
            states_a = self.var_structure['var_to_states'][var_a]
            states_b = self.var_structure['var_to_states'][var_b]
            
            # Enable entire block A -> B
            for i in states_a:
                for j in states_b:
                    skeleton_mask[i, j] = 1
            
            edge_count += 1
        
        # Calculate statistics
        total_possible = self.n_states * self.n_states
        allowed = int(skeleton_mask.sum().item())
        
        print(f"\nSkeleton mask statistics:")
        print(f"  Edges processed: {edge_count}")
        print(f"  Allowed connections: {allowed} / {total_possible} ({allowed/total_possible*100:.2f}%)")
        print(f"  Forbidden connections: {total_possible - allowed}")
        
        return skeleton_mask
    
    def build_direction_prior_from_llm(self, llm_csv_path: str, 
                                       high_confidence: float = 0.7,
                                       low_confidence: float = 0.3) -> torch.Tensor:
        """
        Build direction prior from FCI+LLM hybrid CSV
        
        This reads the edges_Hybrid_FCI_LLM_*.csv file where LLM has resolved
        directions for partial/undirected edges from FCI.
        
        Args:
            llm_csv_path: Path to FCI+LLM hybrid CSV (e.g., edges_Hybrid_FCI_LLM_20251207_230956.csv)
            high_confidence: Weight for LLM-resolved or directed edges (0.7)
            low_confidence: Weight for undirected/partial edges (0.3)
        
        Returns:
            Direction prior matrix (105, 105)
        """
        print("\n" + "=" * 70)
        print("BUILDING DIRECTION PRIOR FROM FCI+LLM HYBRID")
        print("=" * 70)
        
        # Initialize with zeros (no prior)
        direction_prior = torch.zeros(self.n_states, self.n_states)
        
        # Load CSV
        df = pd.read_csv(llm_csv_path)
        
        print(f"Loaded {len(df)} edges from FCI+LLM hybrid")
        
        # Determine columns
        if 'source' in df.columns and 'target' in df.columns:
            source_col, target_col = 'source', 'target'
        elif 'Source' in df.columns and 'Target' in df.columns:
            source_col, target_col = 'Source', 'Target'
        else:
            raise ValueError(f"Cannot find source/target columns in {llm_csv_path}")
        
        print(f"Using columns: {source_col} -> {target_col}")
        
        # Process edges
        directed_count = 0
        llm_resolved_count = 0
        undirected_count = 0
        
        for _, row in df.iterrows():
            var_source = row[source_col]
            var_target = row[target_col]
            edge_type = row.get('edge_type', 'directed')
            
            # Get state indices for these variables
            source_states = self.var_structure['var_to_states'][var_source]
            target_states = self.var_structure['var_to_states'][var_target]
            
            # Determine confidence based on edge type
            if edge_type == 'llm_resolved':
                confidence = high_confidence
                llm_resolved_count += 1
            elif edge_type == 'directed':
                confidence = high_confidence
                directed_count += 1
            else:  # undirected, partial, tail-tail
                confidence = low_confidence
                undirected_count += 1
            
            # Set all state-to-state connections for this variable pair
            for i in source_states:
                for j in target_states:
                    direction_prior[i, j] = confidence
        
        print(f"\nDirection prior statistics:")
        print(f"  Directed edges: {directed_count} (confidence={high_confidence})")
        print(f"  LLM-resolved edges: {llm_resolved_count} (confidence={high_confidence})")
        print(f"  Undirected/partial edges: {undirected_count} (confidence={low_confidence})")
        print(f"  Total edges: {len(df)}")
        print(f"  Non-zero weights: {(direction_prior > 0).sum().item()}")
        print(f"  Mean non-zero weight: {direction_prior[direction_prior > 0].mean().item():.4f}")
        
        return direction_prior
    
    def build_normal_penalty_weights(self, normal_weight: float = 0.1, 
                                     abnormal_weight: float = 1.0) -> torch.Tensor:
        """
        Build penalty weight matrix for Weighted Group Lasso
        
        Critical for Phase 2: This implements Gemini's suggestion
        
        Args:
            normal_weight: Weight for Normal -> Normal (low, e.g., 0.1)
            abnormal_weight: Weight for other connections (high, e.g., 1.0)
        
        Returns:
            Penalty weight matrix (105, 105)
            
        Logic:
            - Normal -> Normal: low weight (allow but don't force)
            - All other: high weight (force sparsity)
        """
        print("\n" + "=" * 70)
        print("BUILDING NORMAL PENALTY WEIGHTS")
        print("=" * 70)
        
        penalty_weights = torch.ones(self.n_states, self.n_states) * abnormal_weight
        
        normal_to_normal_count = 0
        
        for i in range(self.n_states):
            for j in range(self.n_states):
                state_i_name = self.var_structure['idx_to_state'][i]
                state_j_name = self.var_structure['idx_to_state'][j]
                
                is_i_normal = 'Normal' in state_i_name
                is_j_normal = 'Normal' in state_j_name
                
                # Only Normal -> Normal gets low weight
                if is_i_normal and is_j_normal:
                    penalty_weights[i, j] = normal_weight
                    normal_to_normal_count += 1
        
        print(f"\nPenalty weight statistics:")
        print(f"  Normal -> Normal connections: {normal_to_normal_count} (weight={normal_weight})")
        print(f"  Other connections: {self.n_states**2 - normal_to_normal_count} (weight={abnormal_weight})")
        print(f"  Ratio: {normal_to_normal_count / (self.n_states**2) * 100:.2f}% protected")
        
        return penalty_weights
    
    def build_block_structure(self, skeleton_mask: torch.Tensor) -> List[Dict]:
        """
        Build block structure for Group Lasso
        
        CRITICAL: Only create blocks for edges allowed by FCI skeleton!
        This prevents the model from wasting computation on 1332 variable pairs
        when only ~45 are actually allowed by FCI.
        
        Args:
            skeleton_mask: (105, 105) binary mask from FCI
        
        Returns:
            List of block definitions for FCI-allowed variable pairs only
        """
        print("\n" + "=" * 70)
        print("BUILDING BLOCK STRUCTURE (FCI-CONSTRAINED)")
        print("=" * 70)
        
        blocks = []
        var_names = self.var_structure['variable_names']
        
        for var_a in var_names:
            for var_b in var_names:
                if var_a == var_b:
                    continue  # Skip self-loops
                
                # Get state indices for this variable pair
                row_indices = self.var_structure['var_to_states'][var_a]
                col_indices = self.var_structure['var_to_states'][var_b]
                
                # Check if this block has ANY allowed connections in skeleton
                # Extract the sub-matrix for this block
                block_mask = skeleton_mask[row_indices][:, col_indices]
                
                # Only create block if at least one connection is allowed
                if block_mask.sum().item() > 0:
                    blocks.append({
                        'var_pair': (var_a, var_b),
                        'row_indices': row_indices,
                        'col_indices': col_indices
                    })
        
        print(f"Total blocks (FCI-allowed): {len(blocks)}")
        print(f"Total possible blocks: {len(var_names) * (len(var_names) - 1)} = {len(var_names) * (len(var_names) - 1)}")
        print(f"Reduction: {(1 - len(blocks) / (len(var_names) * (len(var_names) - 1))) * 100:.1f}%")
        
        # Show sample blocks
        print(f"\nSample FCI-allowed blocks:")
        for i, block in enumerate(blocks[:5]):
            var_a, var_b = block['var_pair']
            n_rows = len(block['row_indices'])
            n_cols = len(block['col_indices'])
            print(f"  {i+1}. {var_a} -> {var_b}: {n_rows} x {n_cols} = {n_rows * n_cols} connections")
        
        return blocks
    
    def get_all_priors(self, fci_skeleton_path: str, llm_direction_path: str = None, 
                      use_llm_prior: bool = True) -> Dict[str, torch.Tensor]:
        """
        Convenience method to build all priors at once
        
        IMPORTANT: Uses TWO separate CSV files:
        - fci_skeleton_path: Pure FCI results for HARD skeleton mask
        - llm_direction_path: FCI+LLM hybrid results for SOFT direction prior (optional)
        
        Args:
            fci_skeleton_path: Path to pure FCI edges (e.g., edges_FCI_20251207_230824.csv)
            llm_direction_path: Path to FCI+LLM edges (e.g., edges_Hybrid_FCI_LLM_20251207_230956.csv)
                               Can be None if use_llm_prior=False
            use_llm_prior: Whether to use LLM direction prior (default: True)
                          If False, uses uniform initialization (0.5 for all allowed edges)
        
        Returns:
            Dictionary with all prior structures
        """
        # Build skeleton from PURE FCI (hard constraint)
        skeleton_mask = self.build_skeleton_mask_from_fci(fci_skeleton_path)
        
        # Build direction prior
        if use_llm_prior and llm_direction_path:
            # Build direction prior from FCI+LLM hybrid (soft initialization)
            direction_prior = self.build_direction_prior_from_llm(llm_direction_path)
            print("\n[USING LLM DIRECTION PRIOR]")
        else:
            # Uniform initialization: 0.5 for all allowed edges, 0.0 for forbidden
            direction_prior = skeleton_mask * 0.5
            print("\n[NO LLM PRIOR - UNIFORM INITIALIZATION]")
            print("All FCI-allowed edges initialized with weight 0.5")
        
        # Build penalty weights for Normal state handling
        penalty_weights = self.build_normal_penalty_weights()
        
        # Build blocks ONLY for FCI-allowed edges
        blocks = self.build_block_structure(skeleton_mask)
        
        return {
            'skeleton_mask': skeleton_mask,
            'direction_prior': direction_prior,
            'penalty_weights': penalty_weights,
            'blocks': blocks
        }


if __name__ == "__main__":
    # Test the prior builder
    import sys
    sys.path.append('..')
    from modules.data_loader import CausalDataLoader
    
    # Load data first to get variable structure
    loader = CausalDataLoader(
        data_path='data/alarm_data_10000.csv',
        metadata_path='output/knowledge_graph_metadata.json'
    )
    var_structure = loader.get_variable_structure()
    
    # Build priors
    prior_builder = PriorBuilder(var_structure)
    priors = prior_builder.get_all_priors(
        fci_csv_path='data/edges_Hybrid_FCI_LLM_20251207_230956.csv',
        llm_rules_path='llm_prior_rules'
    )
    
    print("\n" + "=" * 70)
    print("ALL PRIORS BUILT SUCCESSFULLY")
    print("=" * 70)
    for key, value in priors.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"{key}: {len(value)} items")

