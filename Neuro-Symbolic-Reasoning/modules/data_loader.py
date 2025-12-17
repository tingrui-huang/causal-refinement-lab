"""
Data Loader Module

Core change from previous version:
- Direct CSV -> OneHot matrix (no triples intermediate format)
- Efficient tensor-based representation
- Variable structure metadata for block operations
"""

import json
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple


class CausalDataLoader:
    """
    Load observational data and build variable structure
    
    Key responsibilities:
    1. Load CSV data directly into (N, 105) OneHot matrix
    2. Build variable-to-states mapping for block operations
    3. Provide metadata for other modules
    """
    
    def __init__(self, data_path: str, metadata_path: str):
        """
        Args:
            data_path: Path to CSV file with observational data
            metadata_path: Path to JSON metadata with state mappings
        """
        self.data_path = Path(data_path)
        self.metadata_path = Path(metadata_path)
        
        # Load metadata first
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Build variable structure
        self.var_structure = self._build_variable_structure()
        
        print("=" * 70)
        print("DATA LOADER INITIALIZED")
        print("=" * 70)
        print(f"Variables: {self.var_structure['n_variables']}")
        print(f"Total states: {self.var_structure['n_states']}")
    
    def _build_variable_structure(self) -> Dict:
        """
        Build comprehensive variable structure metadata
        
        Returns:
            Dictionary with:
            - n_variables: Number of variables (37)
            - n_states: Total number of states (105)
            - var_to_states: {var_name: [state_indices]}
            - state_to_var: {state_idx: var_name}
            - idx_to_state: {state_idx: state_name}
            - state_to_idx: {state_name: state_idx}
        """
        state_to_idx = {}
        idx_to_state = {}
        var_to_states = {}
        state_to_var = {}
        idx_counter = 0
        
        # Build state indices from metadata
        for var_name, state_mapping in self.metadata['state_mappings'].items():
            var_states = []
            
            # Sort by state code to ensure consistent ordering
            sorted_codes = sorted(state_mapping.keys(), key=lambda x: int(x))
            
            for state_code in sorted_codes:
                state_name = state_mapping[state_code]
                
                # Map state to index
                state_to_idx[state_name] = idx_counter
                idx_to_state[idx_counter] = state_name
                state_to_var[idx_counter] = var_name
                var_states.append(idx_counter)
                
                idx_counter += 1
            
            var_to_states[var_name] = var_states
        
        return {
            'n_variables': len(var_to_states),
            'n_states': idx_counter,
            'var_to_states': var_to_states,
            'state_to_var': state_to_var,
            'idx_to_state': idx_to_state,
            'state_to_idx': state_to_idx,
            'variable_names': sorted(var_to_states.keys())
        }
    
    def load_data(self) -> torch.Tensor:
        """
        Load observational data as OneHot matrix
        
        Critical change: Direct CSV -> Tensor, no triples intermediate
        
        Returns:
            Tensor of shape (N_samples, 105) with binary values
            Each row is a patient, each column is a state
            Exactly 37 ones per row (one state per variable)
        """
        print("\n" + "=" * 70)
        print("LOADING OBSERVATIONAL DATA")
        print("=" * 70)
        
        # Read CSV
        df = pd.read_csv(self.data_path)
        
        # Get state columns (exclude sample_id if present)
        state_columns = [col for col in df.columns if col not in ['sample_id', 'patient_id']]
        
        # Verify all states are present
        assert len(state_columns) == self.var_structure['n_states'], \
            f"Expected {self.var_structure['n_states']} states, found {len(state_columns)}"
        
        # Convert to tensor directly
        # Each column in CSV corresponds to a state
        # Values are already 0/1
        data_matrix = torch.tensor(df[state_columns].values, dtype=torch.float32)
        
        # Verify data integrity
        n_samples = data_matrix.shape[0]
        states_per_sample = data_matrix.sum(dim=1)
        
        print(f"Samples loaded: {n_samples}")
        print(f"States per sample: {states_per_sample.mean().item():.1f} (expected: {self.var_structure['n_variables']})")
        print(f"Total facts: {int(data_matrix.sum().item())}")
        print(f"Matrix shape: {data_matrix.shape}")
        print(f"Matrix density: {data_matrix.mean().item() * 100:.2f}%")
        
        # Sanity check: each sample should have exactly n_variables states active
        assert torch.allclose(states_per_sample, torch.tensor(float(self.var_structure['n_variables']))), \
            "Each sample should have exactly one state per variable"
        
        return data_matrix
    
    def get_variable_structure(self) -> Dict:
        """Return variable structure metadata"""
        return self.var_structure
    
    def get_variable_blocks(self) -> List[Dict]:
        """
        Generate block structure for Group Lasso
        
        Returns:
            List of block definitions:
            [
                {
                    'var_pair': (var_a, var_b),
                    'row_indices': [indices for var_a states],
                    'col_indices': [indices for var_b states]
                },
                ...
            ]
        """
        blocks = []
        var_names = self.var_structure['variable_names']
        
        for var_a in var_names:
            for var_b in var_names:
                if var_a == var_b:
                    continue  # Skip self-loops
                
                blocks.append({
                    'var_pair': (var_a, var_b),
                    'row_indices': self.var_structure['var_to_states'][var_a],
                    'col_indices': self.var_structure['var_to_states'][var_b]
                })
        
        return blocks
    
    def get_state_info(self, state_idx: int) -> Dict:
        """
        Get information about a specific state
        
        Args:
            state_idx: State index (0-104)
        
        Returns:
            Dictionary with state_name, variable, is_normal
        """
        state_name = self.var_structure['idx_to_state'][state_idx]
        var_name = self.var_structure['state_to_var'][state_idx]
        is_normal = 'Normal' in state_name
        
        return {
            'state_name': state_name,
            'variable': var_name,
            'is_normal': is_normal
        }


if __name__ == "__main__":
    # Test the data loader
    loader = CausalDataLoader(
        data_path='data/alarm_data_10000.csv',
        metadata_path='output/knowledge_graph_metadata.json'
    )
    
    # Load data
    data = loader.load_data()
    print(f"\nData shape: {data.shape}")
    
    # Get variable structure
    var_struct = loader.get_variable_structure()
    print(f"\nVariables: {var_struct['n_variables']}")
    print(f"States: {var_struct['n_states']}")
    
    # Get blocks
    blocks = loader.get_variable_blocks()
    print(f"\nTotal blocks: {len(blocks)}")
    print(f"Expected: {var_struct['n_variables'] * (var_struct['n_variables'] - 1)}")

