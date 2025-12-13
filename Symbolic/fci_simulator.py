"""
FCI result simulator for testing causal refinement algorithms.
Simulates incorrect/incomplete causal discovery results.
"""

import numpy as np
from typing import List, Tuple, Optional


class FCISimulator:
    """
    Simulates a "bad" FCI result with undirected edges and spurious connections.
    """
    
    def __init__(self, variable_names: List[str]):
        """
        Initialize FCI simulator.
        
        Args:
            variable_names: List of variable names in order
        """
        self.variable_names = variable_names
        self.n_vars = len(variable_names)
        self.var_to_idx = {name: idx for idx, name in enumerate(variable_names)}
    
    def simulate_poor_skeleton(
        self,
        true_edges: List[Tuple[str, str]],
        spurious_edges: Optional[List[Tuple[str, str]]] = None,
        make_undirected: bool = True
    ) -> np.ndarray:
        """
        Simulate a poor FCI result with undirected edges and spurious connections.
        
        Args:
            true_edges: List of true causal edges (from, to)
            spurious_edges: List of spurious edges to add (from, to)
            make_undirected: If True, make all edges undirected (bidirectional)
        
        Returns:
            Adjacency matrix representing the poor skeleton
        """
        skeleton = np.zeros((self.n_vars, self.n_vars))
        
        # Add true edges
        for from_var, to_var in true_edges:
            from_idx = self.var_to_idx[from_var]
            to_idx = self.var_to_idx[to_var]
            skeleton[from_idx, to_idx] = 1
            
            if make_undirected:
                skeleton[to_idx, from_idx] = 1
        
        # Add spurious edges
        if spurious_edges:
            for from_var, to_var in spurious_edges:
                from_idx = self.var_to_idx[from_var]
                to_idx = self.var_to_idx[to_var]
                skeleton[from_idx, to_idx] = 1
                
                if make_undirected:
                    skeleton[to_idx, from_idx] = 1
        
        return skeleton
    
    def generate_mask_matrix(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Generate a mask matrix from the skeleton.
        Mask[i,j] = 1 means edge i -> j is allowed during training.
        
        Args:
            skeleton: Adjacency matrix representing the skeleton
        
        Returns:
            Mask matrix where 1 = trainable, 0 = forbidden
        """
        # Wherever skeleton has an edge (in either direction), allow training
        mask = np.maximum(skeleton, skeleton.T)
        
        # Ensure diagonal is 0 (no self-loops)
        np.fill_diagonal(mask, 0)
        
        return mask
    
    def get_edge_list(self, adjacency: np.ndarray, directed: bool = False) -> List[Tuple[str, str]]:
        """
        Convert adjacency matrix to edge list.
        
        Args:
            adjacency: Adjacency matrix
            directed: If False, only return one edge per undirected pair
        
        Returns:
            List of edges as (from, to) tuples
        """
        edges = []
        for i in range(self.n_vars):
            for j in range(self.n_vars):
                if adjacency[i, j] > 0:
                    if directed or i < j:  # For undirected, only add once
                        edges.append((self.variable_names[i], self.variable_names[j]))
        return edges
    
    def print_skeleton_summary(self, skeleton: np.ndarray) -> None:
        """
        Print a human-readable summary of the skeleton.
        
        Args:
            skeleton: Adjacency matrix to summarize
        """
        print("\n=== FCI Simulated Skeleton ===")
        edges = self.get_edge_list(skeleton, directed=True)
        
        # Group by undirected pairs
        undirected_pairs = set()
        for from_var, to_var in edges:
            from_idx = self.var_to_idx[from_var]
            to_idx = self.var_to_idx[to_var]
            
            # Check if bidirectional
            if skeleton[to_idx, from_idx] > 0:
                pair = tuple(sorted([from_var, to_var]))
                undirected_pairs.add(pair)
        
        if undirected_pairs:
            print("Undirected edges (direction unknown):")
            for var1, var2 in sorted(undirected_pairs):
                print(f"  {var1} - {var2}")
        
        # Find directed edges
        directed_edges = []
        for from_var, to_var in edges:
            from_idx = self.var_to_idx[from_var]
            to_idx = self.var_to_idx[to_var]
            if skeleton[to_idx, from_idx] == 0:
                directed_edges.append((from_var, to_var))
        
        if directed_edges:
            print("Directed edges:")
            for from_var, to_var in directed_edges:
                print(f"  {from_var} -> {to_var}")
        
        print(f"\nTotal edges: {len(undirected_pairs) + len(directed_edges)}")
        print("=" * 30)


class SimpleThreeVarFCISimulator(FCISimulator):
    """
    Specialized simulator for the X -> Y -> Z example with poor FCI results.
    """
    
    def __init__(self):
        super().__init__(variable_names=['X', 'Y', 'Z'])
    
    def simulate_poor_result(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the specific poor FCI result described:
        - X - Y (undirected, should be X -> Y)
        - Y - Z (undirected, should be Y -> Z)
        - X - Z (spurious edge, should not exist)
        
        Returns:
            Tuple of (skeleton, mask_matrix)
        """
        # True edges that FCI found (but got direction wrong)
        true_edges = [('X', 'Y'), ('Y', 'Z')]
        
        # Spurious edge that FCI incorrectly added
        spurious_edges = [('X', 'Z')]
        
        # Generate skeleton with all edges undirected
        skeleton = self.simulate_poor_skeleton(
            true_edges=true_edges,
            spurious_edges=spurious_edges,
            make_undirected=True
        )
        
        # Generate mask matrix
        mask = self.generate_mask_matrix(skeleton)
        
        return skeleton, mask

