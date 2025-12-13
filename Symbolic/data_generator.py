"""
Data generator for causal chain experiments.
Generates synthetic data following specified causal structures.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class CausalChainGenerator:
    """
    Generates data following a simple causal chain: X -> Y -> Z
    """
    
    def __init__(
        self,
        n_samples: int = 1000,
        x_to_y_coef: float = 2.0,
        y_to_z_coef: float = -3.0,
        noise_std: float = 0.5,
        random_seed: Optional[int] = 42
    ):
        """
        Initialize the causal chain data generator.
        
        Args:
            n_samples: Number of samples to generate
            x_to_y_coef: Coefficient for X -> Y relationship
            y_to_z_coef: Coefficient for Y -> Z relationship
            noise_std: Standard deviation of Gaussian noise
            random_seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.x_to_y_coef = x_to_y_coef
        self.y_to_z_coef = y_to_z_coef
        self.noise_std = noise_std
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate(self) -> pd.DataFrame:
        """
        Generate data following the causal chain X -> Y -> Z.
        
        Returns:
            DataFrame with columns ['X', 'Y', 'Z']
        """
        # Generate root cause X from standard normal
        X = np.random.randn(self.n_samples)
        
        # Generate Y from X with noise: Y = coef * X + noise
        noise_y = np.random.randn(self.n_samples) * self.noise_std
        Y = self.x_to_y_coef * X + noise_y
        
        # Generate Z from Y with noise: Z = coef * Y + noise
        noise_z = np.random.randn(self.n_samples) * self.noise_std
        Z = self.y_to_z_coef * Y + noise_z
        
        # Create DataFrame
        df = pd.DataFrame({
            'X': X,
            'Y': Y,
            'Z': Z
        })
        
        return df
    
    def get_ground_truth_adjacency(self) -> np.ndarray:
        """
        Get the ground truth adjacency matrix for X -> Y -> Z.
        
        Returns:
            3x3 adjacency matrix where entry [i,j] = 1 means i -> j
        """
        # Order: X (0), Y (1), Z (2)
        adj_matrix = np.array([
            [0, 1, 0],  # X -> Y
            [0, 0, 1],  # Y -> Z
            [0, 0, 0]   # Z -> nothing
        ])
        return adj_matrix
    
    def get_ground_truth_weights(self) -> np.ndarray:
        """
        Get the ground truth weight matrix with actual coefficients.
        
        Returns:
            3x3 weight matrix with causal coefficients
        """
        weights = np.array([
            [0.0, self.x_to_y_coef, 0.0],  # X -> Y with coefficient
            [0.0, 0.0, self.y_to_z_coef],  # Y -> Z with coefficient
            [0.0, 0.0, 0.0]                 # Z -> nothing
        ])
        return weights
    
    def save_to_csv(self, filepath: str) -> None:
        """
        Generate and save data to CSV file.
        
        Args:
            filepath: Path to save the CSV file
        """
        df = self.generate()
        df.to_csv(filepath, index=False)
        print(f"Generated {self.n_samples} samples and saved to {filepath}")

