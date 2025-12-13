"""
Step 1 & 2: Generate ground truth data and simulate poor FCI results.

This script:
1. Creates a simple causal chain X -> Y -> Z
2. Generates 1000 samples with noise
3. Simulates a poor FCI result with undirected edges and spurious connections
4. Outputs the mask matrix for Neural LP training
"""

import numpy as np
import os
from data_generator import CausalChainGenerator
from fci_simulator import SimpleThreeVarFCISimulator


def main():
    # Create output directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("=" * 60)
    print("STEP 1: Generate Ground Truth Data")
    print("=" * 60)
    
    # Initialize data generator with ground truth: X -> Y -> Z
    # Y = 2X + noise
    # Z = -3Y + noise
    generator = CausalChainGenerator(
        n_samples=1000,
        x_to_y_coef=2.0,
        y_to_z_coef=-3.0,
        noise_std=0.5,
        random_seed=42
    )
    
    # Generate and save data
    data_path = 'data/ground_truth_data.csv'
    generator.save_to_csv(data_path)
    
    # Get ground truth structures
    gt_adjacency = generator.get_ground_truth_adjacency()
    gt_weights = generator.get_ground_truth_weights()
    
    print("\nGround Truth Causal Structure:")
    print("  X -> Y (coefficient: 2.0)")
    print("  Y -> Z (coefficient: -3.0)")
    
    print("\nGround Truth Adjacency Matrix:")
    print("     X  Y  Z")
    for i, var in enumerate(['X', 'Y', 'Z']):
        print(f"  {var} {gt_adjacency[i]}")
    
    print("\nGround Truth Weight Matrix:")
    print("     X    Y    Z")
    for i, var in enumerate(['X', 'Y', 'Z']):
        print(f"  {var} {gt_weights[i]}")
    
    # Save ground truth matrices
    np.save('data/ground_truth_adjacency.npy', gt_adjacency)
    np.save('data/ground_truth_weights.npy', gt_weights)
    
    print("\n" + "=" * 60)
    print("STEP 2: Simulate Poor FCI Results")
    print("=" * 60)
    
    # Initialize FCI simulator
    fci_sim = SimpleThreeVarFCISimulator()
    
    # Simulate poor FCI result
    skeleton, mask = fci_sim.simulate_poor_result()
    
    # Print summary
    fci_sim.print_skeleton_summary(skeleton)
    
    print("\nFCI Skeleton Matrix (1 = edge exists):")
    print("     X  Y  Z")
    for i, var in enumerate(['X', 'Y', 'Z']):
        print(f"  {var} {skeleton[i].astype(int)}")
    
    print("\nMask Matrix for Neural LP (1 = trainable, 0 = forbidden):")
    print("     X  Y  Z")
    for i, var in enumerate(['X', 'Y', 'Z']):
        print(f"  {var} {mask[i].astype(int)}")
    
    print("\nAllowed edges for training:")
    allowed_edges = []
    for i, from_var in enumerate(['X', 'Y', 'Z']):
        for j, to_var in enumerate(['X', 'Y', 'Z']):
            if mask[i, j] == 1:
                allowed_edges.append(f"{from_var} -> {to_var}")
    print("  " + ", ".join(allowed_edges))
    
    # Save FCI results
    np.save('data/fci_skeleton.npy', skeleton)
    np.save('data/mask_matrix.npy', mask)
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"[OK] Generated {generator.n_samples} samples")
    print(f"[OK] Saved to: {data_path}")
    print(f"[OK] Ground truth: X -> Y -> Z")
    print(f"[OK] FCI mistakes: all edges undirected + spurious X-Z edge")
    print(f"[OK] Mask matrix created with {int(mask.sum())} trainable edges")
    print("\nFiles saved:")
    print("  - data/ground_truth_data.csv")
    print("  - data/ground_truth_adjacency.npy")
    print("  - data/ground_truth_weights.npy")
    print("  - data/fci_skeleton.npy")
    print("  - data/mask_matrix.npy")
    print("\nReady for Step 3: Neural LP training!")


if __name__ == "__main__":
    main()

