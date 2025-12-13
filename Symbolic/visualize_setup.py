"""
Visualization script to understand the ground truth vs FCI simulation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_data_relationships(data_path: str, output_path: str = 'results/data_relationships.png'):
    """
    Plot the relationships between X, Y, Z in the generated data.
    """
    df = pd.read_csv(data_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # X vs Y (should show positive correlation, coef=2)
    axes[0].scatter(df['X'], df['Y'], alpha=0.5, s=10)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('X -> Y (True: coef=2.0)')
    axes[0].grid(True, alpha=0.3)
    
    # Y vs Z (should show negative correlation, coef=-3)
    axes[1].scatter(df['Y'], df['Z'], alpha=0.5, s=10)
    axes[1].set_xlabel('Y')
    axes[1].set_ylabel('Z')
    axes[1].set_title('Y -> Z (True: coef=-3.0)')
    axes[1].grid(True, alpha=0.3)
    
    # X vs Z (indirect relationship through Y, should show negative)
    axes[2].scatter(df['X'], df['Z'], alpha=0.5, s=10)
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Z')
    axes[2].set_title('X vs Z (Indirect, no direct edge)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved data relationships plot to {output_path}")
    plt.close()


def plot_adjacency_matrices(output_path: str = 'results/adjacency_comparison.png'):
    """
    Plot ground truth vs FCI skeleton vs mask matrix.
    """
    gt_adj = np.load('data/ground_truth_adjacency.npy')
    fci_skeleton = np.load('data/fci_skeleton.npy')
    mask = np.load('data/mask_matrix.npy')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels = ['X', 'Y', 'Z']
    
    # Ground truth
    sns.heatmap(gt_adj, annot=True, fmt='.0f', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar=False, ax=axes[0], vmin=0, vmax=1)
    axes[0].set_title('Ground Truth\n(X -> Y -> Z)')
    axes[0].set_ylabel('From')
    axes[0].set_xlabel('To')
    
    # FCI skeleton
    sns.heatmap(fci_skeleton, annot=True, fmt='.0f', cmap='Oranges',
                xticklabels=labels, yticklabels=labels,
                cbar=False, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title('FCI Skeleton (Poor Result)\n(All undirected + spurious X-Z)')
    axes[1].set_ylabel('From')
    axes[1].set_xlabel('To')
    
    # Mask matrix
    sns.heatmap(mask, annot=True, fmt='.0f', cmap='Greens',
                xticklabels=labels, yticklabels=labels,
                cbar=False, ax=axes[2], vmin=0, vmax=1)
    axes[2].set_title('Mask Matrix\n(1=trainable, 0=forbidden)')
    axes[2].set_ylabel('From')
    axes[2].set_xlabel('To')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved adjacency comparison plot to {output_path}")
    plt.close()


def print_statistics(data_path: str):
    """
    Print statistics about the generated data.
    """
    df = pd.read_csv(data_path)
    
    print("\n" + "=" * 60)
    print("Data Statistics")
    print("=" * 60)
    print(f"Number of samples: {len(df)}")
    print(f"\nDescriptive statistics:")
    print(df.describe())
    
    print(f"\nCorrelation matrix:")
    print(df.corr())
    
    # Calculate empirical coefficients using linear regression
    from scipy import stats
    
    # X -> Y
    slope_xy, intercept_xy, r_xy, p_xy, se_xy = stats.linregress(df['X'], df['Y'])
    print(f"\nEmpirical X -> Y coefficient: {slope_xy:.3f} (true: 2.0)")
    print(f"  R-squared: {r_xy**2:.3f}")
    
    # Y -> Z
    slope_yz, intercept_yz, r_yz, p_yz, se_yz = stats.linregress(df['Y'], df['Z'])
    print(f"\nEmpirical Y -> Z coefficient: {slope_yz:.3f} (true: -3.0)")
    print(f"  R-squared: {r_yz**2:.3f}")
    
    # X -> Z (indirect)
    slope_xz, intercept_xz, r_xz, p_xz, se_xz = stats.linregress(df['X'], df['Z'])
    print(f"\nEmpirical X -> Z coefficient: {slope_xz:.3f} (should be ~-6.0 via Y)")
    print(f"  R-squared: {r_xz**2:.3f}")
    print(f"  (Expected: 2.0 * -3.0 = -6.0 through Y)")


def main():
    print("=" * 60)
    print("Visualizing Step 1 & 2 Results")
    print("=" * 60)
    
    # Print statistics
    print_statistics('data/ground_truth_data.csv')
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)
    
    try:
        plot_data_relationships('data/ground_truth_data.csv')
        plot_adjacency_matrices()
        print("\n[OK] All visualizations created successfully!")
    except ImportError as e:
        print(f"\n[WARNING] Could not create plots: {e}")
        print("Install matplotlib and seaborn to enable visualizations:")
        print("  pip install matplotlib seaborn")


if __name__ == "__main__":
    main()

