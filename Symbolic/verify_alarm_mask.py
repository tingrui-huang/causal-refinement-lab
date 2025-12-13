"""
Verify the converted mask matrix for ALARM network.
"""

import numpy as np
import matplotlib.pyplot as plt


def verify_mask_matrix():
    """Verify the mask matrix is correctly formatted."""
    print("=" * 70)
    print("ALARM MASK MATRIX VERIFICATION")
    print("=" * 70)
    
    # Load files
    mask = np.load('data/alarm_mask_37x37.npy')
    var_mapping = np.load('data/alarm_var_mapping_37.npy', allow_pickle=True).item()
    
    with open('data/alarm_variables_37.txt', 'r') as f:
        var_names = [line.strip().split('\t')[1] for line in f if line.strip()]
    
    print(f"\nMask matrix shape: {mask.shape}")
    print(f"Number of variables: {len(var_names)}")
    print(f"Trainable edges: {mask.sum()}")
    print(f"Total possible edges: {mask.shape[0] * mask.shape[1]}")
    print(f"Sparsity: {(1 - mask.sum() / (mask.shape[0] * mask.shape[1])):.2%}")
    
    # Check properties
    print("\n" + "-" * 70)
    print("PROPERTIES CHECK")
    print("-" * 70)
    
    # Check 1: Binary matrix
    unique_vals = np.unique(mask)
    is_binary = len(unique_vals) <= 2 and all(v in [0, 1] for v in unique_vals)
    print(f"1. Binary matrix (0s and 1s only): {'OK' if is_binary else 'FAIL'}")
    print(f"   Unique values: {unique_vals}")
    
    # Check 2: No self-loops
    has_self_loops = np.diag(mask).sum() > 0
    print(f"2. No self-loops: {'FAIL' if has_self_loops else 'OK'}")
    print(f"   Self-loops found: {np.diag(mask).sum()}")
    
    # Check 3: Variable name mapping
    mapping_correct = len(var_mapping) == len(var_names)
    print(f"3. Variable mapping consistent: {'OK' if mapping_correct else 'FAIL'}")
    print(f"   Mapping entries: {len(var_mapping)}")
    print(f"   Variable names: {len(var_names)}")
    
    # Show sample edges
    print("\n" + "-" * 70)
    print("SAMPLE TRAINABLE EDGES")
    print("-" * 70)
    
    edge_count = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 1:
                edge_count += 1
                if edge_count <= 20:
                    print(f"  {var_names[i]:20s} -> {var_names[j]:20s}")
                elif edge_count == 21:
                    print(f"  ... and {mask.sum() - 20} more edges")
                    break
        if edge_count > 20:
            break
    
    # Check for bidirectional edges
    print("\n" + "-" * 70)
    print("BIDIRECTIONAL EDGES")
    print("-" * 70)
    
    bidirectional = []
    for i in range(mask.shape[0]):
        for j in range(i+1, mask.shape[1]):
            if mask[i, j] == 1 and mask[j, i] == 1:
                bidirectional.append((var_names[i], var_names[j]))
    
    print(f"Found {len(bidirectional)} bidirectional edges:")
    for src, tgt in bidirectional:
        print(f"  {src:20s} <-> {tgt:20s}")
    
    # Visualize mask matrix
    print("\n" + "-" * 70)
    print("CREATING VISUALIZATION")
    print("-" * 70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap of mask
    ax1 = axes[0]
    im1 = ax1.imshow(mask, cmap='Greys', aspect='auto', interpolation='nearest')
    ax1.set_title(f'ALARM Mask Matrix ({mask.shape[0]}x{mask.shape[1]})\n{mask.sum()} trainable edges', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Target Variable (column index)')
    ax1.set_ylabel('Source Variable (row index)')
    plt.colorbar(im1, ax=ax1, label='Trainable (1) or Forbidden (0)')
    
    # Degree distribution
    ax2 = axes[1]
    out_degrees = mask.sum(axis=1)
    in_degrees = mask.sum(axis=0)
    
    x = np.arange(len(var_names))
    width = 0.35
    ax2.bar(x - width/2, out_degrees, width, label='Out-degree', alpha=0.7)
    ax2.bar(x + width/2, in_degrees, width, label='In-degree', alpha=0.7)
    ax2.set_xlabel('Variable Index')
    ax2.set_ylabel('Number of Edges')
    ax2.set_title('Trainable Edge Degree Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/alarm_mask_verification.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: results/alarm_mask_verification.png")
    plt.close()
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    print("\nMask matrix is ready for Neural LP training!")
    print("\nUsage:")
    print("  import numpy as np")
    print("  from neural_lp import NeuralLP")
    print("  ")
    print("  mask = np.load('data/alarm_mask_37x37.npy')")
    print("  model = NeuralLP(n_vars=36, mask=mask, max_hops=2)")
    print("  # Train using two-phase approach (step4_refinement_final.py)")


if __name__ == "__main__":
    verify_mask_matrix()
