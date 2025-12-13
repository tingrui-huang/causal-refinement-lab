"""
Quick Start: Convert Hybrid FCI-LLM to Neural LP Mask

This script demonstrates the complete pipeline:
1. Load hybrid edges from CSV
2. Create mask matrix
3. Verify the mask
4. Show usage example
"""

import numpy as np
import pandas as pd
from pathlib import Path


def quick_demo():
    """Quick demonstration of the mask matrix."""
    
    print("=" * 70)
    print("ALARM NETWORK - NEURAL LP MASK MATRIX")
    print("=" * 70)
    
    # Load the mask
    mask_file = Path('data/alarm_mask_37x37.npy')
    if not mask_file.exists():
        print(f"\n[ERROR] Mask file not found: {mask_file}")
        print("\nRun the conversion script first:")
        print("  python convert_hybrid_to_mask.py")
        return
    
    mask = np.load(mask_file)
    
    # Load variable names
    var_file = Path('data/alarm_variables_37.txt')
    with open(var_file, 'r') as f:
        var_names = [line.strip().split('\t')[1] for line in f if line.strip()]
    
    # Display info
    print(f"\n[OK] Loaded mask matrix: {mask.shape}")
    print(f"[OK] Loaded {len(var_names)} variable names")
    
    print("\n" + "-" * 70)
    print("MASK STATISTICS")
    print("-" * 70)
    print(f"Trainable edges:  {mask.sum():4d}")
    print(f"Forbidden edges:  {mask.shape[0] * mask.shape[1] - mask.sum():4d}")
    print(f"Sparsity:         {(1 - mask.sum() / (mask.shape[0] * mask.shape[1])) * 100:5.2f}%")
    print(f"Self-loops:       {np.diag(mask).sum():4d}")
    
    # Show example usage
    print("\n" + "-" * 70)
    print("USAGE EXAMPLE")
    print("-" * 70)
    
    print("\n1. Load the mask:")
    print("   mask = np.load('Symbolic/data/alarm_mask_37x37.npy')")
    
    print("\n2. Initialize Neural LP model:")
    print("   from neural_lp import NeuralLP")
    print("   model = NeuralLP(n_vars=36, mask=mask, max_hops=2)")
    
    print("\n3. Train with two-phase approach (recommended):")
    print("   # Phase 1: Learn signal")
    print("   train_phase1(model, data, n_epochs=1500, l1_lambda=0.001)")
    print("   ")
    print("   # Phase 2: Prune noise")
    print("   train_phase2(model, data, n_epochs=2000, l1_lambda=0.08, dag_lambda=2.0)")
    
    print("\n4. Get learned adjacency:")
    print("   learned_adj = model.get_adjacency_matrix()")
    
    # Show some variables
    print("\n" + "-" * 70)
    print("ALARM VARIABLES (36 total)")
    print("-" * 70)
    
    print("\nPhysiological variables:")
    phys_vars = [v for v in var_names if v in ['HR', 'BP', 'CO', 'TPR', 'SAO2', 'PVSAT', 'CVP', 'PCWP']]
    for var in phys_vars:
        idx = var_names.index(var)
        out_edges = mask[idx, :].sum()
        in_edges = mask[:, idx].sum()
        print(f"  {var:15s}: {out_edges:2d} outgoing, {in_edges:2d} incoming")
    
    print("\nVentilation variables:")
    vent_vars = [v for v in var_names if 'VENT' in v or v in ['INTUBATION', 'VENTTUBE']]
    for var in vent_vars:
        idx = var_names.index(var)
        out_edges = mask[idx, :].sum()
        in_edges = mask[:, idx].sum()
        print(f"  {var:15s}: {out_edges:2d} outgoing, {in_edges:2d} incoming")
    
    # Show bidirectional edges
    print("\n" + "-" * 70)
    print("BIDIRECTIONAL EDGES (to be oriented by Neural LP)")
    print("-" * 70)
    
    bidirectional = []
    for i in range(mask.shape[0]):
        for j in range(i+1, mask.shape[1]):
            if mask[i, j] == 1 and mask[j, i] == 1:
                bidirectional.append((i, j))
    
    for i, j in bidirectional:
        print(f"  {var_names[i]:20s} <-> {var_names[j]:20s}")
    
    print(f"\nTotal: {len(bidirectional)} bidirectional edge pairs")
    print("Neural LP will learn weights in both directions and select the stronger one.")
    
    print("\n" + "=" * 70)
    print("READY FOR TRAINING!")
    print("=" * 70)
    
    print("\nNext step:")
    print("  Prepare ALARM dataset and train the Neural LP model")
    print("  Expected: ~83% success rate (based on Step 4 results)")


if __name__ == "__main__":
    quick_demo()
