"""
Convert Hybrid FCI-LLM Results to Neural LP Mask Matrix

This script processes the hybrid causal discovery results and creates a mask matrix
that defines which edges the Neural LP model is allowed to learn.

The mask matrix M_Mask is a 37x37 binary matrix where:
- M_Mask[i,j] = 1 means the edge from variable i to variable j is trainable
- M_Mask[i,j] = 0 means the edge is forbidden (masked out)

Edge type handling:
- directed: Only source -> target is trainable
- undirected: Both source <-> target are trainable
- partial: Both source <-> target are trainable
- llm_resolved: Only source -> target is trainable (LLM resolved the direction)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_hybrid_edges(csv_path):
    """
    Load the hybrid FCI-LLM edges from CSV.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame with columns: source, target, edge_type, status
    """
    print("=" * 70)
    print("LOADING HYBRID FCI-LLM EDGES")
    print("=" * 70)
    
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} edges from {csv_path}")
    print(f"\nColumns: {list(df.columns)}")
    
    # Filter only accepted edges
    df_accepted = df[df['status'] == 'accepted'].copy()
    print(f"Accepted edges: {len(df_accepted)}")
    
    # Count edge types
    print("\nEdge type distribution:")
    for edge_type, count in df_accepted['edge_type'].value_counts().items():
        print(f"  {edge_type}: {count}")
    
    return df_accepted


def extract_variable_names(df):
    """
    Extract unique variable names from source and target columns.
    
    Args:
        df: DataFrame with 'source' and 'target' columns
        
    Returns:
        Sorted list of unique variable names
    """
    print("\n" + "=" * 70)
    print("EXTRACTING VARIABLE NAMES")
    print("=" * 70)
    
    # Get all unique variables
    sources = set(df['source'].unique())
    targets = set(df['target'].unique())
    all_vars = sorted(sources.union(targets))
    
    print(f"\nFound {len(all_vars)} unique variables:")
    for i, var in enumerate(all_vars, 1):
        print(f"  {i:2d}. {var}")
    
    return all_vars


def create_mask_matrix(df, var_names):
    """
    Create the mask matrix from hybrid edges.
    
    Args:
        df: DataFrame with edges
        var_names: List of variable names (ordered)
        
    Returns:
        numpy array of shape (n_vars, n_vars) with binary mask
    """
    print("\n" + "=" * 70)
    print("CREATING MASK MATRIX")
    print("=" * 70)
    
    n_vars = len(var_names)
    
    # Create variable name to index mapping
    var_to_idx = {var: idx for idx, var in enumerate(var_names)}
    
    # Initialize mask matrix with zeros (all edges forbidden)
    mask = np.zeros((n_vars, n_vars), dtype=int)
    
    print(f"\nInitialized {n_vars}x{n_vars} mask matrix (all zeros)")
    print(f"Processing {len(df)} edges...")
    
    # Track statistics
    stats = {
        'directed': 0,
        'undirected': 0,
        'partial': 0,
        'llm_resolved': 0,
        'total_entries': 0
    }
    
    # Process each edge
    for idx, row in df.iterrows():
        source = row['source']
        target = row['target']
        edge_type = row['edge_type']
        
        # Get indices
        i = var_to_idx[source]
        j = var_to_idx[target]
        
        # Set mask based on edge type
        if edge_type == 'directed':
            # Directed edge: only source -> target
            mask[i, j] = 1
            stats['directed'] += 1
            stats['total_entries'] += 1
            
        elif edge_type == 'undirected':
            # Undirected edge: both source <-> target
            mask[i, j] = 1
            mask[j, i] = 1
            stats['undirected'] += 1
            stats['total_entries'] += 2
            
        elif edge_type == 'partial':
            # Partially oriented: both directions trainable
            mask[i, j] = 1
            mask[j, i] = 1
            stats['partial'] += 1
            stats['total_entries'] += 2
            
        elif edge_type == 'llm_resolved':
            # LLM resolved direction: only source -> target
            mask[i, j] = 1
            stats['llm_resolved'] += 1
            stats['total_entries'] += 1
        
        else:
            print(f"[WARNING] Unknown edge type: {edge_type}")
    
    print("\n" + "-" * 70)
    print("MASK MATRIX STATISTICS")
    print("-" * 70)
    print(f"Directed edges:       {stats['directed']:3d} (source -> target only)")
    print(f"Undirected edges:     {stats['undirected']:3d} (source <-> target)")
    print(f"Partial edges:        {stats['partial']:3d} (source <-> target)")
    print(f"LLM-resolved edges:   {stats['llm_resolved']:3d} (source -> target only)")
    print(f"\nTotal edges processed:     {len(df)}")
    print(f"Total trainable entries:   {stats['total_entries']}")
    print(f"Total possible entries:    {n_vars * n_vars}")
    print(f"Sparsity: {(1 - stats['total_entries'] / (n_vars * n_vars)):.2%}")
    
    return mask, stats


def visualize_mask_summary(mask, var_names):
    """Print summary of mask matrix."""
    print("\n" + "=" * 70)
    print("MASK MATRIX SUMMARY")
    print("=" * 70)
    
    n_vars = len(var_names)
    
    # Count trainable edges per variable
    print("\nTrainable outgoing edges per variable (row sums):")
    row_sums = mask.sum(axis=1)
    for i, var in enumerate(var_names):
        if row_sums[i] > 0:
            print(f"  {var:20s}: {row_sums[i]:2d} outgoing edges")
    
    print("\nTrainable incoming edges per variable (column sums):")
    col_sums = mask.sum(axis=0)
    for i, var in enumerate(var_names):
        if col_sums[i] > 0:
            print(f"  {var:20s}: {col_sums[i]:2d} incoming edges")
    
    # Find bidirectional edges
    bidirectional = []
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            if mask[i, j] == 1 and mask[j, i] == 1:
                bidirectional.append((var_names[i], var_names[j]))
    
    print(f"\nBidirectional edges (undirected or partial): {len(bidirectional)}")
    if bidirectional:
        print("Sample bidirectional edges:")
        for src, tgt in bidirectional[:10]:
            print(f"  {src} <-> {tgt}")
        if len(bidirectional) > 10:
            print(f"  ... and {len(bidirectional) - 10} more")


def save_mask_matrix(mask, var_names, output_dir='data'):
    """
    Save mask matrix and variable names.
    
    Args:
        mask: Binary mask matrix
        var_names: List of variable names
        output_dir: Output directory
    """
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save mask matrix
    mask_file = output_path / 'alarm_mask_37x37.npy'
    np.save(mask_file, mask)
    print(f"\nSaved mask matrix to: {mask_file}")
    print(f"  Shape: {mask.shape}")
    print(f"  Dtype: {mask.dtype}")
    print(f"  Trainable entries: {mask.sum()}")
    
    # Save variable names
    var_names_file = output_path / 'alarm_variables_37.txt'
    with open(var_names_file, 'w') as f:
        for i, var in enumerate(var_names):
            f.write(f"{i}\t{var}\n")
    print(f"\nSaved variable names to: {var_names_file}")
    
    # Save variable mapping (for easy lookup)
    var_mapping_file = output_path / 'alarm_var_mapping_37.npy'
    var_to_idx = {var: idx for idx, var in enumerate(var_names)}
    np.save(var_mapping_file, var_to_idx)
    print(f"Saved variable mapping to: {var_mapping_file}")
    
    print("\n" + "-" * 70)
    print("Files created:")
    print(f"  1. {mask_file} - Binary mask matrix (37x37)")
    print(f"  2. {var_names_file} - Variable names list")
    print(f"  3. {var_mapping_file} - Variable to index mapping")


def main():
    """Main function."""
    print("=" * 70)
    print("HYBRID FCI-LLM TO NEURAL LP MASK CONVERTER")
    print("=" * 70)
    print("\nPurpose: Convert hybrid causal discovery results to trainable mask")
    print("Input:   edges_Hybrid_FCI_LLM_20251207_230956.csv")
    print("Output:  alarm_mask_37x37.npy (37x37 binary mask matrix)")
    
    # Path to the CSV file (relative to project root)
    csv_path = Path('../outputs/alarm/edges_Hybrid_FCI_LLM_20251207_230956.csv')
    
    # Also check in Symbolic/data (in case it was moved there)
    if not csv_path.exists():
        csv_path = Path('data/edges_Hybrid_FCI_LLM_20251207_230956.csv')
    
    # Also check absolute path from project root
    if not csv_path.exists():
        csv_path = Path(__file__).parent.parent / 'outputs' / 'alarm' / 'edges_Hybrid_FCI_LLM_20251207_230956.csv'
    
    if not csv_path.exists():
        print(f"\n[ERROR] Could not find CSV file at any of these locations:")
        print(f"  - ../outputs/alarm/edges_Hybrid_FCI_LLM_20251207_230956.csv")
        print(f"  - data/edges_Hybrid_FCI_LLM_20251207_230956.csv")
        print(f"  - {csv_path.absolute()}")
        return
    
    # Load edges
    df = load_hybrid_edges(csv_path)
    
    # Extract variable names
    var_names = extract_variable_names(df)
    
    # Verify we have 37 variables
    if len(var_names) != 37:
        print(f"\n[WARNING] Expected 37 variables but found {len(var_names)}")
        print("This is OK, continuing with actual variable count...")
    
    # Create mask matrix
    mask, stats = create_mask_matrix(df, var_names)
    
    # Visualize summary
    visualize_mask_summary(mask, var_names)
    
    # Save results
    save_mask_matrix(mask, var_names, output_dir='Symbolic/data')
    
    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE!")
    print("=" * 70)
    
    print("\nNext steps:")
    print("  1. Load the mask matrix: mask = np.load('Symbolic/data/alarm_mask_37x37.npy')")
    print("  2. Initialize Neural LP model with this mask")
    print("  3. Train the model using two-phase training (step4_refinement_final.py)")
    print("  4. Compare learned adjacency matrix against ground truth")
    
    print("\nMask Matrix Info:")
    print(f"  Shape: {mask.shape}")
    print(f"  Trainable edges: {mask.sum()}")
    print(f"  Forbidden edges: {mask.shape[0] * mask.shape[1] - mask.sum()}")
    print(f"  Sparsity: {(1 - mask.sum() / (mask.shape[0] * mask.shape[1])):.2%}")


if __name__ == "__main__":
    main()
