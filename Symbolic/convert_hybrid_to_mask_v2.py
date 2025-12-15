"""
Convert Hybrid FCI-LLM Results to Neural LP Mask + Init Weights (v2)

KEY IMPROVEMENT (Gemini's Insight):
- Mask = FCI Skeleton (bidirectional for uncertain edges)
- Init Weights = LLM Prior (soft preference for suggested direction)

This allows Neural LP to CORRECT wrong LLM orientations based on data!

Previous Problem:
- If LLM said A->B, mask had M[A,B]=1, M[B,A]=0
- Neural LP could not learn B->A even if data strongly supported it
- Direction was LOCKED by mask

New Solution:
- FCI says: A - B (edge exists, direction uncertain)
- LLM says: A -> B (suggested direction)
- Mask: M[A,B]=1, M[B,A]=1 (BOTH directions allowed)
- Init: W[A,B]=0.5, W[B,A]=0.05 (LLM preference as soft prior)
- Neural LP can now choose the correct direction based on data!
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
    print("LOADING HYBRID FCI-LLM EDGES (v2)")
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


def create_mask_and_init_weights(df, var_names):
    """
    Create BOTH mask matrix (FCI skeleton) AND init weights (LLM prior).
    
    KEY LOGIC (Gemini's improvement):
    
    1. FCI Skeleton (Mask):
       - For ALL edges (regardless of LLM resolution), allow BOTH directions
       - This gives Neural LP the freedom to correct wrong orientations
       
    2. LLM Prior (Init Weights):
       - Strong weight (0.5) for LLM-suggested direction
       - Weak weight (0.05) for reverse direction
       - This biases learning toward LLM's suggestion but allows reversal
    
    Edge Type Handling:
    - 'directed' (FCI certain): Mask bidirectional, init asymmetric (0.5 vs 0.05)
    - 'llm_resolved' (FCI uncertain, LLM resolved): Mask bidirectional, init asymmetric
    - 'undirected' (FCI uncertain, LLM didn't resolve): Mask bidirectional, init symmetric (0.3 vs 0.3)
    - 'partial' (FCI partially oriented): Mask bidirectional, init asymmetric
    
    Args:
        df: DataFrame with edges
        var_names: List of variable names (ordered)
        
    Returns:
        Tuple of (mask, init_weights, stats)
    """
    print("\n" + "=" * 70)
    print("CREATING MASK (FCI SKELETON) + INIT WEIGHTS (LLM PRIOR)")
    print("=" * 70)
    
    n_vars = len(var_names)
    
    # Create variable name to index mapping
    var_to_idx = {var: idx for idx, var in enumerate(var_names)}
    
    # Initialize mask matrix (all zeros = forbidden)
    mask = np.zeros((n_vars, n_vars), dtype=int)
    
    # Initialize weights matrix (all zeros = no edge)
    init_weights = np.zeros((n_vars, n_vars), dtype=np.float32)
    
    print(f"\nInitialized {n_vars}x{n_vars} matrices")
    print(f"Processing {len(df)} edges...")
    
    # Track statistics
    stats = {
        'directed_fci': 0,           # FCI certain about direction
        'llm_resolved': 0,            # FCI uncertain, LLM resolved
        'undirected': 0,              # FCI uncertain, LLM didn't resolve
        'partial': 0,                 # FCI partially oriented
        'total_mask_entries': 0,      # Total trainable entries in mask
        'bidirectional_edges': 0      # Edges with both directions allowed
    }
    
    # Process each edge
    for idx, row in df.iterrows():
        source = row['source']
        target = row['target']
        edge_type = row['edge_type']
        
        # Get indices
        i = var_to_idx[source]  # source index
        j = var_to_idx[target]  # target index
        
        # CRITICAL: For ALL edge types, mask is BIDIRECTIONAL
        # This allows Neural LP to correct wrong orientations
        mask[i, j] = 1
        mask[j, i] = 1
        stats['total_mask_entries'] += 2
        stats['bidirectional_edges'] += 1
        
        # Init weights depend on edge type (LLM's confidence)
        if edge_type == 'directed':
            # FCI is certain: A -> B
            # Strong prior for this direction
            init_weights[i, j] = 0.5   # Forward (suggested)
            init_weights[j, i] = 0.05  # Reverse (discouraged but allowed)
            stats['directed_fci'] += 1
            
        elif edge_type == 'llm_resolved':
            # FCI uncertain (A - B), LLM resolved to A -> B
            # Moderate prior (LLM might be wrong!)
            init_weights[i, j] = 0.5   # Forward (LLM suggestion)
            init_weights[j, i] = 0.05  # Reverse (allow correction)
            stats['llm_resolved'] += 1
            
        elif edge_type == 'undirected':
            # FCI uncertain (A - B), LLM didn't resolve
            # Symmetric prior (no preference)
            init_weights[i, j] = 0.3   # Forward
            init_weights[j, i] = 0.3   # Reverse (equal weight)
            stats['undirected'] += 1
            
        elif edge_type == 'partial':
            # FCI partially oriented (e.g., A o-> B)
            # Moderate asymmetric prior
            init_weights[i, j] = 0.4   # Forward (partial evidence)
            init_weights[j, i] = 0.1   # Reverse (less likely)
            stats['partial'] += 1
        
        else:
            print(f"[WARNING] Unknown edge type: {edge_type}")
    
    print("\n" + "-" * 70)
    print("MASK & INIT WEIGHTS STATISTICS")
    print("-" * 70)
    print(f"Directed (FCI certain):   {stats['directed_fci']:3d} edges")
    print(f"LLM-resolved:             {stats['llm_resolved']:3d} edges (FCI uncertain)")
    print(f"Undirected:               {stats['undirected']:3d} edges (no LLM resolution)")
    print(f"Partial:                  {stats['partial']:3d} edges")
    print(f"\nTotal edges processed:         {len(df)}")
    print(f"Bidirectional edges in mask:   {stats['bidirectional_edges']}")
    print(f"Total trainable entries:       {stats['total_mask_entries']}")
    print(f"Total possible entries:        {n_vars * n_vars}")
    print(f"Sparsity: {(1 - stats['total_mask_entries'] / (n_vars * n_vars)):.2%}")
    
    print("\n" + "-" * 70)
    print("INIT WEIGHTS SUMMARY")
    print("-" * 70)
    print(f"Non-zero weights: {(init_weights > 0).sum()}")
    print(f"Max weight: {init_weights.max():.3f}")
    print(f"Min non-zero weight: {init_weights[init_weights > 0].min():.3f}")
    
    # Show weight distribution
    print("\nWeight distribution:")
    unique_weights = np.unique(init_weights[init_weights > 0])
    for w in unique_weights:
        count = (init_weights == w).sum()
        print(f"  Weight {w:.2f}: {count} entries")
    
    return mask, init_weights, stats


def visualize_comparison(mask, init_weights, var_names):
    """Print comparison of mask vs init weights."""
    print("\n" + "=" * 70)
    print("MASK vs INIT WEIGHTS COMPARISON")
    print("=" * 70)
    
    n_vars = len(var_names)
    
    print("\nSample edges showing mask and init weights:")
    print(f"{'Source':<15} {'Target':<15} {'Mask[i,j]':<10} {'Mask[j,i]':<10} {'Init[i,j]':<10} {'Init[j,i]':<10}")
    print("-" * 80)
    
    count = 0
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            if mask[i, j] == 1 or mask[j, i] == 1:
                print(f"{var_names[i]:<15} {var_names[j]:<15} "
                      f"{mask[i,j]:<10} {mask[j,i]:<10} "
                      f"{init_weights[i,j]:<10.2f} {init_weights[j,i]:<10.2f}")
                count += 1
                if count >= 15:
                    break
        if count >= 15:
            break
    
    print(f"\n... (showing first 15 edges)")
    
    # Find asymmetric init weights (LLM has preference)
    asymmetric = []
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            if mask[i, j] == 1 and mask[j, i] == 1:
                w_ij = init_weights[i, j]
                w_ji = init_weights[j, i]
                if abs(w_ij - w_ji) > 0.1:
                    asymmetric.append((var_names[i], var_names[j], w_ij, w_ji))
    
    print(f"\n\nEdges with LLM directional preference (asymmetric init): {len(asymmetric)}")
    if asymmetric:
        print("Sample edges where LLM suggested a direction:")
        for src, tgt, w_ij, w_ji in asymmetric[:10]:
            direction = f"{src} -> {tgt}" if w_ij > w_ji else f"{tgt} -> {src}"
            print(f"  {direction:<30} (weights: {max(w_ij, w_ji):.2f} vs {min(w_ij, w_ji):.2f})")


def save_results(mask, init_weights, var_names, output_dir='data'):
    """
    Save mask matrix, init weights, and variable names.
    
    Args:
        mask: Binary mask matrix (FCI skeleton)
        init_weights: Float init weights matrix (LLM prior)
        var_names: List of variable names
        output_dir: Output directory
    """
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save mask matrix
    mask_file = output_path / 'alarm_mask_skeleton.npy'
    np.save(mask_file, mask)
    print(f"\nSaved FCI skeleton mask to: {mask_file}")
    print(f"  Shape: {mask.shape}")
    print(f"  Dtype: {mask.dtype}")
    print(f"  Trainable entries: {mask.sum()}")
    
    # Save init weights
    init_weights_file = output_path / 'alarm_init_weights.npy'
    np.save(init_weights_file, init_weights)
    print(f"\nSaved LLM prior init weights to: {init_weights_file}")
    print(f"  Shape: {init_weights.shape}")
    print(f"  Dtype: {init_weights.dtype}")
    print(f"  Non-zero entries: {(init_weights > 0).sum()}")
    
    # Save variable names
    var_names_file = output_path / 'alarm_variables.txt'
    with open(var_names_file, 'w') as f:
        for i, var in enumerate(var_names):
            f.write(f"{i}\t{var}\n")
    print(f"\nSaved variable names to: {var_names_file}")
    
    # Save variable mapping (for easy lookup)
    var_mapping_file = output_path / 'alarm_var_mapping.npy'
    var_to_idx = {var: idx for idx, var in enumerate(var_names)}
    np.save(var_mapping_file, var_to_idx)
    print(f"Saved variable mapping to: {var_mapping_file}")
    
    print("\n" + "-" * 70)
    print("Files created:")
    print(f"  1. {mask_file} - FCI skeleton (bidirectional mask)")
    print(f"  2. {init_weights_file} - LLM prior (init weights)")
    print(f"  3. {var_names_file} - Variable names list")
    print(f"  4. {var_mapping_file} - Variable to index mapping")


def main():
    """Main function."""
    print("=" * 70)
    print("HYBRID FCI-LLM TO NEURAL LP (v2 - GEMINI'S IMPROVEMENT)")
    print("=" * 70)
    print("\nKEY INNOVATION:")
    print("  - Mask = FCI Skeleton (bidirectional, allows direction correction)")
    print("  - Init Weights = LLM Prior (soft preference, can be overridden)")
    print("\nInput:   edges_Hybrid_FCI_LLM_20251207_230956.csv")
    print("Output:  alarm_mask_skeleton.npy + alarm_init_weights.npy")
    
    # Path to the CSV file
    csv_path = Path('../outputs/alarm/edges_Hybrid_FCI_LLM_20251207_230956.csv')
    
    # Also check in Symbolic/data
    if not csv_path.exists():
        csv_path = Path('data/edges_Hybrid_FCI_LLM_20251207_230956.csv')
    
    # Also check absolute path from project root
    if not csv_path.exists():
        csv_path = Path(__file__).parent.parent / 'outputs' / 'alarm' / 'edges_Hybrid_FCI_LLM_20251207_230956.csv'
    
    if not csv_path.exists():
        print(f"\n[ERROR] Could not find CSV file")
        return
    
    # Load edges
    df = load_hybrid_edges(csv_path)
    
    # Extract variable names
    var_names = extract_variable_names(df)
    
    # Create mask and init weights
    mask, init_weights, stats = create_mask_and_init_weights(df, var_names)
    
    # Visualize comparison
    visualize_comparison(mask, init_weights, var_names)
    
    # Save results
    save_results(mask, init_weights, var_names, output_dir='Symbolic/data')
    
    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE!")
    print("=" * 70)
    
    print("\nNext steps:")
    print("  1. Modify neural_lp.py to accept init_weights parameter")
    print("  2. Load both mask and init_weights in training script")
    print("  3. Train with two-phase approach")
    print("  4. Neural LP can now CORRECT wrong LLM orientations!")
    
    print("\nKey Advantage:")
    print("  If LLM said A->B but data strongly supports B->A,")
    print("  Neural LP will learn W[B,A] > W[A,B] and DAG constraint")
    print("  will eliminate the weaker direction, correcting the error!")


if __name__ == "__main__":
    main()
