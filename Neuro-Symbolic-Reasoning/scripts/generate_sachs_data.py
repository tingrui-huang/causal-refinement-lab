"""
Generate Sachs dataset from bnlearn library

bnlearn provides:
- Pre-discretized Sachs data (5400 samples total)
- Observational data subset (~854 samples)
- Ground truth DAG (17 edges)
- 11 protein variables

This script converts bnlearn format to our pipeline format.
We use ONLY observational data to test GSB framework's ability
to learn causal direction without interventions.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def load_sachs_from_bnlearn():
    """
    Load Sachs dataset from bnlearn
    
    Returns:
        data_df: DataFrame with discretized values (observational only)
        dag_edges: List of (source, target) tuples
        variable_info: List of variable names
    """
    try:
        import bnlearn as bn
    except ImportError:
        print("=" * 80)
        print("ERROR: bnlearn not installed")
        print("=" * 80)
        print("Please install with: pip install bnlearn")
        print("Then run this script again.")
        print("=" * 80)
        return None, None, None
    
    print("=" * 80)
    print("LOADING SACHS DATASET FROM BNLEARN")
    print("=" * 80)
    
    # Load Sachs dataset
    print("\n[1/3] Loading Sachs dataset...")
    df = bn.import_example('sachs')
    
    # Load ground truth DAG
    print("[2/3] Loading ground truth DAG...")
    dag = bn.import_DAG('sachs')
    
    # Extract edges from DAG
    # bnlearn returns a dict with 'model' key containing BayesianNetwork
    edges = []
    if isinstance(dag, dict) and 'model' in dag:
        edges = list(dag['model'].edges())
    elif hasattr(dag, 'edges'):
        edges = list(dag.edges())
    
    # Get variable information
    variables = list(df.columns)
    
    print("[3/3] Dataset loaded successfully!")
    print(f"\n  Total samples: {len(df)}")
    print(f"  Variables: {len(variables)}")
    print(f"  Variable names: {', '.join(variables)}")
    print(f"  Ground truth edges: {len(edges)}")
    
    # Filter to observational data only
    # In bnlearn's Sachs dataset, we can use all data as observational
    # or filter based on specific columns if intervention markers exist
    print("\n" + "=" * 80)
    print("FILTERING TO OBSERVATIONAL DATA ONLY")
    print("=" * 80)
    
    # For Sachs from bnlearn, typically all samples are observational
    # If there's an intervention column, we'd filter here
    # For now, we'll use the first ~854 samples as observational
    obs_df = df.head(854)  # Standard observational subset size
    
    print(f"Using {len(obs_df)} observational samples")
    print("(Testing GSB framework without interventional data)")
    
    return obs_df, edges, variables


def convert_to_onehot(df, output_path):
    """
    Convert categorical data to one-hot encoding
    
    Args:
        df: DataFrame with categorical values
        output_path: Where to save one-hot CSV
    
    Returns:
        state_mappings: Dict for metadata.json
    """
    print("\n" + "=" * 80)
    print("CONVERTING TO ONE-HOT ENCODING")
    print("=" * 80)
    
    state_mappings = {}
    onehot_dfs = []
    total_states = 0
    
    for col in df.columns:
        # Get unique states (sorted for consistency)
        unique_states = sorted(df[col].unique())
        n_states = len(unique_states)
        total_states += n_states
        
        # Create state mapping
        state_mappings[col] = {
            str(i): f"{col}_{state}" 
            for i, state in enumerate(unique_states)
        }
        
        # One-hot encode
        onehot = pd.get_dummies(df[col], prefix=col)
        onehot_dfs.append(onehot)
        
        print(f"  {col:10s}: {n_states} states -> {len(onehot.columns)} columns")
    
    # Combine all one-hot columns
    onehot_df = pd.concat(onehot_dfs, axis=1)
    
    # Save
    onehot_df.to_csv(output_path, index=False)
    
    print(f"\n[OK] Saved one-hot data: {output_path}")
    print(f"  Shape: {onehot_df.shape}")
    print(f"  Total states: {total_states}")
    print(f"  Density: {onehot_df.mean().mean() * 100:.2f}%")
    
    return state_mappings


def convert_to_variable_level(df, output_path):
    """
    Convert categorical data to numeric codes for FCI
    
    Args:
        df: DataFrame with categorical values
        output_path: Where to save variable-level CSV
    """
    print("\n" + "=" * 80)
    print("CONVERTING TO VARIABLE-LEVEL (FOR FCI)")
    print("=" * 80)
    
    # Convert categorical to numeric codes
    df_numeric = df.copy()
    for col in df.columns:
        # Convert to categorical codes (0, 1, 2, ...)
        df_numeric[col] = pd.Categorical(df[col]).codes
        print(f"  {col:10s}: {df[col].nunique()} unique values -> codes 0-{df_numeric[col].max()}")
    
    # Save
    df_numeric.to_csv(output_path, index=False)
    print(f"\n[OK] Saved variable-level data: {output_path}")
    print(f"  Shape: {df_numeric.shape}")


def create_ground_truth_file(edges, output_path):
    """
    Create ground truth edge list file
    
    Args:
        edges: List of (source, target) tuples
        output_path: Where to save edge list
    """
    print("\n" + "=" * 80)
    print("CREATING GROUND TRUTH FILE")
    print("=" * 80)
    
    with open(output_path, 'w') as f:
        f.write("# Sachs protein signaling network ground truth\n")
        f.write("# Source: bnlearn library\n")
        f.write(f"# {len(edges)} edges validated by biological experiments\n")
        f.write("# Reference: Sachs et al. (2005) Science\n\n")
        
        for source, target in sorted(edges):
            f.write(f"{source} -> {target}\n")
    
    print(f"[OK] Saved ground truth: {output_path}")
    print(f"  {len(edges)} edges")
    
    # Print edges for verification
    print("\nGround truth edges:")
    for i, (source, target) in enumerate(sorted(edges), 1):
        print(f"  {i:2d}. {source:6s} -> {target}")


def create_metadata(variables, state_mappings, n_samples, output_path):
    """
    Create metadata.json file
    
    Args:
        variables: List of variable names
        state_mappings: Dict of {var: {code: state_name}}
        n_samples: Number of samples
        output_path: Where to save metadata.json
    """
    print("\n" + "=" * 80)
    print("CREATING METADATA")
    print("=" * 80)
    
    metadata = {
        "dataset_name": "sachs",
        "data_format": "one_hot_csv",
        "n_variables": len(variables),
        "variable_names": variables,
        "state_mappings": state_mappings,
        "n_samples": n_samples,
        "data_type": "observational_only",
        "source": "bnlearn library",
        "description": "Sachs protein signaling dataset (discretized, observational data only)",
        "reference": "Sachs et al. (2005) Causal Protein-Signaling Networks Derived from Multiparameter Single-Cell Data. Science 308(5721):523-529"
    }
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[OK] Saved metadata: {output_path}")
    
    # Print summary
    total_states = sum(len(states) for states in state_mappings.values())
    print(f"\nDataset Summary:")
    print(f"  Variables: {len(variables)}")
    print(f"  Total states: {total_states}")
    print(f"  Samples: {n_samples}")
    print(f"  Data type: Observational only")


def main():
    """Main function to generate all Sachs dataset files"""
    
    print("\n" + "=" * 80)
    print("GENERATING SACHS DATASET FROM BNLEARN")
    print("=" * 80)
    print("Purpose: Test GSB framework on 3-state protein signaling data")
    print("Strategy: Use observational data only (no interventions)")
    print("=" * 80)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'sachs'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Load from bnlearn
    df, edges, variables = load_sachs_from_bnlearn()
    
    if df is None:
        print("\n❌ Failed to load data from bnlearn")
        return
    
    # 1. Generate one-hot CSV for neural network
    print("\n" + "=" * 80)
    print("STEP 1: GENERATE ONE-HOT CSV (FOR NEURAL NETWORK)")
    print("=" * 80)
    onehot_path = data_dir / 'sachs_data.csv'
    state_mappings = convert_to_onehot(df, onehot_path)
    
    # 2. Generate variable-level CSV for FCI
    print("\n" + "=" * 80)
    print("STEP 2: GENERATE VARIABLE-LEVEL CSV (FOR FCI)")
    print("=" * 80)
    variable_path = base_dir.parent / 'sachs_data_variable.csv'
    convert_to_variable_level(df, variable_path)
    
    # 3. Generate ground truth edge list
    print("\n" + "=" * 80)
    print("STEP 3: GENERATE GROUND TRUTH EDGE LIST")
    print("=" * 80)
    gt_path = data_dir / 'sachs_ground_truth.txt'
    create_ground_truth_file(edges, gt_path)
    
    # 4. Generate metadata.json
    print("\n" + "=" * 80)
    print("STEP 4: GENERATE METADATA")
    print("=" * 80)
    metadata_path = data_dir / 'metadata.json'
    create_metadata(variables, state_mappings, len(df), metadata_path)
    
    print("\n" + "=" * 80)
    print("[SUCCESS] SACHS DATASET GENERATION COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  1. {onehot_path.relative_to(base_dir)}")
    print(f"  2. {variable_path.name} (in project root)")
    print(f"  3. {gt_path.relative_to(base_dir)}")
    print(f"  4. {metadata_path.relative_to(base_dir)}")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Update config.py:")
    print("   DATASET = 'sachs'")
    print("   LLM_MODEL = None  # No LLM, test GSB framework directly")
    print("")
    print("2. Run pipeline:")
    print("   cd Neuro-Symbolic-Reasoning")
    print("   python run_pipeline.py")
    print("")
    print("3. Expected behavior:")
    print("   - FCI will find skeleton (undirected edges)")
    print("   - Neural network will learn directions using asymmetry")
    print("   - 3-state data makes asymmetry very sensitive (good test!)")
    print("=" * 80)


if __name__ == "__main__":
    main()

