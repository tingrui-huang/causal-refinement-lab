"""
Generate Win95pts dataset for causal discovery

The Win95pts dataset is a Bayesian network for printer troubleshooting
in Windows 95 operating system.

Data source: bnlearn library (preferred) or BIF file
Ground truth: 112 edges for printer diagnosis

Reference: Microsoft troubleshooting network
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def load_win95pts_from_bnlearn():
    """
    Try to load Win95pts dataset from bnlearn library
    
    Returns:
        df: DataFrame with samples
        edges: List of ground truth edges
        variables: List of variable names
    """
    try:
        import bnlearn as bn
    except ImportError:
        print("ERROR: bnlearn not installed")
        return None, None, None
    
    print("=" * 80)
    print("LOADING WIN95PTS DATASET FROM BNLEARN")
    print("=" * 80)
    
    try:
        # Try to load data
        print("\n[1/3] Loading Win95pts dataset...")
        df = bn.import_example('win95pts')
        
        # Try to load DAG
        print("[2/3] Loading ground truth DAG...")
        dag = bn.import_DAG('win95pts')
        
        # Extract edges
        if isinstance(dag, dict) and 'model' in dag:
            edges = list(dag['model'].edges())
        else:
            edges = list(dag.edges())
        
        variables = list(df.columns)
        
        print(f"[3/3] Dataset loaded successfully!")
        print(f"  Samples: {len(df)}")
        print(f"  Variables: {len(variables)}")
        print(f"  Ground truth edges: {len(edges)}")
        
        return df, edges, variables
        
    except Exception as e:
        print(f"[INFO] Failed to load from bnlearn: {e}")
        return None, None, None


def load_win95pts_from_bif(bif_path):
    """
    Load Win95pts dataset from BIF file and generate samples
    
    Args:
        bif_path: Path to win95pts.bif file
    
    Returns:
        df: DataFrame with samples
        edges: List of ground truth edges
        variables: List of variable names
    """
    try:
        from pgmpy.readwrite import BIFReader
        from pgmpy.sampling import BayesianModelSampling
    except ImportError:
        print("ERROR: pgmpy not installed. Install with: pip install pgmpy")
        return None, None, None
    
    print("=" * 80)
    print("LOADING WIN95PTS DATASET FROM BIF")
    print("=" * 80)
    
    # Read BIF file
    print(f"\n[1/3] Reading BIF file: {bif_path}")
    reader = BIFReader(bif_path)
    model = reader.get_model()
    
    # Get structure
    edges = list(model.edges())
    variables = list(model.nodes())
    
    print(f"[2/3] Structure loaded")
    print(f"  Variables: {len(variables)}")
    print(f"  Edges: {len(edges)}")
    
    # Generate samples
    print(f"[3/3] Generating samples...")
    sampler = BayesianModelSampling(model)
    df = sampler.forward_sample(size=10000, show_progress=False)
    
    print(f"  Generated {len(df)} samples")
    
    return df, edges, variables


def convert_to_onehot(df, output_path):
    """Convert categorical data to one-hot encoding"""
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
    """Convert categorical data to numeric codes for FCI"""
    print("\n" + "=" * 80)
    print("CONVERTING TO VARIABLE-LEVEL (FOR FCI)")
    print("=" * 80)
    
    # Convert categorical to numeric codes
    df_numeric = df.copy()
    for col in df.columns:
        # Convert to categorical codes (0, 1, 2, ...)
        df_numeric[col] = pd.Categorical(df[col]).codes
    
    # Save
    df_numeric.to_csv(output_path, index=False)
    print(f"\n[OK] Saved variable-level data: {output_path}")
    print(f"  Shape: {df_numeric.shape}")


def create_ground_truth_file(edges, output_path):
    """Create ground truth edge list file"""
    print("\n" + "=" * 80)
    print("CREATING GROUND TRUTH FILE")
    print("=" * 80)
    
    with open(output_path, 'w') as f:
        f.write("# Win95pts printer troubleshooting network ground truth\n")
        f.write("# Source: Microsoft Windows 95 printer troubleshooting\n")
        f.write(f"# {len(edges)} edges for printer diagnosis\n\n")
        
        for source, target in sorted(edges):
            f.write(f"{source} -> {target}\n")
    
    print(f"[OK] Saved ground truth: {output_path}")
    print(f"  {len(edges)} edges")
    
    # Print first 10 edges for verification
    print("\nFirst 10 ground truth edges:")
    for i, (source, target) in enumerate(sorted(edges)[:10], 1):
        print(f"  {i:2d}. {source:25s} -> {target}")


def create_metadata(variables, state_mappings, n_samples, output_path):
    """Create metadata.json file"""
    print("\n" + "=" * 80)
    print("CREATING METADATA")
    print("=" * 80)
    
    metadata = {
        "dataset_name": "win95pts",
        "data_format": "one_hot_csv",
        "n_variables": len(variables),
        "variable_names": sorted(variables),
        "state_mappings": state_mappings,
        "n_samples": n_samples,
        "domain": "computer_troubleshooting",
        "source": "Microsoft Windows 95 printer troubleshooting network",
        "description": "Win95pts network for diagnosing printer problems in Windows 95",
        "reference": "Microsoft troubleshooting expert system"
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
    print(f"  Domain: Computer troubleshooting (printer diagnosis)")


def main():
    """Main function to generate all Win95pts dataset files"""
    
    print("\n" + "=" * 80)
    print("GENERATING WIN95PTS DATASET")
    print("=" * 80)
    print("Purpose: Printer troubleshooting benchmark for causal discovery")
    print("Strategy: 76 variables, discrete states, expert-validated DAG")
    print("=" * 80)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'win95pts'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to load from bnlearn first (preferred)
    df, edges, variables = load_win95pts_from_bnlearn()
    
    # If bnlearn fails, try from BIF file
    if df is None:
        print("\n[INFO] Trying to load from BIF file...")
        bif_path = data_dir / 'win95pts.bif'
        
        if not bif_path.exists():
            print("\n" + "=" * 80)
            print("ERROR: Win95pts dataset not available")
            print("=" * 80)
            print("\nPlease download win95pts.bif from:")
            print("  https://www.bnlearn.com/bnrepository/")
            print("\nAnd place it in:")
            print(f"  {bif_path}")
            print("\nThen run this script again.")
            print("=" * 80)
            return
        
        df, edges, variables = load_win95pts_from_bif(bif_path)
    
    if df is None:
        print("\n[ERROR] Failed to load Win95pts dataset")
        return
    
    # 1. Generate one-hot CSV for neural network
    print("\n" + "=" * 80)
    print("STEP 1: GENERATE ONE-HOT CSV (FOR NEURAL NETWORK)")
    print("=" * 80)
    onehot_path = data_dir / 'win95pts_data.csv'
    state_mappings = convert_to_onehot(df, onehot_path)
    
    # 2. Generate variable-level CSV for FCI
    print("\n" + "=" * 80)
    print("STEP 2: GENERATE VARIABLE-LEVEL CSV (FOR FCI)")
    print("=" * 80)
    variable_path = base_dir.parent / 'win95pts_data_variable.csv'
    convert_to_variable_level(df, variable_path)
    
    # 3. Generate ground truth edge list
    print("\n" + "=" * 80)
    print("STEP 3: GENERATE GROUND TRUTH EDGE LIST")
    print("=" * 80)
    gt_path = data_dir / 'win95pts_ground_truth.txt'
    create_ground_truth_file(edges, gt_path)
    
    # 4. Generate metadata.json
    print("\n" + "=" * 80)
    print("STEP 4: GENERATE METADATA")
    print("=" * 80)
    metadata_path = data_dir / 'metadata.json'
    create_metadata(variables, state_mappings, len(df), metadata_path)
    
    print("\n" + "=" * 80)
    print("[SUCCESS] WIN95PTS DATASET GENERATION COMPLETE!")
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
    print("   DATASET = 'win95pts'")
    print("   LLM_MODEL = None  # Test GSB framework")
    print("")
    print("2. Run pipeline:")
    print("   cd Neuro-Symbolic-Reasoning")
    print("   python run_pipeline.py")
    print("=" * 80)


if __name__ == "__main__":
    main()

