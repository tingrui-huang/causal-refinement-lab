"""
Generate Insurance dataset from BIF file

This script:
1. Loads the insurance.bif file using pgmpy
2. Samples observational data from the Bayesian Network
3. Saves the data as CSV in one-hot encoded format
4. Generates metadata.json for the dataset

Usage:
    python scripts/generate_insurance_data.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from pgmpy.readwrite import BIFReader
    from pgmpy.sampling import BayesianModelSampling
except ImportError:
    print("ERROR: pgmpy is not installed.")
    print("Please install it with: pip install pgmpy")
    sys.exit(1)

from modules.metadata_generator import create_metadata_for_dataset


def load_bif_and_sample(bif_path: str, n_samples: int = 10000) -> pd.DataFrame:
    """
    Load BIF file and sample data from the Bayesian Network
    
    Args:
        bif_path: Path to .bif file
        n_samples: Number of samples to generate
    
    Returns:
        DataFrame with sampled data (raw categorical values)
    """
    print(f"Loading BIF file: {bif_path}")
    
    # Read BIF file
    reader = BIFReader(bif_path)
    model = reader.get_model()
    
    print(f"Model loaded successfully")
    print(f"  Variables: {len(model.nodes())}")
    print(f"  Edges: {len(model.edges())}")
    print(f"  Variables: {sorted(model.nodes())}")
    
    # Sample from the model
    print(f"\nSampling {n_samples} observations...")
    sampler = BayesianModelSampling(model)
    samples = sampler.forward_sample(size=n_samples)
    
    print(f"Sampling complete: {samples.shape}")
    
    return samples


def convert_to_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical data to one-hot encoded format
    
    Args:
        df: DataFrame with categorical values
    
    Returns:
        DataFrame with one-hot encoded columns
    """
    print("\nConverting to one-hot encoding...")
    
    onehot_dfs = []
    
    for col in df.columns:
        # Get unique values (states) for this variable
        unique_values = sorted(df[col].unique())
        
        # Create one-hot columns for this variable
        for value in unique_values:
            col_name = f"{col}_{value}"
            onehot_dfs.append(pd.DataFrame({
                col_name: (df[col] == value).astype(int)
            }))
    
    # Concatenate all one-hot columns
    onehot_df = pd.concat(onehot_dfs, axis=1)
    
    print(f"One-hot encoding complete: {onehot_df.shape}")
    print(f"  Original variables: {len(df.columns)}")
    print(f"  Total states (one-hot columns): {len(onehot_df.columns)}")
    
    return onehot_df


def main():
    """Generate Insurance dataset and metadata"""
    
    print("=" * 70)
    print("GENERATING INSURANCE DATASET")
    print("=" * 70)
    print()
    
    # Paths
    project_root = Path(__file__).parent.parent
    bif_path = project_root / 'data' / 'insurance' / 'insurance.bif'
    csv_output_path = project_root / 'data' / 'insurance' / 'insurance_data_10000.csv'
    metadata_output_path = project_root / 'data' / 'insurance' / 'metadata.json'
    
    # Check if BIF file exists
    if not bif_path.exists():
        print(f"ERROR: BIF file not found: {bif_path}")
        return
    
    # Step 1: Load BIF and sample data
    raw_samples = load_bif_and_sample(str(bif_path), n_samples=10000)
    
    # Step 2: Convert to one-hot encoding
    onehot_samples = convert_to_onehot(raw_samples)
    
    # Step 3: Save CSV
    print(f"\nSaving CSV to: {csv_output_path}")
    onehot_samples.to_csv(csv_output_path, index=False)
    print("CSV saved successfully")
    
    # Step 4: Generate metadata
    print("\n" + "=" * 70)
    print("GENERATING METADATA")
    print("=" * 70)
    print()
    
    metadata = create_metadata_for_dataset(
        dataset_type='discrete',
        dataset_name='Insurance',
        data_path=str(csv_output_path),
        output_path=str(metadata_output_path)
    )
    
    print()
    print("=" * 70)
    print("INSURANCE DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"CSV saved to: {csv_output_path}")
    print(f"Metadata saved to: {metadata_output_path}")
    print()
    print("Dataset statistics:")
    print(f"  Samples: {len(onehot_samples)}")
    print(f"  Variables: {len(raw_samples.columns)}")
    print(f"  Total states: {len(onehot_samples.columns)}")
    print()
    print("You can now use this dataset for training!")


if __name__ == "__main__":
    main()

