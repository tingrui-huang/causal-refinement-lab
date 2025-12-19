"""
Generate metadata for Tuebingen dataset

This script generates the standardized metadata JSON file for Tuebingen cause-effect pairs.
Each pair is treated as a separate dataset with 2 continuous variables.

Usage:
    python generate_tuebingen_metadata.py <pair_id> <data_csv_path>
    
Example:
    python generate_tuebingen_metadata.py pair1 data/tuebingen/pair0001.csv
"""

import sys
from pathlib import Path
import argparse

# Add parent directory (project root) to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.metadata_generator import create_metadata_for_dataset


def main():
    """Generate Tuebingen metadata"""
    
    parser = argparse.ArgumentParser(description='Generate metadata for Tuebingen dataset')
    parser.add_argument('pair_id', type=str, help='Pair identifier (e.g., pair1, pair0001)')
    parser.add_argument('data_path', type=str, help='Path to CSV file with pair data')
    parser.add_argument('--var1-name', type=str, default='X', help='Name of first variable (default: X)')
    parser.add_argument('--var2-name', type=str, default='Y', help='Name of second variable (default: Y)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"GENERATING TUEBINGEN METADATA FOR {args.pair_id.upper()}")
    print("=" * 70)
    print()
    
    # Paths
    data_path = Path(args.data_path)
    output_path = data_path.parent / f'metadata_{args.pair_id}.json'
    
    # Check if data file exists
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Please ensure the Tuebingen data file is in the correct location.")
        return
    
    # Variable names
    variable_names = [args.var1_name, args.var2_name]
    
    # Generate metadata
    dataset_name = f"Tuebingen_{args.pair_id}"
    metadata = create_metadata_for_dataset(
        dataset_type='continuous',
        dataset_name=dataset_name,
        data_path=str(data_path),
        output_path=str(output_path),
        variable_names=variable_names
    )
    
    print()
    print("=" * 70)
    print("METADATA GENERATION COMPLETE")
    print("=" * 70)
    print(f"Metadata saved to: {output_path}")
    print()
    print("You can now use this metadata with:")
    print("  - CausalDataLoader")
    print("  - PriorBuilder (with continuous data support)")
    print("  - Training scripts")
    print()
    print("Note: Tuebingen pairs are continuous data.")
    print("      You may need to adapt the training pipeline for continuous variables.")


if __name__ == "__main__":
    main()
