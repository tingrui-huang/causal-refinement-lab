"""
Generate metadata for Insurance dataset

This script generates the standardized metadata JSON file for the Insurance dataset.
Run this once to create the metadata file that will be used by the data loader.
"""

import sys
from pathlib import Path

# Add parent directory (project root) to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.metadata_generator import create_metadata_for_dataset


def main():
    """Generate Insurance metadata"""
    
    print("=" * 70)
    print("GENERATING INSURANCE METADATA")
    print("=" * 70)
    print()
    
    # Paths (relative to project root)
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'insurance' / 'insurance_data_10000.csv'
    output_path = project_root / 'data' / 'insurance' / 'metadata.json'
    
    # Check if data file exists
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Please run generate_insurance_data.py first to create the data file.")
        return
    
    # Generate metadata
    metadata = create_metadata_for_dataset(
        dataset_type='discrete',
        dataset_name='Insurance',
        data_path=str(data_path),
        output_path=str(output_path)
    )
    
    print()
    print("=" * 70)
    print("METADATA GENERATION COMPLETE")
    print("=" * 70)
    print(f"Metadata saved to: {output_path}")
    print()
    print("You can now use this metadata with:")
    print("  - CausalDataLoader")
    print("  - PriorBuilder")
    print("  - Training scripts")
    print()
    print("Next steps:")
    print("  1. Verify the metadata looks correct")
    print("  2. Update your training scripts to use the new data paths")
    print("  3. Run training with the new structure")


if __name__ == "__main__":
    main()

