"""
Convert one-hot encoded data back to variable-level data

This script converts the one-hot encoded CSV (88 columns for Insurance)
back to variable-level format (27 columns) for FCI algorithm.

Usage:
    python scripts/convert_onehot_to_variable.py
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def convert_onehot_to_variable(onehot_csv_path: str, 
                                 metadata_path: str,
                                 output_path: str):
    """
    Convert one-hot encoded data to variable-level data
    
    Args:
        onehot_csv_path: Path to one-hot encoded CSV
        metadata_path: Path to metadata.json
        output_path: Path to save variable-level CSV
    """
    print("=" * 70)
    print("CONVERTING ONE-HOT TO VARIABLE-LEVEL DATA")
    print("=" * 70)
    print()
    
    # Load metadata
    print(f"Loading metadata: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load one-hot data
    print(f"Loading one-hot data: {onehot_csv_path}")
    df_onehot = pd.read_csv(onehot_csv_path)
    print(f"  Shape: {df_onehot.shape}")
    
    # Build reverse mapping: variable -> states
    # Note: state_name in metadata already includes variable prefix (e.g., "Age_Adult")
    var_to_states = {}
    for var_name, state_mapping in metadata['state_mappings'].items():
        # state_mapping is like {"0": "Age_Adolescent", "1": "Age_Adult", "2": "Age_Senior"}
        states = []
        for state_code, full_state_name in state_mapping.items():
            # full_state_name is like "Age_Adult"
            # Extract just the state part (e.g., "Adult")
            state_part = full_state_name.replace(f"{var_name}_", "")
            if full_state_name in df_onehot.columns:
                states.append((int(state_code), state_part, full_state_name))
        var_to_states[var_name] = sorted(states, key=lambda x: x[0])
    
    print(f"\nVariables found: {len(var_to_states)}")
    
    # Convert each variable
    var_data = {}
    for var_name, states in var_to_states.items():
        # Find which state is active for each sample
        # Use numeric codes instead of string labels for FCI
        var_values = []
        for idx in range(len(df_onehot)):
            # Find the active state (value = 1)
            found = False
            for state_code, state_part, col_name in states:
                if df_onehot.loc[idx, col_name] == 1:
                    # Use numeric code instead of string
                    var_values.append(state_code)
                    found = True
                    break
            
            if not found:
                # No state is active (shouldn't happen in valid data)
                var_values.append(-1)  # Use -1 for unknown
        
        var_data[var_name] = var_values
    
    # Create variable-level DataFrame
    df_variable = pd.DataFrame(var_data)
    
    # CRITICAL: Use metadata variable order instead of alphabetical sorting
    # This ensures FCI data matches the order in One-Hot data and metadata
    variable_order = metadata.get('variable_names', sorted(df_variable.columns))
    
    # Reorder columns to match metadata
    df_variable = df_variable[variable_order]
    
    print(f"\nConverted data shape: {df_variable.shape}")
    print(f"Variables (in metadata order): {list(df_variable.columns)}")
    
    # Save
    print(f"\nSaving to: {output_path}")
    df_variable.to_csv(output_path, index=False)
    print("Saved successfully!")
    
    # Show sample
    print("\nSample data (first 5 rows):")
    print(df_variable.head())
    
    return df_variable


def main():
    """Convert Insurance one-hot data to variable-level"""
    
    # Paths
    project_root = Path(__file__).parent.parent
    onehot_csv_path = project_root / 'data' / 'insurance' / 'insurance_data_10000.csv'
    metadata_path = project_root / 'data' / 'insurance' / 'metadata.json'
    output_path = project_root.parent / 'insurance_data.csv'  # Save to project root for FCI
    
    # Convert
    df_variable = convert_onehot_to_variable(
        str(onehot_csv_path),
        str(metadata_path),
        str(output_path)
    )
    
    print()
    print("=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print(f"\nVariable-level data saved to: {output_path}")
    print(f"  Samples: {len(df_variable)}")
    print(f"  Variables: {len(df_variable.columns)}")
    print()
    print("This file can now be used with FCI algorithm!")


if __name__ == "__main__":
    main()

