"""
为所有数据集生成 FCI 变量级别数据
Generate FCI variable-level data for all datasets

这个脚本会：
1. 清理项目根目录的旧 FCI 文件
2. 为每个数据集重新生成正确的 FCI 文件
3. 确保变量顺序与 metadata.json 一致

Usage:
    python convert_all_datasets.py
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_variables_from_metadata(df_columns, metadata):
    """
    从 metadata 反向提取变量顺序（修复多下划线问题）
    
    Args:
        df_columns: CSV 列名列表
        metadata: metadata.json 内容
    
    Returns:
        变量名列表（按 CSV 中首次出现的顺序）
    """
    state_mappings = metadata['state_mappings']
    
    # 建立 state_name -> variable_name 的映射
    state_to_var = {}
    for var_name, states in state_mappings.items():
        for state_code, state_name in states.items():
            state_to_var[state_name] = var_name
    
    # 按 CSV 列顺序提取变量（保持首次出现顺序）
    variables_ordered = []
    seen_vars = set()
    
    for col in df_columns:
        if col in state_to_var:
            var_name = state_to_var[col]
            if var_name not in seen_vars:
                variables_ordered.append(var_name)
                seen_vars.add(var_name)
    
    return variables_ordered


def convert_onehot_to_variable(onehot_csv_path: Path, 
                                metadata_path: Path,
                                output_path: Path,
                                dataset_name: str):
    """
    Convert one-hot encoded data to variable-level data
    
    Args:
        onehot_csv_path: Path to one-hot encoded CSV
        metadata_path: Path to metadata.json
        output_path: Path to save variable-level CSV
        dataset_name: Name of dataset (for display)
    """
    print("\n" + "=" * 80)
    print(f"CONVERTING {dataset_name.upper()} TO VARIABLE-LEVEL DATA")
    print("=" * 80)
    
    # Check if files exist
    if not onehot_csv_path.exists():
        print(f"[SKIP] One-hot data not found: {onehot_csv_path}")
        return False
    
    if not metadata_path.exists():
        print(f"[SKIP] Metadata not found: {metadata_path}")
        return False
    
    try:
        # Load metadata
        print(f"\n[1/5] Loading metadata: {metadata_path.name}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load one-hot data
        print(f"[2/5] Loading one-hot data: {onehot_csv_path.name}")
        df_onehot = pd.read_csv(onehot_csv_path)
        
        # Remove non-numeric columns (like sample_id)
        non_numeric_cols = df_onehot.select_dtypes(include=['object']).columns.tolist()
        if non_numeric_cols:
            print(f"      Removing non-numeric columns: {non_numeric_cols}")
            df_onehot = df_onehot.select_dtypes(include=['number'])
        
        print(f"      Shape: {df_onehot.shape}")
        
        # Build reverse mapping: variable -> states
        print(f"[3/5] Building variable mappings")
        var_to_states = {}
        for var_name, state_mapping in metadata['state_mappings'].items():
            states = []
            for state_code, full_state_name in state_mapping.items():
                if full_state_name in df_onehot.columns:
                    states.append((int(state_code), full_state_name))
            if states:
                var_to_states[var_name] = sorted(states, key=lambda x: x[0])
        
        print(f"      Variables found: {len(var_to_states)}")
        
        # Convert each variable
        print(f"[4/5] Converting to variable-level format")
        var_data = {}
        for var_name, states in var_to_states.items():
            var_values = []
            for idx in range(len(df_onehot)):
                # Find the active state (value = 1)
                found = False
                for state_code, col_name in states:
                    if df_onehot.loc[idx, col_name] == 1:
                        var_values.append(state_code)
                        found = True
                        break
                
                if not found:
                    var_values.append(-1)  # Unknown state
            
            var_data[var_name] = var_values
        
        # Create variable-level DataFrame
        df_variable = pd.DataFrame(var_data)
        
        # CRITICAL FIX: Use metadata-based extraction for correct variable order
        # This handles datasets with multi-underscore state names (Child, Hailfinder, Win95pts)
        if 'variable_names' in metadata:
            # Use predefined order from metadata
            variable_order = metadata['variable_names']
        else:
            # Extract order from CSV columns using metadata mapping
            variable_order = extract_variables_from_metadata(df_onehot.columns, metadata)
        
        # Filter to only include variables that exist in the data
        variable_order = [v for v in variable_order if v in df_variable.columns]
        
        # Reorder columns to match metadata
        df_variable = df_variable[variable_order]
        
        print(f"      Converted shape: {df_variable.shape}")
        print(f"      Variable order (first 10): {list(df_variable.columns)[:10]}")
        
        # Save
        print(f"[5/5] Saving to: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_variable.to_csv(output_path, index=False)
        print(f"      [OK] Saved successfully!")
        
        # Verify
        print(f"\n[VERIFY] Sample data (first 3 rows):")
        print(df_variable.head(3))
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed to convert {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def clean_old_fci_files(project_root: Path):
    """Clean old FCI files from project root"""
    print("\n" + "=" * 80)
    print("CLEANING OLD FCI FILES")
    print("=" * 80)
    
    # Pattern to match: *_data.csv or *_data_variable.csv in project root
    patterns = ['*_data.csv', '*_data_variable.csv']
    
    cleaned = []
    for pattern in patterns:
        for file in project_root.glob(pattern):
            # Skip if it's in a subdirectory
            if file.parent == project_root:
                print(f"  Removing: {file.name}")
                file.unlink()
                cleaned.append(file.name)
    
    if cleaned:
        print(f"\n[OK] Cleaned {len(cleaned)} old FCI files")
    else:
        print(f"\n[OK] No old FCI files found")
    
    return cleaned


def main():
    """Convert all datasets"""
    
    print("\n" + "=" * 80)
    print("BATCH CONVERSION: ALL DATASETS TO FCI FORMAT")
    print("=" * 80)
    print("\nThis script will:")
    print("  1. Clean old FCI files from project root")
    print("  2. Generate new FCI files for each dataset")
    print("  3. Use metadata variable order (NOT alphabetical)")
    print("=" * 80)
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    output_dir = project_root.parent  # Project root for FCI files
    
    # Step 1: Clean old files
    cleaned = clean_old_fci_files(output_dir)
    
    # Step 2: Define datasets to convert
    datasets = [
        {
            'name': 'alarm',
            'onehot': data_dir / 'alarm' / 'alarm_data_10000.csv',
            'metadata': data_dir / 'alarm' / 'metadata.json',
            'output': output_dir / 'alarm_data.csv'
        },
        {
            'name': 'sachs',
            'onehot': data_dir / 'sachs' / 'sachs_data.csv',
            'metadata': data_dir / 'sachs' / 'metadata.json',
            'output': output_dir / 'sachs_data_variable.csv'
        },
        {
            'name': 'andes',
            'onehot': data_dir / 'andes' / 'andes_data.csv',
            'metadata': data_dir / 'andes' / 'metadata.json',
            'output': output_dir / 'andes_data_variable.csv'
        },
        {
            'name': 'child',
            'onehot': data_dir / 'child' / 'child_data.csv',
            'metadata': data_dir / 'child' / 'metadata.json',
            'output': output_dir / 'child_data_variable.csv'
        },
        {
            'name': 'hailfinder',
            'onehot': data_dir / 'hailfinder' / 'hailfinder_data.csv',
            'metadata': data_dir / 'hailfinder' / 'metadata.json',
            'output': output_dir / 'hailfinder_data_variable.csv'
        },
        {
            'name': 'insurance',
            'onehot': data_dir / 'insurance' / 'insurance_data_10000.csv',
            'metadata': data_dir / 'insurance' / 'metadata.json',
            'output': output_dir / 'insurance_data.csv'
        },
        {
            'name': 'win95pts',
            'onehot': data_dir / 'win95pts' / 'win95pts_data.csv',
            'metadata': data_dir / 'win95pts' / 'metadata.json',
            'output': output_dir / 'win95pts_data_variable.csv'
        },
    ]
    
    # Step 3: Convert each dataset
    results = {}
    for dataset in datasets:
        success = convert_onehot_to_variable(
            onehot_csv_path=dataset['onehot'],
            metadata_path=dataset['metadata'],
            output_path=dataset['output'],
            dataset_name=dataset['name']
        )
        results[dataset['name']] = success
    
    # Summary
    print("\n" + "=" * 80)
    print("CONVERSION SUMMARY")
    print("=" * 80)
    
    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]
    
    print(f"\n[OK] Successful: {len(successful)}/{len(results)}")
    for name in successful:
        print(f"  - {name}")
    
    if failed:
        print(f"\n[X] Failed: {len(failed)}/{len(results)}")
        for name in failed:
            print(f"  - {name}")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Run validation to verify fixes:")
    print("   python validate_data_health.py --all")
    print("\n2. Check that variable orders now match:")
    print("   - One-Hot data order")
    print("   - Metadata variable_names order")
    print("   - FCI data column order")
    print("\n3. If validation passes, you can safely run experiments!")
    print("=" * 80)


if __name__ == "__main__":
    main()

