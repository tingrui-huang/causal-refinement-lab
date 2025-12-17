"""
Inspect Data Matrix Structure
打印data_matrix的前30行，查看矩阵结构和列的含义
"""

import json
import torch
import pandas as pd
from pathlib import Path

def inspect_data_matrix():
    """
    Load and inspect the data matrix structure
    """
    print("=" * 80)
    print("DATA MATRIX INSPECTION")
    print("=" * 80)
    
    # Load metadata
    print("\n1. Loading metadata...")
    with open('output/knowledge_graph_metadata.json', 'r') as f:
        meta = json.load(f)
    
    # Build state index mapping
    state_to_idx = {}
    idx_to_state = {}
    idx_counter = 0
    
    for var, mapping in meta['state_mappings'].items():
        sorted_vals = sorted(mapping.keys(), key=lambda x: int(x))
        for val_code in sorted_vals:
            state_name = mapping[val_code]
            state_to_idx[state_name] = idx_counter
            idx_to_state[idx_counter] = state_name
            idx_counter += 1
    
    print(f"   Total states (columns): {len(state_to_idx)}")
    
    # Load patient data (facts)
    print("\n2. Loading patient data...")
    with open('output/knowledge_graph_triples.json', 'r') as f:
        triples = json.load(f)
    
    # Build data matrix
    patient_ids = set()
    for t in triples:
        if isinstance(t, dict):
            patient_ids.add(t['subject'])
        else:
            patient_ids.add(t[0])
    
    n_patients = len(patient_ids)
    patient_to_row = {pid: i for i, pid in enumerate(sorted(list(patient_ids)))}
    row_to_patient = {i: pid for pid, i in patient_to_row.items()}
    
    data_matrix = torch.zeros(n_patients, len(state_to_idx))
    
    for t in triples:
        if isinstance(t, dict):
            sub, pred, obj = t['subject'], t['predicate'], t['object']
        else:
            sub, pred, obj = t[0], t[1], t[2]
        
        if obj in state_to_idx:
            row = patient_to_row[sub]
            col = state_to_idx[obj]
            data_matrix[row, col] = 1.0
    
    print(f"   Total patients (rows): {n_patients}")
    print(f"   Total facts: {int(data_matrix.sum().item())}")
    print(f"   Matrix shape: {data_matrix.shape}")
    print(f"   Matrix density: {data_matrix.sum().item() / (n_patients * len(state_to_idx)) * 100:.2f}%")
    
    # Print column information
    print("\n" + "=" * 80)
    print("COLUMN INFORMATION (States)")
    print("=" * 80)
    print(f"\nTotal columns: {len(idx_to_state)}")
    print("\nFirst 50 columns (states):")
    print(f"{'Index':<8} | {'State Name':<40} | {'Variable':<20} | {'Count'}")
    print("-" * 80)
    
    for i in range(min(50, len(idx_to_state))):
        state_name = idx_to_state[i]
        # Extract variable name
        parts = state_name.split('_')
        var_name = '_'.join(parts[:-1]) if len(parts) >= 2 else state_name
        count = int(data_matrix[:, i].sum().item())
        print(f"{i:<8} | {state_name:<40} | {var_name:<20} | {count}")
    
    if len(idx_to_state) > 50:
        print(f"... and {len(idx_to_state) - 50} more columns")
    
    # Group by variable
    print("\n" + "=" * 80)
    print("STATES GROUPED BY VARIABLE")
    print("=" * 80)
    
    var_to_states = {}
    for idx, state_name in idx_to_state.items():
        parts = state_name.split('_')
        var_name = '_'.join(parts[:-1]) if len(parts) >= 2 else state_name
        if var_name not in var_to_states:
            var_to_states[var_name] = []
        var_to_states[var_name].append((idx, state_name))
    
    print(f"\nTotal variables: {len(var_to_states)}")
    print(f"\n{'Variable':<20} | {'# States':<10} | {'State Names'}")
    print("-" * 80)
    
    for var_name in sorted(var_to_states.keys()):
        states = var_to_states[var_name]
        state_names = [s[1].split('_')[-1] for s in states]
        print(f"{var_name:<20} | {len(states):<10} | {', '.join(state_names)}")
    
    # Print first 30 rows
    print("\n" + "=" * 80)
    print("FIRST 30 ROWS OF DATA MATRIX")
    print("=" * 80)
    print("\nNote: Showing only columns with at least one '1' in first 30 rows for readability")
    
    # Get first 30 rows
    first_30 = data_matrix[:30].numpy()
    
    # Find columns that have at least one 1 in first 30 rows
    active_cols = []
    for col_idx in range(first_30.shape[1]):
        if first_30[:, col_idx].sum() > 0:
            active_cols.append(col_idx)
    
    print(f"\nActive columns in first 30 rows: {len(active_cols)} / {len(idx_to_state)}")
    
    # Create DataFrame for better visualization
    # Show first 30 patients
    df_rows = []
    for row_idx in range(min(30, n_patients)):
        patient_id = row_to_patient[row_idx]
        row_data = {'Patient': patient_id}
        
        # Add active states
        active_states = []
        for col_idx in range(data_matrix.shape[1]):
            if data_matrix[row_idx, col_idx] == 1:
                active_states.append(idx_to_state[col_idx])
        
        row_data['Active_States'] = len(active_states)
        row_data['States'] = ', '.join(active_states[:5])  # Show first 5
        if len(active_states) > 5:
            row_data['States'] += f' ... (+{len(active_states) - 5} more)'
        
        df_rows.append(row_data)
    
    df = pd.DataFrame(df_rows)
    print("\n" + df.to_string(index=False))
    
    # Show detailed view for first 5 patients
    print("\n" + "=" * 80)
    print("DETAILED VIEW: FIRST 5 PATIENTS")
    print("=" * 80)
    
    for row_idx in range(min(5, n_patients)):
        patient_id = row_to_patient[row_idx]
        print(f"\n{patient_id}:")
        
        # Group states by variable
        patient_states = {}
        for col_idx in range(data_matrix.shape[1]):
            if data_matrix[row_idx, col_idx] == 1:
                state_name = idx_to_state[col_idx]
                parts = state_name.split('_')
                var_name = '_'.join(parts[:-1]) if len(parts) >= 2 else state_name
                state_value = parts[-1] if len(parts) >= 2 else state_name
                patient_states[var_name] = state_value
        
        # Print in columns
        items = list(patient_states.items())
        for i in range(0, len(items), 3):
            row_items = items[i:i+3]
            row_str = "  "
            for var, val in row_items:
                row_str += f"{var:<20} = {val:<15} "
            print(row_str)
    
    # Statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    
    states_per_patient = data_matrix.sum(dim=1)
    print(f"\nStates per patient:")
    print(f"  Mean:   {states_per_patient.mean().item():.2f}")
    print(f"  Median: {states_per_patient.median().item():.2f}")
    print(f"  Min:    {states_per_patient.min().item():.0f}")
    print(f"  Max:    {states_per_patient.max().item():.0f}")
    
    patients_per_state = data_matrix.sum(dim=0)
    print(f"\nPatients per state:")
    print(f"  Mean:   {patients_per_state.mean().item():.2f}")
    print(f"  Median: {patients_per_state.median().item():.2f}")
    print(f"  Min:    {patients_per_state.min().item():.0f}")
    print(f"  Max:    {patients_per_state.max().item():.0f}")
    
    # Most common states
    print(f"\nTop 20 most common states:")
    top_states_indices = torch.argsort(patients_per_state, descending=True)[:20]
    for rank, idx in enumerate(top_states_indices, 1):
        state_name = idx_to_state[idx.item()]
        count = int(patients_per_state[idx].item())
        percentage = count / n_patients * 100
        print(f"  {rank:2d}. {state_name:<40} {count:5d} patients ({percentage:5.1f}%)")
    
    # Least common states
    print(f"\nTop 20 least common states:")
    bottom_states_indices = torch.argsort(patients_per_state, descending=False)[:20]
    for rank, idx in enumerate(bottom_states_indices, 1):
        state_name = idx_to_state[idx.item()]
        count = int(patients_per_state[idx].item())
        percentage = count / n_patients * 100
        print(f"  {rank:2d}. {state_name:<40} {count:5d} patients ({percentage:5.1f}%)")
    
    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    inspect_data_matrix()

