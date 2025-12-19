"""
Convert knowledge graph triples to CSV format
This is a one-time conversion script for Phase 1
"""

import json
import pandas as pd
from pathlib import Path


def convert_triples_to_csv():
    """
    Convert triples format to direct CSV format
    
    Input: knowledge_graph_triples.json (Subject, Predicate, Object)
    Output: alarm_data_10000.csv (N_samples x N_states matrix)
    """
    print("Converting triples to CSV format...")
    
    # Load metadata
    with open('output/knowledge_graph_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Build state index mapping
    state_to_idx = {}
    idx_to_state = {}
    idx_counter = 0
    
    for var, mapping in metadata['state_mappings'].items():
        sorted_vals = sorted(mapping.keys(), key=lambda x: int(x))
        for val_code in sorted_vals:
            state_name = mapping[val_code]
            state_to_idx[state_name] = idx_counter
            idx_to_state[idx_counter] = state_name
            idx_counter += 1
    
    n_states = len(state_to_idx)
    print(f"Total states: {n_states}")
    
    # Load triples
    with open('output/knowledge_graph_triples.json', 'r') as f:
        triples = json.load(f)
    
    # Get unique patients
    patients = set()
    for triple in triples:
        if isinstance(triple, dict):
            patients.add(triple['subject'])
        else:
            patients.add(triple[0])
    
    patients = sorted(list(patients))
    n_patients = len(patients)
    print(f"Total patients: {n_patients}")
    
    # Build patient-to-row mapping
    patient_to_row = {pid: i for i, pid in enumerate(patients)}
    
    # Initialize data matrix
    data_matrix = [[0] * n_states for _ in range(n_patients)]
    
    # Fill matrix from triples
    for triple in triples:
        if isinstance(triple, dict):
            patient = triple['subject']
            state = triple['object']
        else:
            patient = triple[0]
            state = triple[2]
        
        if state in state_to_idx:
            row = patient_to_row[patient]
            col = state_to_idx[state]
            data_matrix[row][col] = 1
    
    # Create DataFrame
    column_names = [idx_to_state[i] for i in range(n_states)]
    df = pd.DataFrame(data_matrix, columns=column_names)
    
    # Add patient ID column
    df.insert(0, 'sample_id', patients)
    
    # Save to CSV
    output_path = Path('data/alarm_data_10000.csv')
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(column_names)} states + 1 sample_id")
    
    # Verify
    states_per_patient = df[column_names].sum(axis=1)
    print(f"States per patient: mean={states_per_patient.mean():.1f}, "
          f"min={states_per_patient.min()}, max={states_per_patient.max()}")
    
    return df


if __name__ == "__main__":
    df = convert_triples_to_csv()
    print("\nConversion complete!")

