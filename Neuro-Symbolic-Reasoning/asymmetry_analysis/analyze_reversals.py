"""
Analyze Reversals: Find the 7 edges with incorrect direction

This script:
1. Loads the learned adjacency matrix
2. Loads ground truth from alarm.bif
3. Identifies the 7 reversed edges
4. Checks what LLM prior suggested for these edges
"""

import torch
import pandas as pd
from pathlib import Path
from modules.data_loader import CausalDataLoader
from modules.evaluator import CausalGraphEvaluator


def load_learned_edges(adjacency_path: str, var_structure: dict, threshold: float = 0.1):
    """
    Load learned edges from adjacency matrix
    
    Returns:
        Set of (source, target) tuples at variable level
    """
    adjacency = torch.load(adjacency_path)
    
    learned_edges = set()
    var_names = var_structure['variable_names']
    
    for var_a in var_names:
        for var_b in var_names:
            if var_a == var_b:
                continue
            
            # Get state indices
            states_a = var_structure['var_to_states'][var_a]
            states_b = var_structure['var_to_states'][var_b]
            
            # Extract block weights
            block_weights = adjacency[states_a][:, states_b]
            mean_weight = block_weights.mean().item()
            
            # If weight exceeds threshold, add edge
            if mean_weight > threshold:
                learned_edges.add((var_a, var_b))
    
    return learned_edges


def load_ground_truth_edges(bif_path: str):
    """
    Load ground truth edges from BIF file
    
    Returns:
        Set of (parent, child) tuples
    """
    ground_truth = set()
    
    with open(bif_path, 'r') as f:
        content = f.read()
    
    # Parse BIF format
    import re
    
    # Find all variable definitions
    var_pattern = r'variable\s+(\w+)\s*{'
    prob_pattern = r'probability\s*\(\s*(\w+)(?:\s*\|\s*([^)]+))?\s*\)'
    
    for match in re.finditer(prob_pattern, content):
        child = match.group(1)
        parents_str = match.group(2)
        
        if parents_str:
            # Parse parents
            parents = [p.strip() for p in parents_str.split(',')]
            for parent in parents:
                ground_truth.add((parent, child))
    
    return ground_truth


def load_llm_prior_edges(llm_csv_path: str):
    """
    Load LLM prior edges from CSV
    
    Returns:
        Dict mapping (source, target) to edge_type
    """
    df = pd.read_csv(llm_csv_path)
    
    llm_edges = {}
    
    for _, row in df.iterrows():
        source = row['source']
        target = row['target']
        edge_type = row.get('edge_type', 'directed')
        
        llm_edges[(source, target)] = edge_type
    
    return llm_edges


def find_reversals(learned_edges, ground_truth_edges):
    """
    Find edges that are reversed compared to ground truth
    
    Returns:
        List of (learned_source, learned_target, gt_source, gt_target) tuples
    """
    reversals = []
    
    for learned_edge in learned_edges:
        source, target = learned_edge
        reverse_edge = (target, source)
        
        # Check if the reverse exists in ground truth
        if reverse_edge in ground_truth_edges and learned_edge not in ground_truth_edges:
            reversals.append({
                'model_edge': learned_edge,
                'gt_edge': reverse_edge,
                'model_direction': f"{source} -> {target}",
                'gt_direction': f"{target} -> {source}"
            })
    
    return reversals


def main():
    print("=" * 70)
    print("ANALYZING REVERSALS: Finding the 7 Incorrectly Oriented Edges")
    print("=" * 70)
    
    # Paths
    adjacency_path = '../results/complete/complete_adjacency.pt'
    metadata_path = '../old_version/output/knowledge_graph_metadata.json'
    ground_truth_path = 'data/alarm.bif'
    llm_prior_path = 'data/edges_Hybrid_FCI_LLM_20251207_230956.csv'
    
    # Load data
    print("\n[1/4] Loading learned adjacency matrix...")
    loader = CausalDataLoader(
        data_path='data/alarm_data_10000.csv',
        metadata_path=metadata_path
    )
    var_structure = loader.get_variable_structure()
    
    learned_edges = load_learned_edges(adjacency_path, var_structure, threshold=0.1)
    print(f"Learned edges: {len(learned_edges)}")
    
    # Load ground truth
    print("\n[2/4] Loading ground truth from BIF...")
    ground_truth_edges = load_ground_truth_edges(ground_truth_path)
    print(f"Ground truth edges: {len(ground_truth_edges)}")
    
    # Load LLM prior
    print("\n[3/4] Loading LLM prior suggestions...")
    llm_prior_edges = load_llm_prior_edges(llm_prior_path)
    print(f"LLM prior edges: {len(llm_prior_edges)}")
    
    # Find reversals
    print("\n[4/4] Finding reversals...")
    reversals = find_reversals(learned_edges, ground_truth_edges)
    
    print("\n" + "=" * 70)
    print(f"FOUND {len(reversals)} REVERSALS")
    print("=" * 70)
    
    if len(reversals) == 0:
        print("\nNo reversals found! All learned edges have correct direction.")
        return
    
    # Print detailed analysis
    print("\nDetailed Analysis:")
    print("-" * 70)
    
    for i, reversal in enumerate(reversals, 1):
        model_edge = reversal['model_edge']
        gt_edge = reversal['gt_edge']
        
        print(f"\n{i}. REVERSAL:")
        print(f"   Model learned:    {reversal['model_direction']}")
        print(f"   Ground truth:     {reversal['gt_direction']}")
        
        # Check LLM prior
        llm_forward = llm_prior_edges.get(model_edge)
        llm_backward = llm_prior_edges.get(gt_edge)
        
        print(f"   LLM Prior:")
        if llm_forward:
            print(f"     - {model_edge[0]} -> {model_edge[1]}: {llm_forward}")
        else:
            print(f"     - {model_edge[0]} -> {model_edge[1]}: NOT in LLM prior")
        
        if llm_backward:
            print(f"     - {gt_edge[0]} -> {gt_edge[1]}: {llm_backward}")
        else:
            print(f"     - {gt_edge[0]} -> {gt_edge[1]}: NOT in LLM prior")
        
        # Diagnosis
        if llm_forward and not llm_backward:
            print(f"   Diagnosis: LLM suggested WRONG direction (model followed LLM)")
        elif llm_backward and not llm_forward:
            print(f"   Diagnosis: LLM suggested CORRECT direction (model ignored LLM)")
        elif llm_forward and llm_backward:
            print(f"   Diagnosis: LLM had BOTH directions (undirected edge)")
        else:
            print(f"   Diagnosis: Edge NOT in LLM prior (model inferred from data)")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    llm_wrong = 0
    llm_correct = 0
    llm_both = 0
    llm_missing = 0
    
    for reversal in reversals:
        model_edge = reversal['model_edge']
        gt_edge = reversal['gt_edge']
        
        llm_forward = llm_prior_edges.get(model_edge)
        llm_backward = llm_prior_edges.get(gt_edge)
        
        if llm_forward and not llm_backward:
            llm_wrong += 1
        elif llm_backward and not llm_forward:
            llm_correct += 1
        elif llm_forward and llm_backward:
            llm_both += 1
        else:
            llm_missing += 1
    
    print(f"\nLLM Prior Analysis:")
    print(f"  LLM suggested WRONG direction: {llm_wrong} / {len(reversals)}")
    print(f"  LLM suggested CORRECT direction (model ignored): {llm_correct} / {len(reversals)}")
    print(f"  LLM had BOTH directions (undirected): {llm_both} / {len(reversals)}")
    print(f"  Edge NOT in LLM prior: {llm_missing} / {len(reversals)}")
    
    print("\nConclusion:")
    if llm_wrong > len(reversals) / 2:
        print("  Most reversals are due to LLM giving WRONG direction.")
        print("  The model correctly learned from LLM, but LLM was incorrect.")
    elif llm_correct > len(reversals) / 2:
        print("  Most reversals are due to model IGNORING correct LLM suggestions.")
        print("  Need to increase lambda_cycle or LLM prior confidence.")
    else:
        print("  Mixed causes. Need case-by-case analysis.")
    
    # Save results
    output_path = 'results/complete/reversals_analysis.txt'
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"REVERSALS ANALYSIS: {len(reversals)} Incorrectly Oriented Edges\n")
        f.write("=" * 70 + "\n\n")
        
        for i, reversal in enumerate(reversals, 1):
            model_edge = reversal['model_edge']
            gt_edge = reversal['gt_edge']
            
            f.write(f"{i}. REVERSAL:\n")
            f.write(f"   Model learned:    {reversal['model_direction']}\n")
            f.write(f"   Ground truth:     {reversal['gt_direction']}\n")
            
            llm_forward = llm_prior_edges.get(model_edge)
            llm_backward = llm_prior_edges.get(gt_edge)
            
            f.write(f"   LLM Prior:\n")
            if llm_forward:
                f.write(f"     - {model_edge[0]} -> {model_edge[1]}: {llm_forward}\n")
            if llm_backward:
                f.write(f"     - {gt_edge[0]} -> {gt_edge[1]}: {llm_backward}\n")
            f.write("\n")
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()





