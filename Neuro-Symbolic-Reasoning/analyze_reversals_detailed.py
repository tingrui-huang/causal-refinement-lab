"""
Detailed Reversal Analysis with Conditional Probability Check

For each of the 7 reversed edges, compute:
1. P(B|A) vs P(A|B) from data
2. Check if data supports model's direction or ground truth's direction
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from modules.data_loader import CausalDataLoader


def compute_conditional_probability(data: torch.Tensor, var_structure: dict, 
                                    var_a: str, var_b: str):
    """
    Compute P(B|A) and P(A|B) from data
    
    Returns:
        dict with P(B|A), P(A|B), and asymmetry score
    """
    # Get state indices
    states_a = var_structure['var_to_states'][var_a]
    states_b = var_structure['var_to_states'][var_b]
    
    # Convert to numpy for easier computation
    data_np = data.numpy()
    
    # For each state of A, compute P(B states | A state)
    p_b_given_a_list = []
    for state_a in states_a:
        # Find samples where state_a is active
        mask_a = data_np[:, state_a] == 1
        n_a = mask_a.sum()
        
        if n_a == 0:
            continue
        
        # For each state of B, compute P(state_b | state_a)
        for state_b in states_b:
            mask_b = data_np[:, state_b] == 1
            n_ab = (mask_a & mask_b).sum()
            p_b_given_a = n_ab / n_a if n_a > 0 else 0
            p_b_given_a_list.append(p_b_given_a)
    
    # For each state of B, compute P(A states | B state)
    p_a_given_b_list = []
    for state_b in states_b:
        mask_b = data_np[:, state_b] == 1
        n_b = mask_b.sum()
        
        if n_b == 0:
            continue
        
        for state_a in states_a:
            mask_a = data_np[:, state_a] == 1
            n_ab = (mask_a & mask_b).sum()
            p_a_given_b = n_ab / n_b if n_b > 0 else 0
            p_a_given_b_list.append(p_a_given_b)
    
    # Average conditional probabilities
    p_b_given_a = np.mean(p_b_given_a_list) if p_b_given_a_list else 0
    p_a_given_b = np.mean(p_a_given_b_list) if p_a_given_b_list else 0
    
    # Asymmetry score: positive means A->B more likely, negative means B->A
    asymmetry = p_b_given_a - p_a_given_b
    
    return {
        'P(B|A)': p_b_given_a,
        'P(A|B)': p_a_given_b,
        'asymmetry': asymmetry,
        'data_suggests': 'A->B' if asymmetry > 0 else 'B->A' if asymmetry < 0 else 'unclear'
    }


def main():
    print("=" * 70)
    print("DETAILED REVERSAL ANALYSIS WITH CONDITIONAL PROBABILITY")
    print("=" * 70)
    
    # The 7 reversals identified
    reversals = [
        ('VENTTUBE', 'DISCONNECT'),      # Model: VENTTUBE->DISCONNECT, GT: DISCONNECT->VENTTUBE
        ('VENTLUNG', 'INTUBATION'),      # Model: VENTLUNG->INTUBATION, GT: INTUBATION->VENTLUNG
        ('LVEDVOLUME', 'HYPOVOLEMIA'),   # Model: LVEDVOLUME->HYPOVOLEMIA, GT: HYPOVOLEMIA->LVEDVOLUME
        ('VENTTUBE', 'VENTMACH'),        # Model: VENTTUBE->VENTMACH, GT: VENTMACH->VENTTUBE
        ('CATECHOL', 'TPR'),             # Model: CATECHOL->TPR, GT: TPR->CATECHOL
        ('PRESS', 'KINKEDTUBE'),         # Model: PRESS->KINKEDTUBE, GT: KINKEDTUBE->PRESS
        ('SHUNT', 'PULMEMBOLUS'),        # Model: SHUNT->PULMEMBOLUS, GT: PULMEMBOLUS->SHUNT
    ]
    
    # Load data
    print("\n[1/2] Loading observational data...")
    loader = CausalDataLoader(
        data_path='data/alarm_data_10000.csv',
        metadata_path='output/knowledge_graph_metadata.json'
    )
    data = loader.load_data()
    var_structure = loader.get_variable_structure()
    print(f"Loaded {data.shape[0]} samples")
    
    # Analyze each reversal
    print("\n[2/2] Computing conditional probabilities...")
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS OF 7 REVERSALS")
    print("=" * 70)
    
    results = []
    
    for i, (var_a, var_b) in enumerate(reversals, 1):
        print(f"\n{i}. Edge: {var_a} <-> {var_b}")
        print("-" * 70)
        
        # Compute conditional probabilities
        stats = compute_conditional_probability(data, var_structure, var_a, var_b)
        
        print(f"   Model learned:    {var_a} -> {var_b}")
        print(f"   Ground truth:     {var_b} -> {var_a}")
        print(f"   LLM suggested:    {var_a} -> {var_b} (llm_resolved)")
        print(f"\n   Conditional Probabilities from Data:")
        print(f"     P({var_b}|{var_a}) = {stats['P(B|A)']:.4f}")
        print(f"     P({var_a}|{var_b}) = {stats['P(A|B)']:.4f}")
        print(f"     Asymmetry = {stats['asymmetry']:+.4f}")
        print(f"\n   Data suggests:    {stats['data_suggests']}")
        
        # Determine who is correct
        model_direction = f"{var_a}->{var_b}"
        gt_direction = f"{var_b}->{var_a}"
        
        if stats['asymmetry'] > 0.01:  # Significant asymmetry
            data_supports_model = True
            verdict = "Model & LLM CORRECT, Ground Truth might be WRONG"
        elif stats['asymmetry'] < -0.01:
            data_supports_model = False
            verdict = "Ground Truth CORRECT, Model & LLM WRONG"
        else:
            data_supports_model = None
            verdict = "Data is AMBIGUOUS (weak asymmetry)"
        
        print(f"\n   Verdict: {verdict}")
        
        results.append({
            'edge': f"{var_a} <-> {var_b}",
            'model_direction': model_direction,
            'gt_direction': gt_direction,
            'P(B|A)': stats['P(B|A)'],
            'P(A|B)': stats['P(A|B)'],
            'asymmetry': stats['asymmetry'],
            'data_suggests': stats['data_suggests'],
            'verdict': verdict
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    model_correct = sum(1 for r in results if 'Model & LLM CORRECT' in r['verdict'])
    gt_correct = sum(1 for r in results if 'Ground Truth CORRECT' in r['verdict'])
    ambiguous = sum(1 for r in results if 'AMBIGUOUS' in r['verdict'])
    
    print(f"\nData Analysis Results:")
    print(f"  Data supports Model & LLM: {model_correct} / 7")
    print(f"  Data supports Ground Truth: {gt_correct} / 7")
    print(f"  Data is ambiguous: {ambiguous} / 7")
    
    print("\nKey Insights:")
    if model_correct > gt_correct:
        print("  The model and LLM are mostly CORRECT!")
        print("  The 'reversals' might actually be errors in the ground truth BIF file.")
        print("  Observational data supports the learned directions.")
    elif gt_correct > model_correct:
        print("  The ground truth is mostly CORRECT.")
        print("  The model failed to learn the correct causal direction from data.")
        print("  LLM priors are misleading the model.")
    else:
        print("  Mixed results. Some edges have weak causal signals in the data.")
        print("  May need interventional data to resolve these ambiguities.")
    
    # Save detailed report
    output_path = 'results/complete/reversals_detailed_analysis.txt'
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DETAILED REVERSAL ANALYSIS WITH CONDITIONAL PROBABILITY\n")
        f.write("=" * 70 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"{i}. {result['edge']}\n")
            f.write(f"   Model:     {result['model_direction']}\n")
            f.write(f"   GT:        {result['gt_direction']}\n")
            f.write(f"   P(B|A):    {result['P(B|A)']:.4f}\n")
            f.write(f"   P(A|B):    {result['P(A|B)']:.4f}\n")
            f.write(f"   Asymmetry: {result['asymmetry']:+.4f}\n")
            f.write(f"   Data suggests: {result['data_suggests']}\n")
            f.write(f"   Verdict:   {result['verdict']}\n\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Data supports Model & LLM: {model_correct} / 7\n")
        f.write(f"Data supports Ground Truth: {gt_correct} / 7\n")
        f.write(f"Data is ambiguous: {ambiguous} / 7\n")
    
    print(f"\nDetailed report saved to: {output_path}")
    
    # Also create a CSV for easy analysis
    df = pd.DataFrame(results)
    csv_path = 'results/complete/reversals_analysis.csv'
    df.to_csv(csv_path, index=False)
    print(f"CSV report saved to: {csv_path}")


if __name__ == "__main__":
    main()

