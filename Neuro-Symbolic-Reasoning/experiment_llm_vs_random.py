"""
Experiment: LLM as "Intelligent Guide" vs "Blind Perturbation"

Question: Does LLM direction matter, or just any asymmetric initialization?

Setup:
    1. Baseline (LLM): Use LLM direction prior (normal training)
    2. Control (Random): Use random direction prior (same magnitude, random direction)

Expected Results:
    - If LLM is just "blind perturbation":
        Both should have similar Unresolved Ratio AND Orientation Accuracy
    
    - If LLM is "intelligent guide":
        Random: Unresolved Ratio -> 0%, but Orientation Accuracy ~50% (random guessing)
        LLM: Unresolved Ratio -> 0%, and Orientation Accuracy >> 50% (guided)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from train_complete import train_complete

# Shared configuration
base_config = {
    'data_path': 'data/alarm/alarm_data_10000.csv',
    'metadata_path': 'data/alarm/metadata.json',
    'fci_skeleton_path': 'data/alarm/edges_FCI_20251207_230824.csv',
    'llm_direction_path': 'data/alarm/edges_Hybrid_FCI_LLM_20251207_230956.csv',
    'ground_truth_path': 'data/alarm/alarm.bif',
    'n_epochs': 200,
    'learning_rate': 0.01,
    'n_hops': 1,
    'lambda_group': 0.01,
    'lambda_cycle': 0.001,
    'monitor_interval': 20,
    'edge_threshold': 0.1,
}

print("\n" + "=" * 80)
print("EXPERIMENT: LLM AS 'INTELLIGENT GUIDE' VS 'BLIND PERTURBATION'")
print("=" * 80)
print("\nResearch Question:")
print("  Does LLM provide meaningful directional guidance, or does it just")
print("  break symmetry like random noise?")
print("\nHypothesis:")
print("  - If LLM is 'Intelligent Guide': High orientation accuracy")
print("  - If LLM is 'Blind Perturbation': ~50% orientation accuracy (random)")
print("=" * 80)

# ============================================================================
# Experiment 1: LLM Prior (Baseline)
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 1: LLM PRIOR (Baseline - 'Intelligent Guide')")
print("=" * 80)

config_llm = base_config.copy()
config_llm.update({
    'use_llm_prior': True,
    'use_random_prior': False,
    'output_dir': 'results/experiment_llm_vs_random/llm_prior'
})

results_llm = train_complete(config_llm)

# ============================================================================
# Experiment 2: Random Prior (Control)
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 2: RANDOM PRIOR (Control - 'Blind Perturbation')")
print("=" * 80)

config_random = base_config.copy()
config_random.update({
    'use_llm_prior': False,
    'use_random_prior': True,
    'random_seed': 42,  # For reproducibility
    'output_dir': 'results/experiment_llm_vs_random/random_prior'
})

results_random = train_complete(config_random)

# ============================================================================
# Comparison
# ============================================================================
print("\n" + "=" * 80)
print("RESULTS COMPARISON")
print("=" * 80)

print("\n[Symmetry Breaking] Unresolved Ratio:")
print(f"  LLM Prior:    {results_llm['history']['unresolved_ratio'][-1]*100:5.1f}%")
print(f"  Random Prior: {results_random['history']['unresolved_ratio'][-1]*100:5.1f}%")
print(f"  Difference:   {(results_llm['history']['unresolved_ratio'][-1] - results_random['history']['unresolved_ratio'][-1])*100:+5.1f}%")

print("\n[Orientation Accuracy] Direction Correctness:")
print(f"  LLM Prior:    {results_llm['metrics']['orientation_accuracy']*100:5.1f}%")
print(f"  Random Prior: {results_random['metrics']['orientation_accuracy']*100:5.1f}%")
print(f"  Difference:   {(results_llm['metrics']['orientation_accuracy'] - results_random['metrics']['orientation_accuracy'])*100:+5.1f}%")

print("\n[Other Metrics]:")
print(f"  Edge Precision:")
print(f"    LLM:    {results_llm['metrics']['edge_precision']*100:5.1f}%")
print(f"    Random: {results_random['metrics']['edge_precision']*100:5.1f}%")
print(f"  Edge Recall:")
print(f"    LLM:    {results_llm['metrics']['edge_recall']*100:5.1f}%")
print(f"    Random: {results_random['metrics']['edge_recall']*100:5.1f}%")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

unresolved_diff = abs(results_llm['history']['unresolved_ratio'][-1] - 
                      results_random['history']['unresolved_ratio'][-1])
orientation_diff = results_llm['metrics']['orientation_accuracy'] - \
                   results_random['metrics']['orientation_accuracy']

if unresolved_diff < 0.1 and orientation_diff > 0.2:
    print("\n[V] LLM is an 'INTELLIGENT GUIDE'")
    print("   Both break symmetry (low unresolved ratio), but LLM guides")
    print("   in the correct direction (high orientation accuracy).")
    print(f"\n   Evidence:")
    print(f"   - Both achieve similar symmetry breaking (~{results_llm['history']['unresolved_ratio'][-1]*100:.1f}%)")
    print(f"   - But LLM has {orientation_diff*100:+.1f}% higher orientation accuracy")
    print(f"   - Random prior is essentially guessing (~50%)")
elif unresolved_diff < 0.1 and orientation_diff < 0.1:
    print("\n[!] LLM is a 'BLIND PERTURBATION'")
    print("   Both break symmetry equally, and orientation is similar.")
    print("   LLM doesn't provide meaningful directional guidance.")
else:
    print("\n[?] MIXED RESULTS - Further investigation needed")
    print(f"   Unresolved difference: {unresolved_diff*100:.1f}%")
    print(f"   Orientation difference: {orientation_diff*100:+.1f}%")

# Save comparison report
output_dir = Path('results/experiment_llm_vs_random')
output_dir.mkdir(exist_ok=True, parents=True)

with open(output_dir / 'comparison_report.txt', 'w') as f:
    f.write("LLM vs Random Prior Experiment\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("Research Question:\n")
    f.write("  Is LLM an 'Intelligent Guide' or 'Blind Perturbation'?\n\n")
    
    f.write("Results:\n")
    f.write(f"  Symmetry Breaking (Unresolved Ratio):\n")
    f.write(f"    LLM:    {results_llm['history']['unresolved_ratio'][-1]*100:5.1f}%\n")
    f.write(f"    Random: {results_random['history']['unresolved_ratio'][-1]*100:5.1f}%\n")
    f.write(f"    Diff:   {(results_llm['history']['unresolved_ratio'][-1] - results_random['history']['unresolved_ratio'][-1])*100:+5.1f}%\n\n")
    
    f.write(f"  Orientation Accuracy:\n")
    f.write(f"    LLM:    {results_llm['metrics']['orientation_accuracy']*100:5.1f}%\n")
    f.write(f"    Random: {results_random['metrics']['orientation_accuracy']*100:5.1f}%\n")
    f.write(f"    Diff:   {orientation_diff*100:+5.1f}%\n\n")
    
    f.write(f"  Edge Precision:\n")
    f.write(f"    LLM:    {results_llm['metrics']['edge_precision']*100:5.1f}%\n")
    f.write(f"    Random: {results_random['metrics']['edge_precision']*100:5.1f}%\n\n")
    
    f.write(f"  Edge Recall:\n")
    f.write(f"    LLM:    {results_llm['metrics']['edge_recall']*100:5.1f}%\n")
    f.write(f"    Random: {results_random['metrics']['edge_recall']*100:5.1f}%\n\n")
    
    f.write("Conclusion:\n")
    if unresolved_diff < 0.1 and orientation_diff > 0.2:
        f.write("  [V] LLM is an 'INTELLIGENT GUIDE'\n")
        f.write("  LLM provides meaningful directional guidance beyond just breaking symmetry.\n")
    elif unresolved_diff < 0.1 and orientation_diff < 0.1:
        f.write("  [!] LLM is a 'BLIND PERTURBATION'\n")
        f.write("  LLM only breaks symmetry but doesn't guide direction.\n")
    else:
        f.write("  [?] MIXED RESULTS\n")

print(f"\nComparison report saved to: {output_dir / 'comparison_report.txt'}")
print("\n" + "=" * 80)

