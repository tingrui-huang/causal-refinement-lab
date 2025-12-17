"""
Compare Training with and without LLM Direction Prior

This script trains the model in 3 configurations:
1. With GPT-3.5 direction prior
2. With Zephyr direction prior  
3. WITHOUT any LLM prior (uniform initialization)

Goal: Understand how much the LLM prior helps vs pure data-driven learning
"""

import torch
from train_complete import train_complete


def run_comparison():
    """
    Run training with different LLM configurations
    """
    
    # Base configuration
    base_config = {
        'data_path': 'data/alarm_data_10000.csv',
        'metadata_path': 'output/knowledge_graph_metadata.json',
        'fci_skeleton_path': 'data/edges_FCI_20251207_230824.csv',
        'ground_truth_path': 'data/alarm.bif',
        'n_epochs': 200,
        'learning_rate': 0.01,
        'n_hops': 1,
        'lambda_group': 0.01,
        'lambda_cycle': 0.001,
        'monitor_interval': 20,
        'edge_threshold': 0.1,
    }
    
    # Configurations to test
    configs = [
        {
            'name': 'GPT-3.5',
            'llm_direction_path': 'data/edges_Hybrid_FCI_LLM_20251207_230956.csv',
            'use_llm_prior': True,
            'output_dir': 'results/gpt35'
        },
        {
            'name': 'Zephyr',
            'llm_direction_path': 'data/edges_Hybrid_FCI_Zephyr_20251207_231914.csv',
            'use_llm_prior': True,
            'output_dir': 'results/zephyr'
        },
        {
            'name': 'No-LLM (Pure Data)',
            'llm_direction_path': None,
            'use_llm_prior': False,
            'output_dir': 'results/no_llm'
        }
    ]
    
    results_summary = []
    
    print("=" * 70)
    print("COMPARING LLM PRIOR CONFIGURATIONS")
    print("=" * 70)
    print(f"\nTesting {len(configs)} configurations:")
    for cfg in configs:
        print(f"  - {cfg['name']}")
    print()
    
    # Train with each configuration
    for i, exp_cfg in enumerate(configs, 1):
        print("\n" + "=" * 70)
        print(f"EXPERIMENT {i}/{len(configs)}: {exp_cfg['name']}")
        print("=" * 70)
        
        # Merge configurations
        config = {**base_config, **exp_cfg}
        
        # Train
        model, metrics, history = train_complete(config)
        
        # Store results
        results_summary.append({
            'name': exp_cfg['name'],
            'metrics': metrics,
            'final_losses': {
                'reconstruction': history['loss_reconstruction'][-1],
                'group_lasso': history['loss_group_lasso'][-1],
                'cycle': history['loss_cycle'][-1],
                'total': history['loss_total'][-1]
            },
            'direction_learning': {
                'initial_bidirectional_ratio': history['bidirectional_ratio'][0],
                'final_bidirectional_ratio': history['bidirectional_ratio'][-1],
                'change': history['bidirectional_ratio'][-1] - history['bidirectional_ratio'][0]
            },
            'sparsity': {
                'overall': history['overall_sparsity'][-1],
                'block': history['block_sparsity'][-1],
                'active_connections': history['active_connections'][-1],
                'active_blocks': history['active_blocks'][-1]
            }
        })
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("EDGE DISCOVERY METRICS")
    print("=" * 70)
    print(f"\n{'Metric':<25} | {'GPT-3.5':<12} | {'Zephyr':<12} | {'No-LLM':<12}")
    print("-" * 70)
    
    metrics_to_compare = [
        ('Edge Precision', 'edge_precision', '%'),
        ('Edge Recall', 'edge_recall', '%'),
        ('Edge F1 Score', 'edge_f1', '%'),
        ('Learned Edges', 'learned_edges', ''),
    ]
    
    for metric_name, metric_key, unit in metrics_to_compare:
        values = []
        for result in results_summary:
            val = result['metrics'][metric_key]
            if unit == '%':
                values.append(f"{val*100:>6.1f}%")
            else:
                values.append(f"{val:>6}")
        print(f"{metric_name:<25} | {values[0]:<12} | {values[1]:<12} | {values[2]:<12}")
    
    print("\n" + "=" * 70)
    print("DIRECTION LEARNING METRICS")
    print("=" * 70)
    print(f"\n{'Metric':<25} | {'GPT-3.5':<12} | {'Zephyr':<12} | {'No-LLM':<12}")
    print("-" * 70)
    
    direction_metrics = [
        ('Orientation Accuracy', 'orientation_accuracy', '%'),
        ('Reversals', 'reversals', ''),
        ('SHD', 'shd', ''),
    ]
    
    for metric_name, metric_key, unit in direction_metrics:
        values = []
        for result in results_summary:
            val = result['metrics'][metric_key]
            if unit == '%':
                values.append(f"{val*100:>6.1f}%")
            else:
                values.append(f"{val:>6}")
        print(f"{metric_name:<25} | {values[0]:<12} | {values[1]:<12} | {values[2]:<12}")
    
    print("\n" + "=" * 70)
    print("DIRECTION LEARNING DYNAMICS")
    print("=" * 70)
    print(f"\n{'Metric':<25} | {'GPT-3.5':<12} | {'Zephyr':<12} | {'No-LLM':<12}")
    print("-" * 70)
    
    for result in results_summary:
        dl = result['direction_learning']
        print(f"Initial Bidir Ratio      | ", end="")
        for r in results_summary:
            print(f"{r['direction_learning']['initial_bidirectional_ratio']*100:>6.1f}%    | ", end="")
        print()
        break
    
    for result in results_summary:
        print(f"Final Bidir Ratio        | ", end="")
        for r in results_summary:
            print(f"{r['direction_learning']['final_bidirectional_ratio']*100:>6.1f}%    | ", end="")
        print()
        break
    
    for result in results_summary:
        print(f"Change                   | ", end="")
        for r in results_summary:
            change = r['direction_learning']['change']
            print(f"{change*100:>+6.1f}%    | ", end="")
        print()
        break
    
    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    no_llm_result = results_summary[2]
    gpt_result = results_summary[0]
    
    print(f"\n1. Edge Discovery:")
    print(f"   - GPT-3.5:  F1={gpt_result['metrics']['edge_f1']*100:.1f}%")
    print(f"   - No-LLM:   F1={no_llm_result['metrics']['edge_f1']*100:.1f}%")
    diff = (gpt_result['metrics']['edge_f1'] - no_llm_result['metrics']['edge_f1']) * 100
    print(f"   - Difference: {diff:+.1f}%")
    
    print(f"\n2. Direction Learning:")
    print(f"   - GPT-3.5:  Orientation={gpt_result['metrics']['orientation_accuracy']*100:.1f}%")
    print(f"   - No-LLM:   Orientation={no_llm_result['metrics']['orientation_accuracy']*100:.1f}%")
    diff = (gpt_result['metrics']['orientation_accuracy'] - no_llm_result['metrics']['orientation_accuracy']) * 100
    print(f"   - Difference: {diff:+.1f}%")
    
    print(f"\n3. Reversals:")
    print(f"   - GPT-3.5:  {gpt_result['metrics']['reversals']} reversals")
    print(f"   - No-LLM:   {no_llm_result['metrics']['reversals']} reversals")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if gpt_result['metrics']['edge_f1'] > no_llm_result['metrics']['edge_f1'] + 0.05:
        print("\nLLM prior SIGNIFICANTLY helps with edge discovery")
    elif gpt_result['metrics']['edge_f1'] > no_llm_result['metrics']['edge_f1']:
        print("\nLLM prior SLIGHTLY helps with edge discovery")
    else:
        print("\nLLM prior does NOT help with edge discovery")
    
    if gpt_result['metrics']['orientation_accuracy'] > no_llm_result['metrics']['orientation_accuracy'] + 0.05:
        print("LLM prior SIGNIFICANTLY helps with direction learning")
    elif gpt_result['metrics']['orientation_accuracy'] > no_llm_result['metrics']['orientation_accuracy']:
        print("LLM prior SLIGHTLY helps with direction learning")
    else:
        print("LLM prior does NOT help with direction learning")
        print("=> Fine-grained data asymmetry is sufficient!")
    
    print("\nThis demonstrates the value of:")
    if gpt_result['metrics']['edge_f1'] > no_llm_result['metrics']['edge_f1'] + 0.05:
        print("  1. LLM priors for initialization")
    print("  2. Fine-grained state-level modeling (Neural LP)")
    print("  3. Cycle Consistency Loss for direction learning")
    
    return results_summary


if __name__ == "__main__":
    results = run_comparison()

