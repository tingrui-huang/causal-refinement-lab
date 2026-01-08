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

UPDATED: Now uses weaker initialization (0.6/0.4 instead of 0.7/0.3)
         Tests on multiple datasets: Alarm, Sachs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import config to get dataset paths
import config
from train_complete import train_complete


def run_experiment_for_dataset(dataset_name: str, 
                               high_conf: float = 0.6, 
                               low_conf: float = 0.4,
                               n_epochs: int = 200,
                               run_mode: str = 'both'):
    """
    Run LLM vs Random experiment for a specific dataset
    
    Args:
        dataset_name: Name of dataset ('alarm', 'sachs', etc.)
        high_conf: High confidence weight (default: 0.6, weaker than 0.7)
        low_conf: Low confidence weight (default: 0.4, weaker than 0.3)
        n_epochs: Number of training epochs
    """
    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {dataset_name.upper()} DATASET")
    print("=" * 80)
    print(f"Prior weights: High={high_conf}, Low={low_conf}")
    print("=" * 80)
    
    # Get dataset configuration
    dataset_config = config.DATASET_CONFIGS[dataset_name]
    
    # Auto-detect latest FCI and LLM files
    # IMPORTANT: Use different skeleton files for different experiments!
    # - Pure FCI skeleton (edges_FCI_[0-9]*.csv): For Random Prior (e.g., 40 edges)
    # - FCI+LLM skeleton (edges_FCI_LLM_*.csv): For LLM Prior (e.g., 38 edges, LLM-filtered)
    pure_fci_skeleton_path = config._auto_detect_latest_file('edges_FCI_[0-9]*.csv', config.FCI_OUTPUT_DIR / dataset_name)
    llm_skeleton_path = config._auto_detect_latest_file('edges_FCI_LLM_*.csv', config.FCI_OUTPUT_DIR / dataset_name)

    print("\nUsing files:")
    print(f"  Data path:       {dataset_config['data_path']}")
    print(f"  Metadata path:   {dataset_config['metadata_path']}")
    print(f"  Pure FCI skeleton (for Random Prior): {pure_fci_skeleton_path}")
    print(f"  FCI+LLM skeleton (for LLM Prior):     {llm_skeleton_path}")
    print(f"  Ground truth:    {dataset_config['ground_truth_path']}")

    if not pure_fci_skeleton_path or not llm_skeleton_path:
        print(f"\n[ERROR] Missing FCI or LLM files for {dataset_name}")
        print("Please run the pipeline first:")
        print(f"  1. Set DATASET = '{dataset_name}' in config.py")
        print(f"  2. Run: python run_pipeline.py")
        return None

    # Get dataset-specific hyperparameters
    if dataset_name == 'sachs':
        lambda_group = 0.01
        lambda_cycle = 0.001
        edge_threshold = 0.08
    elif dataset_name == 'alarm':
        lambda_group = 0.01
        lambda_cycle = 0.001
        edge_threshold = 0.08
    elif dataset_name == 'andes':
        lambda_group = 0.05
        lambda_cycle = 0.05
        edge_threshold = 0.08
    elif dataset_name == 'child':
        lambda_group = 0.005
        lambda_cycle = 0.001
        edge_threshold = 0.08
    elif dataset_name == 'hailfinder':
        lambda_group = 0.01
        lambda_cycle = 0.001
        edge_threshold = 0.08
    elif dataset_name == 'win95pts':
        lambda_group = 0.01
        lambda_cycle = 0.001
        edge_threshold = 0.1
    elif dataset_name == 'insurance':
        lambda_group = 0.01
        lambda_cycle = 0.001
        edge_threshold = 0.08
    else:
        # Default values
        lambda_group = 0.01
        lambda_cycle = 0.001
        edge_threshold = 0.1

    # Shared configuration (without skeleton paths - will be set per experiment)
    base_config = {
        'data_path': str(dataset_config['data_path']),
        'metadata_path': str(dataset_config['metadata_path']),
        'ground_truth_path': str(dataset_config['ground_truth_path']),
        'ground_truth_type': dataset_config.get('ground_truth_type', 'bif'),
        'n_epochs': n_epochs,
        'learning_rate': 0.01,
        'n_hops': 1,
        'lambda_group': lambda_group,
        'lambda_cycle': lambda_cycle,
        'monitor_interval': 20,
        'edge_threshold': edge_threshold,
        'high_confidence': high_conf,  # Pass to prior builder
        'low_confidence': low_conf,    # Pass to prior builder
    }

    # ============================================================================
    # Run experiments based on run_mode
    # ============================================================================
    results_llm = None
    results_random = None

    if run_mode in ['both', 'llm']:
        # Experiment 1: LLM Prior (Baseline)
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {dataset_name.upper()} - LLM PRIOR (Intelligent Guide)")
        print("=" * 80)

        config_llm = base_config.copy()
        config_llm.update({
            'fci_skeleton_path': str(llm_skeleton_path),      # Use FCI+LLM skeleton (LLM-filtered edges)
            'llm_direction_path': str(llm_skeleton_path),     # Same file for direction
            'use_llm_prior': True,
            'use_random_prior': False,
            'output_dir': f'results/experiment_llm_vs_random/{dataset_name}/llm_prior'
        })

        results_llm = train_complete(config_llm)

    if run_mode in ['both', 'random']:
        # Experiment 2: Random Prior (Control)
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {dataset_name.upper()} - RANDOM PRIOR (Blind Perturbation)")
        print("=" * 80)

        config_random = base_config.copy()
        config_random.update({
            'fci_skeleton_path': str(pure_fci_skeleton_path),  # Use pure FCI skeleton (all FCI edges)
            'llm_direction_path': None,                        # Not used for random prior
            'use_llm_prior': False,
            'use_random_prior': True,
            'random_seed': 42,  # For reproducibility
            'output_dir': f'results/experiment_llm_vs_random/{dataset_name}/random_prior'
        })
        
        results_random = train_complete(config_random)
    
    # ============================================================================
    # Comparison (only if both experiments were run)
    # ============================================================================
    if results_llm and results_random:
        print("\n" + "=" * 80)
        print(f"RESULTS COMPARISON - {dataset_name.upper()}")
        print("=" * 80)
        
        print("\n[Symmetry Breaking] Unresolved Ratio:")
        print(f"  LLM Prior:    {results_llm['history']['unresolved_ratio'][-1]*100:5.1f}%")
        print(f"  Random Prior: {results_random['history']['unresolved_ratio'][-1]*100:5.1f}%")
        print(f"  Difference:   {(results_llm['history']['unresolved_ratio'][-1] - results_random['history']['unresolved_ratio'][-1])*100:+5.1f}%")
        
        print("\n[Orientation Accuracy] Direction Correctness:")
        print(f"  LLM Prior:    {results_llm['metrics']['orientation_accuracy']*100:5.1f}%")
        print(f"  Random Prior: {results_random['metrics']['orientation_accuracy']*100:5.1f}%")
        print(f"  Difference:   {(results_llm['metrics']['orientation_accuracy'] - results_random['metrics']['orientation_accuracy'])*100:+5.1f}%")
        
        print("\n[Edge Metrics]:")
        print(f"  Edge F1:")
        print(f"    LLM:    {results_llm['metrics']['edge_f1']*100:5.1f}%")
        print(f"    Random: {results_random['metrics']['edge_f1']*100:5.1f}%")
        print(f"  Directed F1:")
        print(f"    LLM:    {results_llm['metrics']['directed_f1']*100:5.1f}%")
        print(f"    Random: {results_random['metrics']['directed_f1']*100:5.1f}%")
    elif results_llm:
        print("\n" + "=" * 80)
        print(f"RESULTS - {dataset_name.upper()} - LLM PRIOR ONLY")
        print("=" * 80)
        print(f"\nOrientation Accuracy: {results_llm['metrics']['orientation_accuracy']*100:5.1f}%")
        print(f"Edge F1:              {results_llm['metrics']['edge_f1']*100:5.1f}%")
        print(f"Directed F1:          {results_llm['metrics']['directed_f1']*100:5.1f}%")
        print(f"Unresolved Ratio:     {results_llm['history']['unresolved_ratio'][-1]*100:5.1f}%")
    elif results_random:
        print("\n" + "=" * 80)
        print(f"RESULTS - {dataset_name.upper()} - RANDOM PRIOR ONLY")
        print("=" * 80)
        print(f"\nOrientation Accuracy: {results_random['metrics']['orientation_accuracy']*100:5.1f}%")
        print(f"Edge F1:              {results_random['metrics']['edge_f1']*100:5.1f}%")
        print(f"Directed F1:          {results_random['metrics']['directed_f1']*100:5.1f}%")
        print(f"Unresolved Ratio:     {results_random['history']['unresolved_ratio'][-1]*100:5.1f}%")
    
    # ============================================================================
    # Conclusion (only if both experiments were run)
    # ============================================================================
    orientation_diff = 0
    unresolved_diff = 0
    
    if results_llm and results_random:
        print("\n" + "=" * 80)
        print("CONCLUSION")
        print("=" * 80)
        
        unresolved_diff = abs(results_llm['history']['unresolved_ratio'][-1] - 
                              results_random['history']['unresolved_ratio'][-1])
        orientation_diff = results_llm['metrics']['orientation_accuracy'] - \
                           results_random['metrics']['orientation_accuracy']
    
        if unresolved_diff < 0.1 and orientation_diff > 0.2:
            print("\n[✓] LLM is an 'INTELLIGENT GUIDE'")
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
    
    # ============================================================================
    # Save comparison report (only if both experiments were run)
    # ============================================================================
    if results_llm and results_random:
        output_dir = Path(f'results/experiment_llm_vs_random/{dataset_name}')
        output_dir.mkdir(exist_ok=True, parents=True)
        
        with open(output_dir / 'comparison_report.txt', 'w', encoding='utf-8') as f:
            f.write(f"LLM vs Random Prior Experiment - {dataset_name.upper()}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Prior Configuration:\n")
            f.write(f"  High confidence: {high_conf}\n")
            f.write(f"  Low confidence:  {low_conf}\n\n")
            
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
            
            f.write(f"  Edge F1:\n")
            f.write(f"    LLM:    {results_llm['metrics']['edge_f1']*100:5.1f}%\n")
            f.write(f"    Random: {results_random['metrics']['edge_f1']*100:5.1f}%\n\n")
            
            f.write(f"  Directed F1:\n")
            f.write(f"    LLM:    {results_llm['metrics']['directed_f1']*100:5.1f}%\n")
            f.write(f"    Random: {results_random['metrics']['directed_f1']*100:5.1f}%\n\n")
            
            f.write("Conclusion:\n")
            if unresolved_diff < 0.1 and orientation_diff > 0.2:
                f.write("  [✓] LLM is an 'INTELLIGENT GUIDE'\n")
                f.write("  LLM provides meaningful directional guidance beyond just breaking symmetry.\n")
            elif unresolved_diff < 0.1 and orientation_diff < 0.1:
                f.write("  [!] LLM is a 'BLIND PERTURBATION'\n")
                f.write("  LLM only breaks symmetry but doesn't guide direction.\n")
            else:
                f.write("  [?] MIXED RESULTS\n")
        
        print(f"\nComparison report saved to: {output_dir / 'comparison_report.txt'}")
        print("\n" + "=" * 80)
    
    return {
        'dataset': dataset_name,
        'llm': results_llm,
        'random': results_random,
        'orientation_diff': orientation_diff,
        'unresolved_diff': unresolved_diff
    }


def main():
    """Run experiments on multiple datasets"""
    print("\n" + "=" * 80)
    print("EXPERIMENT: LLM AS 'INTELLIGENT GUIDE' VS 'BLIND PERTURBATION'")
    print("=" * 80)
    print("\nResearch Question:")
    print("  Does LLM provide meaningful directional guidance, or does it just")
    print("  break symmetry like random noise?")
    print("\nHypothesis:")
    print("  - If LLM is 'Intelligent Guide': High orientation accuracy")
    print("  - If LLM is 'Blind Perturbation': ~50% orientation accuracy (random)")
    print("\nNEW: Using weaker initialization (0.6/0.4 instead of 0.7/0.3)")
    print("=" * 80)
    
    # ============================================================================
    # 🔧 CONFIGURATION - 改这里！
    # ============================================================================
    # 选择要测试的数据集（可以是单个或多个）
    # 
    # 单个数据集示例：
    #   datasets = ['andes']        # 只跑 andes
    #   datasets = ['alarm']        # 只跑 alarm
    #   datasets = ['sachs']        # 只跑 sachs
    #
    # 多个数据集示例：
    #   datasets = ['alarm', 'sachs']           # 跑两个
    #   datasets = ['alarm', 'sachs', 'andes']  # 跑三个
    #
    # 可选数据集：'alarm', 'sachs', 'andes', 'child', 'hailfinder', 'insurance', 'win95pts'
    datasets = ['andes']  # ← 改这里！
    
    # 选择要运行的实验类型
    # 'both'   - 运行 LLM 和 Random 两个实验（完整对比）
    # 'llm'    - 只运行 LLM Prior 实验
    # 'random' - 只运行 Random Prior 实验
    run_mode = 'both'  # ← 改这里！选择 'both', 'llm', 或 'random'
    
    # Prior 权重配置
    high_confidence = 0.9  # 强方向的权重（0.5-1.0）
    low_confidence = 0.1   # 弱方向的权重（0.0-0.5）
    
    # 训练轮数
    n_epochs = 2000
    # ← 改这里！(推荐: sachs=300, alarm=1000, andes=1500,hailfinder=1000 )
    # ============================================================================


    all_results = {}
    
    for dataset_name in datasets:
        try:
            result = run_experiment_for_dataset(
                dataset_name=dataset_name,
                high_conf=high_confidence,
                low_conf=low_confidence,
                n_epochs=n_epochs,
                run_mode=run_mode
            )
            if result:
                all_results[dataset_name] = result
        except Exception as e:
            print(f"\n[ERROR] Failed to run experiment for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ============================================================================
    # Overall Summary
    # ============================================================================
    if all_results:
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY - ALL DATASETS")
        print("=" * 80)
        
        for dataset_name, result in all_results.items():
            print(f"\n{dataset_name.upper()}:")
            print(f"  Orientation Diff: {result['orientation_diff']*100:+5.1f}%")
            print(f"  LLM Orient Acc:   {result['llm']['metrics']['orientation_accuracy']*100:5.1f}%")
            print(f"  Random Orient Acc:{result['random']['metrics']['orientation_accuracy']*100:5.1f}%")
            
            if result['orientation_diff'] > 0.2:
                print(f"  → LLM is INTELLIGENT GUIDE ✓")
            elif result['orientation_diff'] < 0.1:
                print(f"  → LLM is BLIND PERTURBATION ✗")
            else:
                print(f"  → MIXED RESULTS ?")
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
