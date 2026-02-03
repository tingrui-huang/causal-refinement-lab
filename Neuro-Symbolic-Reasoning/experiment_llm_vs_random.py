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

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import config to get dataset paths
import config
from train_complete import train_complete


def run_experiment_for_dataset(dataset_name: str, 
                               high_conf: float = 0.6,
                               low_conf: float = 0.4,
                               n_epochs: int = 200,
                               run_mode: str = 'both',
                               random_seed: Optional[int] = None):
    """
    Run LLM vs Random experiment for a specific dataset
    
    Args:
        dataset_name: Name of dataset ('alarm', 'sachs', etc.)
        high_conf: High confidence weight (default: 0.6, weaker than 0.7)
        low_conf: Low confidence weight (default: 0.4, weaker than 0.3)
        n_epochs: Number of training epochs
        run_mode: 'both' | 'llm' | 'random'
        random_seed: Random seed for reproducibility (training + random prior)
    """
    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {dataset_name.upper()} DATASET")
    print("=" * 80)
    print(f"Prior weights: High={high_conf}, Low={low_conf}")
    print("=" * 80)
    
    # Default: follow unified config (single source of truth)
    if random_seed is None:
        random_seed = config.RANDOM_SEED
    random_seed = int(random_seed)
    
    # Get dataset configuration
    dataset_config = config.DATASET_CONFIGS[dataset_name]
    
    # Auto-detect latest skeleton and (optional) LLM direction files.
    #
    # IMPORTANT:
    # - pigs/link may use RFCI instead of FCI (edges_RFCI_*.csv).
    # - Some datasets/runs may NOT have LLM outputs at all. In that case we should still
    #   be able to run Random Prior only.
    # NOTE: edges_FCI_*.csv would also match edges_FCI_LLM_*.csv.
    # For "pure" constraint skeleton we must EXCLUDE any file containing "LLM".
    def _auto_detect_latest_non_llm(patterns, directory):
        from pathlib import Path
        d = Path(directory)
        if not d.exists():
            return None
        for pat in patterns:
            hits = [p for p in d.glob(pat) if "LLM" not in p.name.upper()]
            if hits:
                latest = max(hits, key=lambda p: p.stat().st_mtime)
                return str(latest)
        return None

    pure_skeleton_path = _auto_detect_latest_non_llm(
        ["edges_RFCI_*.csv", "edges_FCI_*.csv"],
        config.FCI_OUTPUT_DIR / dataset_name,
    )
    llm_skeleton_path = config._auto_detect_latest_file_any(
        ["edges_RFCI_LLM_*.csv", "edges_FCI_LLM_*.csv"],
        config.FCI_OUTPUT_DIR / dataset_name,
    )

    # -------------------------------------------------------------------------
    # Training cache: if an experiment output_dir already contains saved results,
    # skip re-training and just load metrics/history.
    # -------------------------------------------------------------------------
    def _try_load_cached_training(output_dir: Union[str, Path]) -> Optional[Dict]:
        out = Path(output_dir)
        metrics_path = out / "complete_metrics.json"
        history_path = out / "complete_history.json"
        if metrics_path.exists() and history_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                history = json.loads(history_path.read_text(encoding="utf-8"))
                return {
                    "metrics": metrics,
                    "history": history,
                    "model": None,  # not needed for evaluation reporting here
                    "fci_baseline_unresolved_ratio": None,
                    "cached": True,
                    "output_dir": str(out),
                }
            except Exception as e:
                print(f"[WARN] Failed to load cached results from {out}: {e}")
                return None
        return None

    requested_mode = run_mode
    effective_mode = run_mode

    # If LLM files are missing, automatically fall back to random prior.
    if effective_mode in ["both", "llm"] and not llm_skeleton_path:
        if effective_mode == "llm":
            print(f"\n[WARN] No LLM skeleton found for {dataset_name}. Falling back to RANDOM prior only.")
        else:
            print(f"\n[WARN] No LLM skeleton found for {dataset_name}. Running RANDOM prior only.")
        effective_mode = "random"

    # For random/both we must have at least a constraint skeleton (RFCI/FCI).
    if effective_mode in ["both", "random"] and not pure_skeleton_path:
        print(f"\n[ERROR] Missing constraint skeleton for {dataset_name}.")
        print(f"Expected one of: edges_RFCI_*.csv or edges_FCI_*.csv under {config.FCI_OUTPUT_DIR / dataset_name}")
        print("Please run the pipeline first:")
        print(f"  1. Set DATASET = '{dataset_name}' in config.py")
        print(f"  2. Run: python run_pipeline.py  (or run refactored/main_rfci.py or refactored/main_fci.py)")
        return None

    print("\nUsing files:")
    print(f"  Data path:       {dataset_config['data_path']}")
    print(f"  Metadata path:   {dataset_config['metadata_path']}")
    print(f"  Skeleton (for Random Prior): {pure_skeleton_path}")
    print(f"  LLM skeleton (for LLM Prior): {llm_skeleton_path}")
    print(f"  Ground truth:    {dataset_config['ground_truth_path']}")
    print(f"  Requested mode:  {requested_mode} -> Effective mode: {effective_mode}")

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
        'random_seed': random_seed,    # For reproducibility (training + random prior)
    }

    # ============================================================================
    # Run experiments based on run_mode
    # ============================================================================
    results_llm = None
    results_random = None

    if effective_mode in ['both', 'llm']:
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
            'output_dir': f'results/experiment_llm_vs_random/{dataset_name}/seed_{random_seed}/llm_prior'
        })

        cached = _try_load_cached_training(config_llm["output_dir"])
        if cached:
            print(f"\n[CACHE] Found existing LLM-prior training results at: {cached['output_dir']}")
            results_llm = cached
        else:
            results_llm = train_complete(config_llm)

    if effective_mode in ['both', 'random']:
        # Experiment 2: Random Prior (Control)
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {dataset_name.upper()} - RANDOM PRIOR (Blind Perturbation)")
        print("=" * 80)

        config_random = base_config.copy()
        config_random.update({
            'fci_skeleton_path': str(pure_skeleton_path),      # Use constraint skeleton (RFCI/FCI)
            'llm_direction_path': None,                        # Not used for random prior
            'use_llm_prior': False,
            'use_random_prior': True,
            'output_dir': f'results/experiment_llm_vs_random/{dataset_name}/seed_{random_seed}/random_prior'
        })
        
        cached = _try_load_cached_training(config_random["output_dir"])
        if cached:
            print(f"\n[CACHE] Found existing RANDOM-prior training results at: {cached['output_dir']}")
            results_random = cached
        else:
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
        output_dir = Path(f'results/experiment_llm_vs_random/{dataset_name}/seed_{random_seed}')
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
    
    # ============================================================================
    # Save run report (ALWAYS, even for random-only)
    # ============================================================================
    seed_dir = Path(f"results/experiment_llm_vs_random/{dataset_name}/seed_{random_seed}")
    seed_dir.mkdir(exist_ok=True, parents=True)
    report_path = seed_dir / "run_report.txt"

    def _safe_get_final_unresolved(res):
        try:
            return res["history"]["unresolved_ratio"][-1]
        except Exception:
            return None

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"dataset={dataset_name}\n")
        f.write(f"seed={random_seed}\n")
        f.write(f"requested_mode={requested_mode}\n")
        f.write(f"effective_mode={effective_mode}\n")
        f.write(f"skeleton_path={pure_skeleton_path}\n")
        f.write(f"llm_skeleton_path={llm_skeleton_path}\n")
        f.write(f"data_path={dataset_config['data_path']}\n")
        f.write(f"metadata_path={dataset_config['metadata_path']}\n")
        f.write(f"ground_truth_path={dataset_config['ground_truth_path']}\n")
        f.write(f"high_confidence={high_conf}\n")
        f.write(f"low_confidence={low_conf}\n")
        f.write(f"n_epochs={n_epochs}\n")
        f.write("\n")

        if results_llm:
            f.write("[LLM PRIOR]\n")
            f.write(f"orientation_accuracy={results_llm['metrics'].get('orientation_accuracy')}\n")
            f.write(f"edge_f1={results_llm['metrics'].get('edge_f1')}\n")
            f.write(f"directed_f1={results_llm['metrics'].get('directed_f1')}\n")
            f.write(f"unresolved_ratio_final={_safe_get_final_unresolved(results_llm)}\n")
            f.write("\n")
        if results_random:
            f.write("[RANDOM PRIOR]\n")
            f.write(f"orientation_accuracy={results_random['metrics'].get('orientation_accuracy')}\n")
            f.write(f"edge_f1={results_random['metrics'].get('edge_f1')}\n")
            f.write(f"directed_f1={results_random['metrics'].get('directed_f1')}\n")
            f.write(f"unresolved_ratio_final={_safe_get_final_unresolved(results_random)}\n")
            f.write("\n")

        if results_llm and results_random:
            f.write("[COMPARISON]\n")
            unresolved_diff = abs((_safe_get_final_unresolved(results_llm) or 0) - (_safe_get_final_unresolved(results_random) or 0))
            orientation_diff = (results_llm["metrics"].get("orientation_accuracy") or 0) - (results_random["metrics"].get("orientation_accuracy") or 0)
            f.write(f"unresolved_diff={unresolved_diff}\n")
            f.write(f"orientation_diff={orientation_diff}\n")

    print(f"\n[OK] Saved run report: {report_path}")

    return {
        'dataset': dataset_name,
        'seed': random_seed,
        'llm': results_llm,
        'random': results_random,
        'requested_mode': requested_mode,
        'effective_mode': effective_mode,
        'skeleton_path': str(pure_skeleton_path) if pure_skeleton_path else None,
        'llm_skeleton_path': str(llm_skeleton_path) if llm_skeleton_path else None,
        'report_path': str(report_path),
        'orientation_diff': orientation_diff,
        'unresolved_diff': unresolved_diff,
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

    # Optional CLI (mirrors run_multi_seed_random_prior.py style)
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--datasets", nargs="+", type=str, default=None,
                    help="Datasets to run (e.g. --datasets alarm sachs). If omitted, uses the defaults below.")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="Random seeds to run (e.g. --seeds 0 1 2). If omitted, uses the defaults below.")
    ap.add_argument("--run_mode", type=str, default=None, choices=["both", "llm", "random"],
                    help="Which experiments to run: both | llm | random. If omitted, uses the defaults below.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Training epochs. If omitted, uses the defaults below.")
    ap.add_argument("--high_conf", type=float, default=None,
                    help="High confidence prior weight. If omitted, uses the defaults below.")
    ap.add_argument("--low_conf", type=float, default=None,
                    help="Low confidence prior weight. If omitted, uses the defaults below.")
    args = ap.parse_args()
    
    # ============================================================================
    #  CONFIGURATION - 改这里！
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
    datasets = ['pigs']  # ← 改这里！(or use CLI: --datasets ...)
    
    # 选择要运行的实验类型
    # 'both'   - 运行 LLM 和 Random 两个实验（完整对比）
    # 'llm'    - 只运行 LLM Prior 实验
    # 'random' - 只运行 Random Prior 实验
    run_mode = 'random'  # ← 改这里！(or use CLI: --run_mode ...)

    # Random seeds: can be a single int or a list of ints
    # Example:
    #   seeds = 42
    #   seeds = [0, 1, 2, 3, 4]
    seeds: Union[int, List[int]] = [5]  # ← 改这里！(or use CLI: --seeds ...)
    
    # Prior 权重配置
    high_confidence = 0.9  # 强方向的权重（0.5-1.0）
    low_confidence = 0.1   # 弱方向的权重（0.0-0.5）
    
    # 训练轮数
    n_epochs = 500
    # ← 改这里！(推荐: sachs=300, alarm=1000, andes=1500,hailfinder=1000 )
    # ============================================================================

    # Apply CLI overrides (if provided)
    if args.datasets is not None:
        datasets = args.datasets
    if args.run_mode is not None:
        run_mode = args.run_mode
    if args.seeds is not None:
        seeds = args.seeds
    if args.epochs is not None:
        n_epochs = int(args.epochs)
    if args.high_conf is not None:
        high_confidence = float(args.high_conf)
    if args.low_conf is not None:
        low_confidence = float(args.low_conf)

    # Normalize seeds to list[int]
    if isinstance(seeds, int):
        seeds_list = [int(seeds)]
    else:
        seeds_list = [int(s) for s in seeds]

    total_start = time.time()
    all_results: Dict[str, Dict[int, Dict]] = {}
    
    for dataset_name in datasets:
        all_results[dataset_name] = {}
        for seed in seeds_list:
            try:
                result = run_experiment_for_dataset(
                    dataset_name=dataset_name,
                    high_conf=high_confidence,
                    low_conf=low_confidence,
                    n_epochs=n_epochs,
                    run_mode=run_mode,
                    random_seed=seed,
                )
                if result:
                    all_results[dataset_name][int(seed)] = result
            except Exception as e:
                print(f"\n[ERROR] Failed to run experiment for {dataset_name} (seed={seed}): {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # ============================================================================
    # Overall Summary
    # ============================================================================
    if any(all_results.get(ds) for ds in all_results):
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY - ALL DATASETS")
        print("=" * 80)
        
        for dataset_name, seed_map in all_results.items():
            if not seed_map:
                continue
            print(f"\n{dataset_name.upper()}:")
            for seed, result in seed_map.items():
                # Only print diffs when both runs exist
                if result.get("llm") and result.get("random"):
                    print(f"  seed_{seed}:")
                    print(f"    Orientation Diff: {result['orientation_diff']*100:+5.1f}%")
                    print(f"    LLM Orient Acc:   {result['llm']['metrics']['orientation_accuracy']*100:5.1f}%")
                    print(f"    Random Orient Acc:{result['random']['metrics']['orientation_accuracy']*100:5.1f}%")
                    if result['orientation_diff'] > 0.2:
                        print("    → LLM is INTELLIGENT GUIDE ✓")
                    elif result['orientation_diff'] < 0.1:
                        print("    → LLM is BLIND PERTURBATION ✗")
                    else:
                        print("    → MIXED RESULTS ?")
                elif result.get("llm"):
                    print(f"  seed_{seed}: LLM-only run completed")
                elif result.get("random"):
                    print(f"  seed_{seed}: Random-only run completed")
        
        print("\n" + "=" * 80)
        print(f"Total runtime (this script): {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
