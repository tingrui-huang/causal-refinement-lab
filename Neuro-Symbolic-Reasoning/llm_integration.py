"""
Test LLM Integration on "Dead Hard" Pairs

This script tests the LLM integration (方案 A: 温和初始化) on the 3 "dead hard" pairs
that have strong signals but wrong direction (Pair 73, 77, 92).

Expected outcome:
- LLM provides soft initial bias (e.g., 0.6 vs 0.4)
- GSB still works on top of this bias
- Direction accuracy should improve from wrong to correct

Configuration:
- Same as benchmark: 5 bins, uniform, lambda_cycle=0.05, lambda_group=0.0
- NEW: use_llm_prior=True (enables LLM consultation)
"""

import sys
from pathlib import Path
import pandas as pd
import time
from datetime import datetime

# Load .env at the very start (before any imports that might need it)
import os
from pathlib import Path

def load_env_file():
    """Load .env file manually if python-dotenv is not available"""
    env_path = Path(__file__).parent.parent / '.env'
    
    if not env_path.exists():
        print(f"[WARN] .env file not found at: {env_path}\n")
        return False
    
    try:
        # Try using python-dotenv first
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=True)
        print(f"[INFO] Loaded .env using python-dotenv from: {env_path}")
    except ImportError:
        # Fallback: manually parse .env file
        print(f"[INFO] python-dotenv not installed, manually parsing .env")
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith('#'):
                        # Parse KEY=VALUE
                        if '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            # Remove quotes if present
                            if value.startswith('"') and value.endswith('"'):
                                value = value[1:-1]
                            elif value.startswith("'") and value.endswith("'"):
                                value = value[1:-1]
                            # Set environment variable
                            os.environ[key] = value
            print(f"[INFO] Manually loaded .env from: {env_path}")
        except Exception as e:
            print(f"[ERROR] Failed to parse .env: {e}\n")
            return False
    
    # Verify API key is loaded
    api_key_len = len(os.getenv('OPENAI_API_KEY', ''))
    if api_key_len > 0:
        print(f"[INFO] API key loaded successfully (length: {api_key_len})\n")
        return True
    else:
        print("[WARN] .env file found but API key is empty!\n")
        return False

# Load .env file
load_env_file()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_tuebingen_bench import (
    discover_tuebingen_pairs,
    run_single_pair
)
from modules.tuebingen_semantic_parser import parse_tuebingen_description


def main():
    """Test LLM integration on specific pairs"""
    print("=" * 80)
    print("LLM INTEGRATION TEST (方案 C: 语义增强)")
    print("=" * 80)
    print("\nTarget Pairs: 1, 73, 77, 92")
    print("  - Pair 1: Has semantic info (Altitude -> Temperature)")
    print("  - Pair 73, 77, 92: Have semantic info (Energy, Solar, Soil)")
    print("\nStrategy:")
    print("  - Parse _des.txt files to extract variable semantics")
    print("  - LLM uses semantic info to make informed judgments")
    print("  - If no semantics, skip LLM (return neutral 0.5, 0.5)")
    print("  - LLM provides soft initial bias (e.g., 0.6 vs 0.4)")
    print("  - GSB (Gradient-based Symmetry Breaking) works on top")
    print("\nConfiguration:")
    print("  - Bins: 5")
    print("  - Strategy: Uniform")
    print("  - Lambda Group Lasso: 0.0")
    print("  - Lambda Cycle: 0.05")
    print("  - Use LLM Prior: TRUE (with semantic info)")
    print("=" * 80)
    
    # Target pairs to test (including pair0001 with clear semantics)
    target_pairs = ['pair0001', 'pair0073', 'pair0077', 'pair0092']
    
    # Discover all pairs
    data_dir = Path(__file__).parent / 'data' / 'tuebingen'
    all_pairs = discover_tuebingen_pairs(data_dir, auto_convert=True)
    
    # Filter target pairs
    test_pairs = [(pid, csv, gt) for pid, csv, gt in all_pairs if pid in target_pairs]
    
    if len(test_pairs) == 0:
        print("\n[ERROR] No target pairs found!")
        print("Available pairs:", [pid for pid, _, _ in all_pairs[:10]])
        return
    
    print(f"\nFound {len(test_pairs)} target pairs:")
    for pair_id, csv_path, gt_dir in test_pairs:
        gt_str = "x->y" if gt_dir == 1 else "y->x" if gt_dir == -1 else "unknown"
        
        # Parse semantic info
        des_path = csv_path.parent / f"{pair_id}_des.txt"
        desc_info = parse_tuebingen_description(des_path)
        
        semantic_status = "[HAS SEMANTICS]" if desc_info['has_semantic_info'] else "[NO SEMANTICS]"
        print(f"  - {pair_id}: {csv_path.name} (GT: {gt_str}) {semantic_status}")
        
        if desc_info['has_semantic_info']:
            x_short = desc_info['var_x_description'][:40] + "..." if len(desc_info['var_x_description']) > 40 else desc_info['var_x_description']
            y_short = desc_info['var_y_description'][:40] + "..." if len(desc_info['var_y_description']) > 40 else desc_info['var_y_description']
            print(f"      X: {x_short}")
            print(f"      Y: {y_short}")
    
    # Create configuration with LLM enabled
    config_template = {
        # Data type
        'data_type': 'continuous',
        'skip_fci': True,
        'use_llm_prior': True,  # ← KEY: Enable LLM with semantic info!
        
        # Hyperparameters (same as benchmark)
        'learning_rate': 0.05,
        'n_epochs': 200,
        'n_hops': 1,
        'batch_size': None,
        'lambda_group': 0.0,
        'lambda_group_lasso': 0.0,
        'lambda_cycle': 0.05,
        'edge_threshold': 0.035,
        
        # Monitoring
        'monitor_interval': 50,
        'verbose': False,
        'log_interval': 50,
        
        # Output
        'output_dir': str(Path(__file__).parent / 'results'),
        'results_dir': str(Path(__file__).parent / 'results'),
        
        # Advanced
        'random_seed': 42,
        'device': 'cpu',
        'early_stopping': False,
    }
    
    # Run tests with semantic info
    results_with_llm = []
    start_time = time.time()
    
    for idx, (pair_id, csv_path, gt_direction) in enumerate(test_pairs, 1):
        print(f"\n[Test {idx}/{len(test_pairs)}]")
        
        # Parse semantic info for this pair
        des_path = csv_path.parent / f"{pair_id}_des.txt"
        desc_info = parse_tuebingen_description(des_path)
        
        # Add semantic info to config
        config_for_pair = config_template.copy()
        config_for_pair['semantic_info'] = desc_info
        
        results = run_single_pair(pair_id, csv_path, gt_direction, config_for_pair, n_bins=5)
        results_with_llm.append(results)
    
    total_time = time.time() - start_time
    
    # Create results DataFrame
    df_results = pd.DataFrame(results_with_llm)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(__file__).parent / 'results' / f'llm_integration_test_{timestamp}.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    
    # Print analysis
    print("\n" + "=" * 80)
    print("LLM INTEGRATION TEST RESULTS")
    print("=" * 80)
    print(f"\nTotal Runtime: {total_time:.1f}s ({total_time/60:.2f} min)")
    
    successful = df_results[df_results['status'] == 'SUCCESS']
    n_success = len(successful)
    
    if n_success > 0:
        n_correct = successful['direction_correct'].sum()
        accuracy = n_correct / n_success * 100
        avg_gsb = successful['gsb_ratio'].mean()
        avg_diff = successful['diff_ratio'].mean()
        
        print(f"\nProcessed: {n_success}/{len(test_pairs)} pairs successfully")
        print(f"Direction Accuracy: {n_correct}/{n_success} ({accuracy:.1f}%)")
        print(f"Average GSB Ratio: {avg_gsb:.4f}")
        print(f"Average Diff Ratio: {avg_diff:.4f}")
        
        print(f"\nDetailed Results:")
        display_cols = ['pair_id', 'ground_truth', 'learned_direction', 'confidence', 
                       'direction_correct', 'llm_forward_weight', 'llm_backward_weight',
                       'diff_ratio', 'gsb_ratio', 'shd']
        print(df_results[display_cols].to_string(index=False))
        
        # LLM effectiveness evaluation
        print("\n" + "=" * 80)
        print("LLM EFFECTIVENESS EVALUATION")
        print("=" * 80)
        
        if n_correct == 3:
            print("\n[SUCCESS]")
            print("  All 3 'dead hard' pairs are now CORRECT with LLM assistance!")
            print("  -> LLM's soft initialization successfully broke the symmetry.")
            print("  -> Recommendation: Enable LLM for full benchmark run.")
        elif n_correct > 0:
            print(f"\n[PARTIAL SUCCESS]")
            print(f"  {n_correct}/3 pairs correct with LLM.")
            print(f"  -> Some improvement, but not all cases fixed.")
            print(f"  -> Check individual results:")
            for _, row in successful.iterrows():
                status = "[OK]" if row['direction_correct'] else "[WRONG]"
                llm_adv = abs(row['llm_forward_weight'] - row['llm_backward_weight'])
                print(f"    {status} {row['pair_id']}: LLM advantage={llm_adv:.2f}, "
                      f"GSB={row['gsb_ratio']:.4f}, Diff={row['diff_ratio']:.4f}")
        else:
            print("\n[NO IMPROVEMENT]")
            print("  None of the 3 pairs are correct even with LLM.")
            print("  -> LLM's soft initialization was not strong enough.")
            print("  -> Consider:")
            print("    1. Increasing LLM weight scale (currently 0.2 max)")
            print("    2. Trying Method B (conditional prior for weak signals only)")
        
        # Check LLM suggestions
        print("\n" + "=" * 80)
        print("LLM SUGGESTIONS ANALYSIS")
        print("=" * 80)
        for _, row in successful.iterrows():
            llm_forward = row['llm_forward_weight']
            llm_backward = row['llm_backward_weight']
            llm_suggests_forward = llm_forward > llm_backward
            
            # Check if LLM suggestion matches ground truth
            gt_is_forward = 'x->y' in row['ground_truth'].lower()
            llm_correct = (llm_suggests_forward == gt_is_forward)
            
            print(f"\n{row['pair_id']}:")
            print(f"  Ground Truth: {row['ground_truth']}")
            print(f"  LLM Weights: forward={llm_forward:.2f}, backward={llm_backward:.2f}")
            print(f"  LLM Suggests: {'x->y' if llm_suggests_forward else 'y->x'}")
            print(f"  LLM Correct: {'[YES]' if llm_correct else '[NO]'}")
            print(f"  Final Result: {row['learned_direction']} ({'[CORRECT]' if row['direction_correct'] else '[WRONG]'})")
    
    print(f"\n[OUTPUT] Results saved to: {output_path}")
    print("=" * 80)
    
    return df_results


if __name__ == "__main__":
    results_df = main()
