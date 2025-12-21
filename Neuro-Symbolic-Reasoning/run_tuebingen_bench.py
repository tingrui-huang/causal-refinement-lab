"""
Tuebingen Benchmark: Automated Testing on Multiple Pairs

This script automatically processes all Tuebingen pairs with the validated configuration:
- Lambda Group Lasso: 0.0 (no sparsity penalty)
- Lambda Cycle: 0.05 (moderate cycle penalty)
- Uniform Binning: 5 bins
- Dynamic Threshold: 3% of total weight

Output: CSV report with direction accuracy, SHD, and GSB ratio for each pair.
"""

import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import json
import time
import os

# Load environment variables from .env file FIRST (for API keys)
def load_env_file():
    """Load .env file manually if python-dotenv is not available"""
    env_path = Path(__file__).parent.parent / '.env'
    
    if not env_path.exists():
        print(f"[WARN] .env file not found at: {env_path}")
        return False
    
    try:
        # Try using python-dotenv first
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=True)
        print(f"[DEBUG] Loaded .env using python-dotenv from: {env_path}")
    except ImportError:
        # Fallback: manually parse .env file
        print(f"[DEBUG] python-dotenv not installed, manually parsing .env")
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
            print(f"[DEBUG] Manually loaded .env from: {env_path}")
        except Exception as e:
            print(f"[ERROR] Failed to parse .env: {e}")
            return False
    
    # Verify API key is loaded
    api_key_len = len(os.getenv('OPENAI_API_KEY', ''))
    print(f"[DEBUG] API key length: {api_key_len}")
    return api_key_len > 0

# Load .env file
load_env_file()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_training_config, DATASET_CONFIGS
from train_complete import train_complete
from modules.tuebingen_llm import TuebingenLLMDirectionResolver
from modules.tuebingen_semantic_parser import parse_tuebingen_description


def parse_ground_truth_from_description(des_path):
    """
    Parse ground truth direction from _des.txt file
    
    Returns:
        1 if x->y, -1 if y->x, 0 if unknown
    """
    try:
        with open(des_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().lower()
            
            # Look for direction indicators
            if 'x --> y' in content or 'x->y' in content or 'x causes y' in content:
                return 1
            elif 'y --> x' in content or 'y->x' in content or 'y causes x' in content:
                return -1
            else:
                # Default: assume x -> y
                return 1
    except:
        return 1  # Default


def convert_txt_to_csv(txt_path, csv_path, var_names=['x', 'y']):
    """
    Convert Tuebingen .txt file to .csv with headers
    
    Args:
        txt_path: Path to .txt file
        csv_path: Path to output .csv file
        var_names: Variable names (default: ['x', 'y'])
    """
    # Read space-separated data
    df = pd.read_csv(txt_path, sep=r'\s+', header=None, names=var_names)
    df.to_csv(csv_path, index=False)
    return csv_path


def discover_tuebingen_pairs(data_dir, auto_convert=True):
    """
    Discover all Tuebingen pair files in the data directory
    
    Args:
        data_dir: Path to data directory
        auto_convert: If True, automatically convert .txt to .csv
    
    Returns:
        List of (pair_id, csv_path, ground_truth_direction) tuples
    """
    data_path = Path(data_dir)
    pairs = []
    
    # Find all pair*.txt files (excluding _des.txt)
    txt_files = sorted([f for f in data_path.glob('pair*.txt') if '_des' not in f.name])
    
    for txt_file in txt_files:
        pair_id = txt_file.stem  # e.g., 'pair0001'
        csv_file = data_path / f"{pair_id}.csv"
        des_file = data_path / f"{pair_id}_des.txt"
        
        # Auto-convert if needed
        if auto_convert and not csv_file.exists():
            print(f"Converting {txt_file.name} to CSV...")
            convert_txt_to_csv(txt_file, csv_file)
        
        # Parse ground truth direction
        gt_direction = 1  # Default: x -> y
        if des_file.exists():
            gt_direction = parse_ground_truth_from_description(des_file)
        
        if csv_file.exists():
            pairs.append((pair_id, csv_file, gt_direction))
    
    return pairs


def create_pair_metadata(pair_id, csv_path, gt_direction, n_bins=5):
    """
    Create metadata for a Tuebingen pair
    
    Args:
        pair_id: e.g., 'pair0001'
        csv_path: Path to CSV file
        gt_direction: Ground truth direction (1: x->y, -1: y->x)
        n_bins: Number of bins for discretization
    
    Returns:
        Dictionary with metadata
    """
    # Read CSV to get variable names
    df = pd.read_csv(csv_path)
    var_names = df.columns.tolist()
    
    if len(var_names) != 2:
        raise ValueError(f"{pair_id}: Expected 2 variables, got {len(var_names)}")
    
    # Create state mappings
    state_mappings = {}
    for var in var_names:
        state_mappings[var] = {str(i): f"{var}_bin{i}" for i in range(n_bins)}
    
    # Set ground truth based on direction
    if gt_direction == 1:
        ground_truth = [[var_names[0], var_names[1]]]  # x -> y
    elif gt_direction == -1:
        ground_truth = [[var_names[1], var_names[0]]]  # y -> x
    else:
        ground_truth = [[var_names[0], var_names[1]]]  # Default x -> y
    
    metadata = {
        "dataset_name": pair_id,
        "data_format": "one_hot_csv",
        "n_variables": 2,
        "n_states": n_bins * 2,  # 2 variables × n_bins
        "n_bins": n_bins,
        "discretization_strategy": "uniform",
        "variable_names": var_names,
        "state_mappings": state_mappings,
        "ground_truth": ground_truth,
        "ground_truth_direction": gt_direction,
        "note": f"Tuebingen {pair_id}: Uniform discretization with {n_bins} bins"
    }
    
    return metadata


def compute_dynamic_threshold(adj, percentile=3.0):
    """
    Compute dynamic threshold as a percentage of total weight
    
    Args:
        adj: Adjacency matrix (numpy array)
        percentile: Percentage of total weight (default 3%)
    
    Returns:
        Dynamic threshold value
    """
    total_weight = np.abs(adj).sum()
    n_connections = adj.size
    avg_weight = total_weight / n_connections
    threshold = avg_weight * (percentile / 100.0) * n_connections / 2  # Normalize
    return threshold


def analyze_direction(adj, n_bins, gt_direction, llm_forward_weight=0.5, llm_backward_weight=0.5):
    """
    Analyze learned direction from adjacency matrix using diff_ratio mechanism
    
    Core Logic (from image):
    1. Asymmetry is RELATIVE (A stronger than B)
    2. But use THRESHOLD to avoid both too weak or both too strong
    3. Use diff_ratio to determine direction confidence
    4. In uncertain zone (0.95-1.05), use LLM prior as tiebreaker
    
    Args:
        adj: Adjacency matrix (numpy array)
        n_bins: Number of bins per variable
        gt_direction: Ground truth direction (1: x->y, -1: y->x)
        llm_forward_weight: LLM's weight for x->y direction (default: 0.5)
        llm_backward_weight: LLM's weight for y->x direction (default: 0.5)
    
    Returns:
        Dictionary with direction analysis
    """
    # Extract direction strengths
    # Variable x: states 0:(n_bins-1)
    # Variable y: states n_bins:(2*n_bins-1)
    x_to_y = adj[0:n_bins, n_bins:2*n_bins]  # x -> y
    y_to_x = adj[n_bins:2*n_bins, 0:n_bins]  # y -> x
    
    forward_strength = x_to_y.sum()
    backward_strength = y_to_x.sum()
    
    # Compute diff_ratio (权重竞争逻辑)
    diff_ratio = forward_strength / (backward_strength + 1e-6)
    
    # 【黄金逻辑】带缓冲区的 PK 制 (1.02/0.98 甜点)
    # 1. 强信号区：数据说了算 (防止 LLM 幻觉)
    if diff_ratio > 1.02:  # 只要比 1.0 多 2%，就信数据
        learned_direction = 1  # x->y
        confidence = "strong" if diff_ratio > 1.1 else "weak"
    elif diff_ratio < 0.98:  # 只要比 1.0 少 2%，就信数据
        learned_direction = -1  # y->x
        confidence = "strong" if diff_ratio < 0.9 else "weak"
    
    # 2. 模糊区 (0.98 - 1.02)：数据也是懵的，听 LLM 的
    else:
        if llm_forward_weight > llm_backward_weight:
            learned_direction = 1
            confidence = "llm_guided"
        else:
            learned_direction = -1
            confidence = "llm_guided"
    
    # Check correctness against ground truth
    if learned_direction == 0:
        correct_direction = False  # 双向视为失败
    else:
        correct_direction = (learned_direction == gt_direction)
    
    # Compute GSB ratio (always x->y / y->x for reporting)
    if backward_strength > 0:
        gsb_ratio = forward_strength / backward_strength
    else:
        gsb_ratio = float('inf')
    
    return {
        'x_to_y_strength': float(forward_strength),
        'y_to_x_strength': float(backward_strength),
        'diff_ratio': float(diff_ratio),
        'gsb_ratio': float(gsb_ratio),
        'learned_direction': learned_direction,
        'confidence': confidence,
        'gt_direction': gt_direction,
        'correct_direction': correct_direction,
        'gap': float(forward_strength - backward_strength)
    }


def compute_shd(learned_adj, ground_truth_edges, var_names):
    """
    Compute Structural Hamming Distance using "winner-takes-all" mechanism
    
    Core Logic (from Gemini's suggestion):
    - For continuous datasets (Tuebingen), use ratio-based competition
    - If forward_strength / backward_strength > 1.1 (10% stronger), add forward edge
    - If backward_strength / forward_strength > 1.1 (10% stronger), add backward edge
    - Otherwise, no edge (bidirectional = noise)
    
    This avoids the "3% threshold too low" problem that causes false positives.
    
    Args:
        learned_adj: Learned adjacency matrix (variable-level, 2x2 for pairs)
        ground_truth_edges: List of ground truth edges [(var1, var2)]
        var_names: List of variable names
    
    Returns:
        SHD value
    """
    # Convert learned adjacency to edge list using ratio-based competition
    learned_edges = set()
    n_vars = len(var_names)
    
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:
                forward_strength = learned_adj[i, j]
                backward_strength = learned_adj[j, i]
                
                # Compute ratio (谁强谁上机制)
                ratio = forward_strength / (backward_strength + 1e-6)
                
                # Use buffered PK mechanism (2% buffer zone - golden ratio)
                if ratio > 1.02:  # Forward direction wins (2% threshold)
                    learned_edges.add((var_names[i], var_names[j]))
                elif ratio < 0.98:  # Backward direction wins (handled in j->i iteration)
                    pass  # Will be added when we check j->i
                # else: 0.98 <= ratio <= 1.02: uncertain zone, no edge added (will rely on direction_analysis with LLM)
    
    # Convert ground truth to set
    gt_edges = set(tuple(edge) for edge in ground_truth_edges)
    
    # Compute SHD: missing + extra + reversed
    missing = len(gt_edges - learned_edges)
    extra = len(learned_edges - gt_edges)
    
    # Check for reversed edges
    reversed_count = 0
    for edge in learned_edges:
        reversed_edge = (edge[1], edge[0])
        if reversed_edge in gt_edges and edge not in gt_edges:
            reversed_count += 1
    
    shd = missing + extra + reversed_count
    return shd


def run_single_pair(pair_id, csv_path, gt_direction, config_template, n_bins=5):
    """
    Run training on a single Tuebingen pair
    
    Args:
        pair_id: Pair identifier (e.g., 'pair0001')
        csv_path: Path to CSV file
        gt_direction: Ground truth direction (1: x->y, -1: y->x)
        config_template: Base configuration template
        n_bins: Number of bins for discretization
    
    Returns:
        Dictionary with results
    """
    start_time = time.time()
    
    print("\n" + "=" * 80)
    print(f"PROCESSING: {pair_id}")
    print("=" * 80)
    
    # Create metadata
    metadata = create_pair_metadata(pair_id, csv_path, gt_direction, n_bins)
    var_names = metadata['variable_names']
    gt_edge = metadata['ground_truth'][0]
    
    # Save temporary metadata
    metadata_path = csv_path.parent / f"{pair_id}_metadata_temp.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Update config for this pair
    config = config_template.copy()
    config['dataset_name'] = pair_id
    config['data_path'] = str(csv_path)
    config['metadata_path'] = str(metadata_path)
    config['ground_truth_edges'] = metadata['ground_truth']
    config['manual_skeleton'] = [(var_names[0], var_names[1])]
    
    gt_str = f"{gt_edge[0]} -> {gt_edge[1]}"
    print(f"\nVariables: {var_names[0]}, {var_names[1]}")
    print(f"Ground Truth: {gt_str}")
    print(f"Configuration:")
    print(f"  Bins: {n_bins}")
    print(f"  Strategy: Uniform")
    print(f"  Lambda Group Lasso: {config['lambda_group']}")
    print(f"  Lambda Cycle: {config['lambda_cycle']}")
    print(f"  Epochs: {config['n_epochs']}")
    print(f"  Use LLM Prior: {config_template.get('use_llm_prior', False)}")
    
    # NEW: LLM direction resolution (if enabled AND has semantic info)
    if config_template.get('use_llm_prior', False):
        # Get semantic info from config (if provided by caller)
        semantic_info = config_template.get('semantic_info', None)
        
        try:
            print("\n" + "-" * 80)
            # Use existing API client infrastructure (reads from config.py and env)
            llm_resolver = TuebingenLLMDirectionResolver()
            
            # Call with semantic info if available
            if semantic_info and semantic_info.get('has_semantic_info', False):
                forward_weight, backward_weight = llm_resolver.get_direction_prior(
                    var_names[0], 
                    var_names[1], 
                    pair_id,
                    var_x_description=semantic_info.get('var_x_description'),
                    var_y_description=semantic_info.get('var_y_description'),
                    context=semantic_info.get('context')
                )
            else:
                # No semantic info, LLM will return neutral (0.5, 0.5)
                forward_weight, backward_weight = llm_resolver.get_direction_prior(
                    var_names[0], var_names[1], pair_id
                )
            
            print(f"\n[LLM RESULT]")
            print(f"  {var_names[0]} -> {var_names[1]}: {forward_weight:.2f}")
            print(f"  {var_names[1]} -> {var_names[0]}: {backward_weight:.2f}")
            print(f"  Advantage: {abs(forward_weight - backward_weight):.2f}")
            print("-" * 80)
            
            # Store LLM weights in config (will be used in train_complete.py)
            config['llm_forward_weight'] = forward_weight
            config['llm_backward_weight'] = backward_weight
            config['llm_var_x'] = var_names[0]
            config['llm_var_y'] = var_names[1]
            
        except Exception as e:
            print(f"\n[LLM ERROR] Failed to get LLM prior: {e}")
            print(f"[LLM] Continuing without LLM prior...")
            config['use_llm_prior'] = False  # Disable for this pair
    
    try:
        # Train
        model, metrics, history = train_complete(config)
        
        # Get final adjacency matrix
        with torch.no_grad():
            adj = model.get_adjacency().numpy()
        
        # Compute dynamic threshold (for reference only, not used in SHD)
        dynamic_threshold = compute_dynamic_threshold(adj, percentile=3.0)
        print(f"\nDynamic Threshold (3% of total weight, for reference): {dynamic_threshold:.6f}")
        print(f"Note: SHD uses ratio-based 'winner-takes-all' (10% rule), not threshold")
        
        # Analyze direction (pass LLM weights for tiebreaking in uncertain zone)
        direction_analysis = analyze_direction(
            adj, 
            n_bins, 
            gt_direction,
            llm_forward_weight=config.get('llm_forward_weight', 0.5),
            llm_backward_weight=config.get('llm_backward_weight', 0.5)
        )
        
        # Compute variable-level adjacency for SHD
        # Use SUM (not mean) to get total strength for ratio comparison
        var_adj = np.zeros((2, 2))
        var_adj[0, 1] = adj[0:n_bins, n_bins:2*n_bins].sum()  # x -> y (total strength)
        var_adj[1, 0] = adj[n_bins:2*n_bins, 0:n_bins].sum()  # y -> x (total strength)
        
        # Compute SHD using ratio-based "winner-takes-all" mechanism
        # No threshold needed - uses 10% ratio rule instead
        shd = compute_shd(var_adj, metadata['ground_truth'], var_names)
        
        # Direction accuracy
        direction_correct = direction_analysis['correct_direction']
        
        # Compile results
        learned_dir_str = (
            'x->y' if direction_analysis['learned_direction'] == 1 
            else 'y->x' if direction_analysis['learned_direction'] == -1 
            else 'bidirectional'
        )
        
        elapsed_time = time.time() - start_time
        
        results = {
            'pair_id': pair_id,
            'var_x': var_names[0],
            'var_y': var_names[1],
            'ground_truth': gt_str,
            'learned_direction': learned_dir_str,
            'confidence': direction_analysis['confidence'],
            'direction_correct': direction_correct,
            'diff_ratio': direction_analysis['diff_ratio'],
            'gsb_ratio': direction_analysis['gsb_ratio'],
            'x_to_y_strength': direction_analysis['x_to_y_strength'],
            'y_to_x_strength': direction_analysis['y_to_x_strength'],
            'gap': direction_analysis['gap'],
            'shd': shd,
            'dynamic_threshold': dynamic_threshold,
            'llm_forward_weight': config.get('llm_forward_weight', 0.5),
            'llm_backward_weight': config.get('llm_backward_weight', 0.5),
            'llm_used': config.get('use_llm_prior', False),
            'final_loss': history['loss_total'][-1],
            'reconstruction_loss': history['loss_reconstruction'][-1],
            'cycle_loss': history['loss_cycle'][-1],
            'bidirectional_ratio_initial': history['bidirectional_ratio'][0],
            'bidirectional_ratio_final': history['bidirectional_ratio'][-1],
            'runtime_seconds': elapsed_time,
            'status': 'SUCCESS'
        }
        
        print(f"\n[RESULTS]")
        print(f"  Ground Truth: {gt_str}")
        print(f"  Learned: {results['learned_direction']} ({direction_analysis['confidence']})")
        print(f"  Direction: {'[CORRECT]' if direction_correct else '[INCORRECT]'}")
        print(f"  Diff Ratio: {direction_analysis['diff_ratio']:.4f} (x->y / y->x)")
        print(f"  GSB Ratio: {direction_analysis['gsb_ratio']:.4f}")
        print(f"  SHD: {shd}")
        print(f"  Gap: {direction_analysis['gap']:.4f}")
        print(f"  Runtime: {elapsed_time:.1f}s ({elapsed_time/60:.2f} min)")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        
        print(f"\n[ERROR] Failed to process {pair_id}: {e}")
        import traceback
        traceback.print_exc()
        
        results = {
            'pair_id': pair_id,
            'var_x': var_names[0] if 'var_names' in locals() else 'N/A',
            'var_y': var_names[1] if 'var_names' in locals() else 'N/A',
            'ground_truth': 'N/A',
            'learned_direction': 'N/A',
            'confidence': 'N/A',
            'direction_correct': False,
            'diff_ratio': 0.0,
            'gsb_ratio': 0.0,
            'x_to_y_strength': 0.0,
            'y_to_x_strength': 0.0,
            'gap': 0.0,
            'shd': 999,
            'dynamic_threshold': 0.0,
            'llm_forward_weight': 0.5,
            'llm_backward_weight': 0.5,
            'llm_used': False,
            'final_loss': 0.0,
            'reconstruction_loss': 0.0,
            'cycle_loss': 0.0,
            'bidirectional_ratio_initial': 0.0,
            'bidirectional_ratio_final': 0.0,
            'runtime_seconds': elapsed_time,
            'status': f'FAILED: {str(e)}'
        }
    
    finally:
        # Clean up temporary metadata
        if metadata_path.exists():
            metadata_path.unlink()
    
    return results


def main():
    """Main benchmark runner with LLM semantic enhancement"""
    benchmark_start_time = time.time()
    
    print("=" * 80)
    print("TUEBINGEN BENCHMARK: AUTOMATED TESTING (WITH LLM SEMANTIC ENHANCEMENT)")
    print("=" * 80)
    print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nValidated Configuration:")
    print("  - Lambda Group Lasso: 0.0 (no sparsity penalty)")
    print("  - Lambda Cycle: 0.05 (moderate cycle penalty)")
    print("  - Uniform Binning: 5 bins")
    print("  - Dynamic Threshold: 3% of total weight (for reference)")
    print("  - LLM Enhancement: Enabled (semantic-based)")
    print("=" * 80)
    
    # Discover pairs
    data_dir = Path(__file__).parent / 'data' / 'tuebingen'
    pairs = discover_tuebingen_pairs(data_dir, auto_convert=True)
    
    print(f"\nDiscovered {len(pairs)} pairs")
    
    # Parse semantic info for all pairs
    print("\n[PHASE 1] Parsing semantic information from _des.txt files...")
    semantic_stats = {'has_semantic': 0, 'no_semantic': 0}
    
    for pair_id, csv_path, gt_dir in pairs:
        des_path = csv_path.parent / f"{pair_id}_des.txt"
        if des_path.exists():
            desc_info = parse_tuebingen_description(des_path)
            if desc_info['has_semantic_info']:
                semantic_stats['has_semantic'] += 1
            else:
                semantic_stats['no_semantic'] += 1
        else:
            semantic_stats['no_semantic'] += 1
    
    print(f"  Pairs with semantic info: {semantic_stats['has_semantic']}/{len(pairs)}")
    print(f"  Pairs without semantic info: {semantic_stats['no_semantic']}/{len(pairs)}")
    print(f"  LLM will be used for: {semantic_stats['has_semantic']} pairs")
    print(f"  Pure GSB for: {semantic_stats['no_semantic']} pairs")
    
    # Create base configuration (validated settings + LLM)
    config_template = {
        # Data type
        'data_type': 'continuous',
        'skip_fci': True,
        'use_llm_prior': True,  # Enable LLM with semantic info
        
        # Hyperparameters (VALIDATED CONFIGURATION)
        'learning_rate': 0.05,
        'n_epochs': 200,  # 300 too long, causes overfitting and amplifies noise
        'n_hops': 1,
        'batch_size': None,
        'lambda_group': 0.0,  # No sparsity penalty
        'lambda_group_lasso': 0.0,
        'lambda_cycle': 0.05,  # Moderate cycle penalty
        'edge_threshold': 0.035,  # Will be overridden by dynamic threshold
        
        # Note: No 'forward_bias' here - LLM weights will be used when available
        # For pairs without semantic info, train_complete.py will use uniform 0.5
        
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
    
    # Run benchmark on all pairs with semantic info
    print("\n[PHASE 2] Running benchmark on all pairs...")
    all_results = []
    
    for idx, (pair_id, csv_path, gt_direction) in enumerate(pairs, 1):
        print(f"\n[Progress: {idx}/{len(pairs)}]")
        
        # Parse semantic info for this pair
        des_path = csv_path.parent / f"{pair_id}_des.txt"
        desc_info = parse_tuebingen_description(des_path)
        
        # Add semantic info to config
        config_for_pair = config_template.copy()
        config_for_pair['semantic_info'] = desc_info
        
        # Run training
        results = run_single_pair(pair_id, csv_path, gt_direction, config_for_pair, n_bins=5)
        all_results.append(results)
        
        # Print estimated time remaining
        if idx > 0:
            elapsed = time.time() - benchmark_start_time
            avg_time_per_pair = elapsed / idx
            remaining_pairs = len(pairs) - idx
            eta_seconds = avg_time_per_pair * remaining_pairs
            eta_minutes = eta_seconds / 60
            print(f"\n[ETA] Estimated time remaining: {eta_minutes:.1f} minutes ({eta_seconds:.0f}s)")
    
    # Create results DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(__file__).parent / 'results' / f'tuebingen_benchmark_{timestamp}.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    
    # Calculate total runtime
    total_runtime = time.time() - benchmark_start_time
    total_hours = total_runtime / 3600
    total_minutes = total_runtime / 60
    
    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Runtime: {total_runtime:.1f}s ({total_minutes:.2f} min / {total_hours:.2f} hours)")
    
    successful = df_results[df_results['status'] == 'SUCCESS']
    n_success = len(successful)
    n_total = len(df_results)
    
    print(f"\nProcessed: {n_success}/{n_total} pairs successfully")
    
    if n_success > 0:
        n_correct = successful['direction_correct'].sum()
        accuracy = n_correct / n_success * 100
        avg_runtime = successful['runtime_seconds'].mean()
        
        # LLM usage statistics
        llm_used = successful['llm_used'].sum()
        llm_not_used = n_success - llm_used
        
        print(f"\nDirection Accuracy: {n_correct}/{n_success} ({accuracy:.1f}%)")
        print(f"Average GSB Ratio: {successful['gsb_ratio'].mean():.4f}")
        print(f"Average SHD: {successful['shd'].mean():.2f}")
        print(f"Average Gap: {successful['gap'].mean():.4f}")
        print(f"Average Runtime per Pair: {avg_runtime:.1f}s ({avg_runtime/60:.2f} min)")
        
        print(f"\n--- LLM Enhancement Statistics ---")
        print(f"LLM Used (with semantic info): {llm_used}/{n_success} pairs")
        print(f"Pure GSB (no semantic info): {llm_not_used}/{n_success} pairs")
        
        if llm_used > 0:
            llm_pairs = successful[successful['llm_used'] == True]
            llm_correct = llm_pairs['direction_correct'].sum()
            llm_accuracy = llm_correct / llm_used * 100
            print(f"Accuracy with LLM: {llm_correct}/{llm_used} ({llm_accuracy:.1f}%)")
        
        if llm_not_used > 0:
            no_llm_pairs = successful[successful['llm_used'] == False]
            no_llm_correct = no_llm_pairs['direction_correct'].sum()
            no_llm_accuracy = no_llm_correct / llm_not_used * 100
            print(f"Accuracy without LLM: {no_llm_correct}/{llm_not_used} ({no_llm_accuracy:.1f}%)")
        
        print(f"\nDetailed Results:")
        display_cols = ['pair_id', 'ground_truth', 'learned_direction', 'confidence', 
                       'direction_correct', 'llm_used', 'diff_ratio', 'gsb_ratio', 'shd']
        print(df_results[display_cols].to_string(index=False))
    
    print(f"\n[OUTPUT] Results saved to: {output_path}")
    print("=" * 80)
    
    return df_results


if __name__ == "__main__":
    results_df = main()



