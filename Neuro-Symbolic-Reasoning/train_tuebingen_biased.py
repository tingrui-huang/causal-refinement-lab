"""
Single Run Training with Initial Bias (0.55 vs 0.45)

This script runs a single training with slight initial bias toward the correct direction.
With 10 bins and uniform discretization, we test if the model can learn from data.
"""

import sys
from pathlib import Path
import torch
import importlib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force reload config to get latest values
import config
importlib.reload(config)
from config import get_training_config, DATASET, print_config
from train_complete import train_complete


def main():
    """Run training with initial bias"""
    
    # Make sure we're using Tuebingen dataset
    if DATASET != 'tuebingen_pair1':
        print("\n[ERROR] Please set DATASET = 'tuebingen_pair1' in config.py")
        print(f"Current dataset: {DATASET}")
        sys.exit(1)
    
    # Print configuration
    print_config()
    
    # Get training configuration
    config = get_training_config()
    
    # Set initial bias: 0.55 vs 0.45 (slight favor to correct direction)
    config['forward_bias'] = 0.55
    
    print("\n" + "=" * 80)
    print("TUEBINGEN TRAINING: With Initial Bias (0.55 vs 0.45)")
    print("=" * 80)
    print("Configuration:")
    print(f"  Bins: 5 (better sample density per bin)")
    print(f"  Strategy: Uniform (preserves density asymmetry)")
    print(f"  Initial Bias: Altitude->Temp = 0.55, Temp->Alt = 0.45")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Lambda Group Lasso: {config['lambda_group']}")
    print(f"  Lambda Cycle: {config['lambda_cycle']}")
    print(f"  Edge Threshold: {config['edge_threshold']}")
    print(f"  Epochs: {config['n_epochs']}")
    print(f"  Random Noise: DISABLED (pure gradient-based learning)")
    print("=" * 80)
    
    # Train
    model, metrics, history = train_complete(config)
    
    # === ANALYZE DIRECTION LEARNING ===
    print("\n" + "=" * 80)
    print("DIRECTION LEARNING ANALYSIS")
    print("=" * 80)
    
    # Get final adjacency matrix
    with torch.no_grad():
        adj = model.get_adjacency().numpy()
    
    # Extract direction strengths
    # With 5 bins: Altitude (states 0-4), Temperature (states 5-9)
    altitude_to_temp = adj[0:5, 5:10]  # Should be STRONG
    temp_to_altitude = adj[5:10, 0:5]  # Should be WEAK
    
    forward_strength = altitude_to_temp.sum()
    backward_strength = temp_to_altitude.sum()
    ratio = forward_strength / backward_strength if backward_strength > 0 else float('inf')
    
    # Calculate gap change
    initial_gap = 0.55 - 0.45  # 0.1
    initial_gap_total = initial_gap * 25  # 25 = 5x5 connections
    final_gap = forward_strength - backward_strength
    gap_change = final_gap - initial_gap_total
    
    print(f"\nInitial Settings:")
    print(f"  Forward Bias: 0.55")
    print(f"  Backward Bias: 0.45")
    print(f"  Initial Gap: {initial_gap_total:.2f} (0.1 × 25 connections)")
    
    print(f"\nFinal Results:")
    print(f"  Altitude -> Temperature: {forward_strength:.4f}")
    print(f"  Temperature -> Altitude: {backward_strength:.4f}")
    print(f"  Final Gap: {final_gap:.4f}")
    print(f"  Gap Change: {gap_change:+.4f}")
    print(f"  Ratio (Forward/Backward): {ratio:.4f}")
    
    # Success check
    print(f"\nSuccess Criteria:")
    success_1 = forward_strength > backward_strength
    success_2 = ratio >= 2.0
    success_3 = gap_change > 0
    
    print(f"  1. Forward > Backward: {'[PASS]' if success_1 else '[FAIL]'}")
    print(f"  2. Ratio >= 2:1: {'[PASS]' if success_2 else '[FAIL]'}")
    print(f"  3. Gap Increased: {'[PASS]' if success_3 else '[FAIL]'}")
    
    overall_success = success_1 and success_2 and success_3
    print(f"\n{'='*80}")
    if overall_success:
        print("SUCCESS: Model learned correct causal direction!")
        print(f"  - Started with weak bias (0.55 vs 0.45)")
        print(f"  - Achieved strong direction (ratio {ratio:.2f}:1)")
        print(f"  - Gap increased by {gap_change:.4f}")
    elif success_1 and success_3:
        print("PARTIAL SUCCESS: Model learned correct direction but ratio < 2:1")
        print("  Suggestions:")
        print("  - Increase training epochs")
        print("  - Increase lambda_cycle")
    else:
        print("NEEDS TUNING: Model did not learn clear direction")
        print("  Suggestions:")
        print("  - Check gradient ratios during training")
        print("  - Increase lambda_cycle (stronger cycle penalty)")
        print("  - Increase epochs or adjust learning rate")
    print("=" * 80)
    
    # Print training summary
    print("\nTraining Summary:")
    print(f"  Final Loss: {history['loss_total'][-1]:.4f}")
    print(f"  Reconstruction: {history['loss_reconstruction'][-1]:.4f}")
    print(f"  Group Lasso: {history['loss_group_lasso'][-1]:.4f}")
    print(f"  Cycle: {history['loss_cycle'][-1]:.4f}")
    print(f"  Bidirectional Ratio: {history['bidirectional_ratio'][0]*100:.1f}% -> {history['bidirectional_ratio'][-1]*100:.1f}%")
    
    return model, metrics, history


if __name__ == "__main__":
    model, metrics, history = main()



