"""
Training Script for Tuebingen Dataset

This script tests the model's ability to learn causal direction from data alone,
without FCI or LLM priors. The task is simple:
- 2 variables: Altitude, Temperature
- Ground truth: Altitude -> Temperature
- Manual skeleton: Altitude <-> Temperature (bidirectional, symmetric)
- Goal: Model should learn that Altitude -> Temperature is stronger than Temperature -> Altitude

Success criteria:
1. adj[Altitude_states, Temperature_states].sum() > adj[Temperature_states, Altitude_states].sum()
2. Ratio should be at least 2:1 (forward >> backward)
3. Cycle consistency loss should decrease (no cycles)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_training_config, DATASET, print_config
from train_complete import train_complete  # Import from same directory
import torch

def main():
    """Run training on Tuebingen dataset"""
    
    # Print configuration
    print_config()
    
    # Get training configuration
    config = get_training_config()
    
    print("\n" + "=" * 80)
    print("TUEBINGEN TRAINING: PURE DATA-DRIVEN DIRECTION LEARNING")
    print("=" * 80)
    print("Task: Learn that Altitude -> Temperature (not Temperature -> Altitude)")
    print("Setup:")
    print("  - Manual skeleton: Altitude <-> Temperature (SYMMETRIC)")
    print("  - No FCI, No LLM")
    print("  - Model must break symmetry using data alone")
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
    # Altitude: states 0-4, Temperature: states 5-9
    altitude_to_temp = adj[0:5, 5:10]  # Should be STRONG
    temp_to_altitude = adj[5:10, 0:5]  # Should be WEAK
    
    forward_strength = altitude_to_temp.sum()
    backward_strength = temp_to_altitude.sum()
    ratio = forward_strength / backward_strength if backward_strength > 0 else float('inf')
    
    print(f"\nDirection Strengths:")
    print(f"  Altitude -> Temperature: {forward_strength:.4f}")
    print(f"  Temperature -> Altitude: {backward_strength:.4f}")
    print(f"  Ratio (forward/backward): {ratio:.2f}:1")
    
    # Success check
    print(f"\nSuccess Criteria:")
    success_1 = forward_strength > backward_strength
    success_2 = ratio >= 2.0
    print(f"  1. Forward > Backward: {'[PASS]' if success_1 else '[FAIL]'}")
    print(f"  2. Ratio >= 2:1: {'[PASS]' if success_2 else '[FAIL]'}")
    
    overall_success = success_1 and success_2
    print(f"\n{'='*80}")
    if overall_success:
        print("SUCCESS: Model learned correct causal direction from data!")
    else:
        print("NEEDS TUNING: Model did not learn clear direction")
        print("Suggestions:")
        print("  - Increase lambda_cycle (stronger cycle penalty)")
        print("  - Increase n_epochs (more training time)")
        print("  - Adjust learning rate")
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
    # Make sure we're using Tuebingen dataset
    if DATASET != 'tuebingen_pair1':
        print("\n[ERROR] Please set DATASET = 'tuebingen_pair1' in config.py")
        print(f"Current dataset: {DATASET}")
        sys.exit(1)
    
    model, metrics, history = main()



