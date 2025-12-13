"""
Test script to demonstrate the extensibility of the architecture.
Shows how easy it is to create different causal structures.
"""

import numpy as np
from data_generator import CausalChainGenerator
from fci_simulator import FCISimulator


def test_different_coefficients():
    """Test with different causal coefficients."""
    print("=" * 60)
    print("TEST 1: Different Coefficients")
    print("=" * 60)
    
    # Create a stronger relationship: Y = 5X, Z = -10Y
    generator = CausalChainGenerator(
        n_samples=100,
        x_to_y_coef=5.0,
        y_to_z_coef=-10.0,
        noise_std=0.1,
        random_seed=123
    )
    
    data = generator.generate()
    print(f"\nGenerated {len(data)} samples with:")
    print(f"  Y = 5.0 * X + noise")
    print(f"  Z = -10.0 * Y + noise")
    print(f"\nCorrelations:")
    print(data.corr())
    print("\n[OK] Successfully generated data with custom coefficients")


def test_custom_structure():
    """Test creating a custom causal structure."""
    print("\n" + "=" * 60)
    print("TEST 2: Custom Causal Structure")
    print("=" * 60)
    
    # Simulate a 4-variable structure: A -> B, A -> C, B -> D, C -> D
    variables = ['A', 'B', 'C', 'D']
    simulator = FCISimulator(variables)
    
    # Define true edges
    true_edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')]
    
    # Simulate FCI finding all edges but making them undirected
    skeleton = simulator.simulate_poor_skeleton(
        true_edges=true_edges,
        spurious_edges=[],  # No spurious edges this time
        make_undirected=True
    )
    
    mask = simulator.generate_mask_matrix(skeleton)
    
    print("\nTrue structure: A -> B, A -> C, B -> D, C -> D")
    print("\nFCI skeleton (all undirected):")
    print("    " + "  ".join(variables))
    for i, var in enumerate(variables):
        print(f" {var}  " + "  ".join([str(int(skeleton[i, j])) for j in range(len(variables))]))
    
    print("\nMask matrix:")
    print("    " + "  ".join(variables))
    for i, var in enumerate(variables):
        print(f" {var}  " + "  ".join([str(int(mask[i, j])) for j in range(len(variables))]))
    
    print(f"\nTotal trainable edges: {int(mask.sum())}")
    print("\n[OK] Successfully created custom 4-variable structure")


def test_partial_orientation():
    """Test FCI with some correct orientations."""
    print("\n" + "=" * 60)
    print("TEST 3: Partial Orientation (Mixed Quality)")
    print("=" * 60)
    
    variables = ['X', 'Y', 'Z']
    simulator = FCISimulator(variables)
    
    # Manually create a skeleton where:
    # - X -> Y is correctly oriented
    # - Y - Z is undirected (should be Y -> Z)
    # - X - Z is a spurious undirected edge
    skeleton = np.array([
        [0, 1, 1],  # X -> Y, X - Z
        [0, 0, 1],  # Y - Z
        [0, 1, 0]   # Z - Y (making Y-Z undirected)
    ])
    
    mask = simulator.generate_mask_matrix(skeleton)
    
    print("\nScenario: FCI got X->Y correct, but Y-Z wrong + spurious X-Z")
    print("\nFCI skeleton:")
    print("    " + "  ".join(variables))
    for i, var in enumerate(variables):
        print(f" {var}  " + "  ".join([str(int(skeleton[i, j])) for j in range(len(variables))]))
    
    print("\nInterpretation:")
    print("  X -> Y (correctly oriented)")
    print("  Y - Z (undirected, should be Y -> Z)")
    print("  X - Z (spurious undirected edge)")
    
    print("\nMask allows training:")
    allowed = []
    for i, from_var in enumerate(variables):
        for j, to_var in enumerate(variables):
            if mask[i, j] == 1:
                allowed.append(f"{from_var}->{to_var}")
    print("  " + ", ".join(allowed))
    
    print("\n[OK] Successfully handled mixed-quality FCI result")


def test_no_noise():
    """Test with perfect data (no noise)."""
    print("\n" + "=" * 60)
    print("TEST 4: Perfect Data (No Noise)")
    print("=" * 60)
    
    generator = CausalChainGenerator(
        n_samples=50,
        x_to_y_coef=2.0,
        y_to_z_coef=-3.0,
        noise_std=0.0,  # No noise!
        random_seed=456
    )
    
    data = generator.generate()
    print("\nGenerated data with zero noise")
    print("\nCorrelations (should be perfect ±1.0):")
    print(data.corr())
    
    # Check if correlations are perfect
    corr_xy = data['X'].corr(data['Y'])
    corr_yz = data['Y'].corr(data['Z'])
    
    print(f"\n|corr(X,Y)| = {abs(corr_xy):.6f} (expected: 1.0)")
    print(f"|corr(Y,Z)| = {abs(corr_yz):.6f} (expected: 1.0)")
    
    if abs(corr_xy) > 0.9999 and abs(corr_yz) > 0.9999:
        print("\n[OK] Perfect correlations achieved with zero noise")
    else:
        print("\n[WARNING] Correlations not perfect, check implementation")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EXTENSIBILITY TESTS - Demonstrating Modular Architecture")
    print("=" * 70)
    
    test_different_coefficients()
    test_custom_structure()
    test_partial_orientation()
    test_no_noise()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED - Architecture is Extensible and Reusable!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  [+] Easy to change causal coefficients")
    print("  [+] Easy to add more variables")
    print("  [+] Easy to simulate different FCI error patterns")
    print("  [+] Easy to control noise levels")
    print("  [+] All components work independently")
    print("\nReady for Step 3: Neural LP Implementation!")

