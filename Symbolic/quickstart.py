"""
Quick Start Guide for Symbolic Causal Refinement

Run this script to get started with the Neural LP causal refinement pipeline.
"""

import os
import sys


def check_dependencies():
    """Check if required packages are installed."""
    required = ['numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("[WARNING] Missing packages:", ", ".join(missing))
        print("Install with: pip install " + " ".join(missing))
        return False
    return True


def run_step_1_2():
    """Run Step 1 & 2: Data generation and FCI simulation."""
    print("\n" + "=" * 70)
    print("STEP 1 & 2: Generate Data and Simulate FCI")
    print("=" * 70)
    
    import subprocess
    result = subprocess.run([sys.executable, 'step1_2_generate_data.py'], 
                          capture_output=False)
    return result.returncode == 0


def run_visualization():
    """Run visualization script."""
    print("\n" + "=" * 70)
    print("VISUALIZATION: Creating Plots")
    print("=" * 70)
    
    import subprocess
    result = subprocess.run([sys.executable, 'visualize_setup.py'], 
                          capture_output=False)
    return result.returncode == 0


def show_summary():
    """Show project summary."""
    print("\n" + "=" * 70)
    print("PROJECT SUMMARY")
    print("=" * 70)
    
    import subprocess
    subprocess.run([sys.executable, 'summary.py'], capture_output=False)


def interactive_menu():
    """Show interactive menu."""
    while True:
        print("\n" + "=" * 70)
        print("SYMBOLIC CAUSAL REFINEMENT - Quick Start Menu")
        print("=" * 70)
        print("\n1. Run Step 1 & 2 (Generate data + Simulate FCI)")
        print("2. Visualize results")
        print("3. Show project summary")
        print("4. Run extensibility tests")
        print("5. Load and inspect data (Python shell)")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            if run_step_1_2():
                print("\n[OK] Step 1 & 2 completed successfully!")
            else:
                print("\n[ERROR] Step 1 & 2 failed!")
        
        elif choice == '2':
            if run_visualization():
                print("\n[OK] Visualizations created!")
                print("Check results/ folder for plots.")
            else:
                print("\n[ERROR] Visualization failed!")
        
        elif choice == '3':
            show_summary()
        
        elif choice == '4':
            print("\n" + "=" * 70)
            print("EXTENSIBILITY TESTS")
            print("=" * 70)
            import subprocess
            subprocess.run([sys.executable, 'test_extensibility.py'], 
                         capture_output=False)
        
        elif choice == '5':
            print("\n" + "=" * 70)
            print("PYTHON SHELL - Data Inspection")
            print("=" * 70)
            print("\nLoading data...")
            
            try:
                from utils import get_all_data, print_matrix
                data_dict = get_all_data()
                
                print("\nAvailable variables:")
                print("  - data_dict: Dictionary with all data")
                print("  - data_dict['data']: DataFrame (X, Y, Z)")
                print("  - data_dict['gt_adjacency']: Ground truth adjacency")
                print("  - data_dict['gt_weights']: Ground truth weights")
                print("  - data_dict['fci_skeleton']: FCI skeleton")
                print("  - data_dict['mask']: Mask matrix")
                print("\nExample usage:")
                print("  >>> data_dict['data'].head()")
                print("  >>> print_matrix(data_dict['gt_adjacency'], 'Ground Truth')")
                
                # Start interactive shell
                import code
                code.interact(local=dict(globals(), **locals()))
                
            except Exception as e:
                print(f"\n[ERROR] Failed to load data: {e}")
        
        elif choice == '6':
            print("\nGoodbye!")
            break
        
        else:
            print("\n[ERROR] Invalid choice. Please enter 1-6.")


def main():
    print("=" * 70)
    print("SYMBOLIC CAUSAL REFINEMENT - Quick Start")
    print("=" * 70)
    print("\nObjective: Improve LLM causal direction accuracy using")
    print("           Neural LP-inspired differentiable refinement")
    print("\nCurrent Status: Step 1 & 2 Implementation")
    
    # Check dependencies
    print("\nChecking dependencies...")
    if not check_dependencies():
        print("\nPlease install missing packages before continuing.")
        return
    
    print("[OK] All dependencies installed")
    
    # Check if data already exists
    data_exists = os.path.exists('data/ground_truth_data.csv')
    
    if data_exists:
        print("\n[INFO] Data already generated")
        print("       Run option 2 to visualize or option 3 for summary")
    else:
        print("\n[INFO] No data found. Run option 1 to generate data.")
    
    # Show interactive menu
    interactive_menu()


if __name__ == "__main__":
    main()

