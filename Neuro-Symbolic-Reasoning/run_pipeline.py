"""
Complete Pipeline Runner

Runs the full causal discovery + neuro-symbolic training pipeline:
1. FCI Algorithm (refactored/)
2. FCI Evaluation (refactored/)
3. Export to Neuro-Symbolic (refactored/)
4. Neural Training (Neuro-Symbolic-Reasoning/)

Usage:
    python run_pipeline.py                    # Full pipeline (FCI + Training)
    python run_pipeline.py --fci-only         # Only run FCI
    python run_pipeline.py --train-only       # Only run training (use existing FCI)
    python run_pipeline.py --skip-evaluation  # Skip FCI evaluation
"""

import subprocess
import sys
import argparse
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, will continue without it
    pass


def run_command(command, cwd=None, description="", env=None):
    """Run a command and handle errors"""
    if description:
        print("\n" + "=" * 80)
        print(description)
        print("=" * 80)
    
    # Use current environment and add custom env vars if provided
    cmd_env = os.environ.copy()
    if env:
        cmd_env.update(env)
    
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        shell=True,
        env=cmd_env,
        encoding='utf-8',
        errors='replace'  # Replace undecodable characters
    )
    
    # Print output (handle encoding issues)
    if result.stdout:
        try:
            print(result.stdout)
        except UnicodeEncodeError:
            # Fallback: print with error handling
            print(result.stdout.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
    
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        if result.stderr:
            print("Error output:")
            print(result.stderr)
        return False
    
    return True


def run_fci():
    """Step 1: Run FCI algorithm"""
    refactored_dir = Path('../refactored')
    
    if not refactored_dir.exists():
        print(f"[ERROR] refactored/ directory not found at {refactored_dir.absolute()}")
        return False
    
    success = run_command(
        f"{sys.executable} main_fci.py",
        cwd=refactored_dir,
        description="STEP 1: RUNNING FCI ALGORITHM"
    )
    
    if success:
        print("\n✓ FCI completed successfully")
    
    return success


def run_fci_evaluation():
    """Step 2: Evaluate FCI results (optional, already done in main_fci.py)"""
    refactored_dir = Path('../refactored')
    
    success = run_command(
        f"{sys.executable} evaluate_fci.py",
        cwd=refactored_dir,
        description="STEP 2: EVALUATING FCI RESULTS"
    )
    
    if success:
        print("\n✓ FCI evaluation completed")
    
    return success


def run_export():
    """Step 3: Export FCI results to Neuro-Symbolic"""
    refactored_dir = Path('../refactored')
    
    success = run_command(
        f"{sys.executable} export_to_neuro.py",
        cwd=refactored_dir,
        description="STEP 3: EXPORTING TO NEURO-SYMBOLIC-REASONING"
    )
    
    if success:
        print("\n✓ Export completed")
    
    return success


def run_training():
    """Step 4: Run neuro-symbolic training"""
    success = run_command(
        f"{sys.executable} train.py",
        cwd=None,  # Already in Neuro-Symbolic-Reasoning/
        description="STEP 4: RUNNING NEURO-SYMBOLIC TRAINING"
    )
    
    if success:
        print("\n✓ Training completed")
    
    return success


def main():
    parser = argparse.ArgumentParser(description='Run complete causal discovery pipeline')
    parser.add_argument('--fci-only', action='store_true', 
                       help='Only run FCI algorithm (skip training)')
    parser.add_argument('--train-only', action='store_true',
                       help='Only run training (use existing FCI results)')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip FCI evaluation step')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("COMPLETE CAUSAL DISCOVERY PIPELINE")
    print("=" * 80)
    print("Pipeline: FCI → Evaluate → Export → Neural Training")
    print("=" * 80)
    
    # Determine which steps to run
    if args.train_only:
        # Only training
        print("\nMode: Training only (using existing FCI results)")
        if not run_training():
            sys.exit(1)
    
    elif args.fci_only:
        # FCI + export only
        print("\nMode: FCI only (skip training)")
        
        if not run_fci():
            sys.exit(1)
        
        # Note: Evaluation is already done in main_fci.py
        # But we can run it again if needed
        if not args.skip_evaluation:
            print("\n[INFO] Evaluation already done in FCI step")
        
        if not run_export():
            sys.exit(1)
    
    else:
        # Full pipeline
        print("\nMode: Full pipeline")
        
        # Step 1: FCI (includes auto-evaluation)
        if not run_fci():
            sys.exit(1)
        
        # Step 2: Export
        if not run_export():
            sys.exit(1)
        
        # Step 3: Training
        if not run_training():
            sys.exit(1)
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nResults saved to:")
    print("  - FCI outputs: refactored/output/")
    print("  - Training results: Neuro-Symbolic-Reasoning/results/")
    print("=" * 80)


if __name__ == "__main__":
    main()
