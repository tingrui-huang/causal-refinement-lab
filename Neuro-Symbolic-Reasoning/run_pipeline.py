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
        # Just let it print directly without capturing encoding errors
        sys.stdout.buffer.write(result.stdout.encode('utf-8', errors='replace'))
        sys.stdout.buffer.flush()
    
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        if result.stderr:
            print("Error output:")
            print(result.stderr)
        return False
    
    return True


def get_reproducibility_env() -> dict:
    """
    Best-effort reproducibility for subprocesses.

    Note: PYTHONHASHSEED must be set before interpreter start to fully take effect,
    so passing it via env to subprocesses is useful.
    """
    try:
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        import config as unified_config

        seed = getattr(unified_config, "RANDOM_SEED", None)
        if seed is None:
            return {}

        return {"PYTHONHASHSEED": str(seed)}
    except Exception:
        return {}


def run_fci():
    """Step 1: Run constraint-based discovery (FCI or RFCI)"""
    refactored_dir = Path('../refactored')
    
    if not refactored_dir.exists():
        print(f"[ERROR] refactored/ directory not found at {refactored_dir.absolute()}")
        return False
    
    # Decide which algorithm to run based on dataset config
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from config import DATASET, get_current_dataset_config

    cfg = get_current_dataset_config()
    algo = str(cfg.get("constraint_algo", "fci")).lower()
    if algo == "rfci":
        script = "main_rfci.py"
        desc = "STEP 1: RUNNING RFCI (TETRAD) ALGORITHM"
    else:
        script = "main_fci.py"
        desc = "STEP 1: RUNNING FCI ALGORITHM"

    success = run_command(
        f"{sys.executable} {script}",
        cwd=refactored_dir,
        description=desc,
        env=get_reproducibility_env(),
    )
    
    if success:
        print(f"\n[OK] {algo.upper()} completed successfully")
    
    return success


def run_llm():
    """Step 2: Run LLM direction resolution (if configured)"""
    # Check if LLM is configured
    import sys
    from pathlib import Path
    
    # Add project root to path to import config
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from config import LLM_MODEL, DATASET, get_current_dataset_config
    
    if not LLM_MODEL:
        print("\n[INFO] No LLM configured. Skipping LLM step.")
        return True

    # RFCI is supported for skeleton generation; the current LLM scripts are FCI-based.
    # We skip by default to avoid accidentally running slow FCI again.
    algo = str(get_current_dataset_config().get("constraint_algo", "fci")).lower()
    if algo == "rfci":
        print("\n[WARN] constraint_algo=rfci for this dataset.")
        print("[WARN] Current refactored LLM scripts are FCI-based; skipping LLM step to avoid running FCI.")
        print("[WARN] If you need RFCI+LLM, we can add a dedicated hybrid script that consumes edges_RFCI_*.csv.")
        return True
    
    refactored_dir = Path('../refactored')
    
    # Map LLM model to script
    if 'gpt' in LLM_MODEL.lower():
        script_name = 'main_hybrid_fci_llm.py'
        llm_name = 'GPT-3.5'
    elif 'zephyr' in LLM_MODEL.lower():
        script_name = 'main_hybrid_fci_zephyr.py'
        llm_name = 'Zephyr'
    else:
        print(f"[WARN] Unknown LLM model: {LLM_MODEL}. Skipping LLM step.")
        return True
    
    print(f"\n[INFO] LLM configured: {llm_name}")
    print(f"[INFO] Checking for existing LLM results...")
    
    # Check if LLM results already exist
    from config import DATASET, FCI_OUTPUT_DIR
    output_dir = FCI_OUTPUT_DIR / DATASET
    
    if 'gpt' in LLM_MODEL.lower():
        pattern = 'edges_FCI_LLM_GPT35_*.csv'
    else:
        pattern = 'edges_FCI_LLM_Zephyr_*.csv'
    
    existing_files = list(output_dir.glob(pattern)) if output_dir.exists() else []
    
    if existing_files:
        latest_file = max(existing_files, key=lambda p: p.stat().st_mtime)
        print(f"[INFO] Found existing LLM results: {latest_file.name}")
        print(f"[INFO] Skipping LLM call. Delete this file to regenerate.")
        
        # Still evaluate the existing LLM output
        evaluate_llm_output()
        
        return True
    
    print(f"[INFO] No existing results found. Calling {llm_name}...")
    
    success = run_command(
        f"{sys.executable} {script_name}",
        cwd=refactored_dir,
        description=f"STEP 2: RUNNING FCI + {llm_name.upper()}",
        env=get_reproducibility_env(),
    )
    
    if success:
        print(f"\n[OK] LLM ({llm_name}) completed successfully")
        
        # Evaluate LLM output
        evaluate_llm_output()
    
    return success


def evaluate_llm_output():
    """Evaluate LLM output against ground truth"""
    try:
        # Import config
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config import DATASET, FCI_OUTPUT_DIR, get_current_dataset_config
        
        dataset_config = get_current_dataset_config()
        
        # Find latest LLM output
        output_dir = FCI_OUTPUT_DIR / DATASET
        if not output_dir.exists():
            print("\n[INFO] No LLM output directory found, skipping LLM evaluation")
            return True
        
        llm_files = list(output_dir.glob('edges_FCI_LLM_*.csv'))
        if not llm_files:
            print("\n[INFO] No LLM output files found, skipping LLM evaluation")
            return True
        
        # Get the most recent LLM output
        latest_llm = max(llm_files, key=lambda p: p.stat().st_mtime)
        
        # Get ground truth path
        ground_truth_path = dataset_config.get('ground_truth_path')
        ground_truth_type = dataset_config.get('ground_truth_type', 'bif')
        
        if not ground_truth_path or not Path(ground_truth_path).exists():
            print("\n[INFO] Ground truth not found, skipping LLM evaluation")
            return True
        
        print("\n" + "=" * 80)
        print("STEP 2.5: EVALUATING LLM OUTPUT AGAINST GROUND TRUTH")
        print("=" * 80)
        print(f"[INFO] Evaluating: {latest_llm.name}")
        
        # Import and use the standalone evaluator
        sys.path.insert(0, str(Path(__file__).parent / 'modules'))
        from evaluator import evaluate_llm_output as eval_llm
        
        # Evaluate LLM output
        metrics = eval_llm(
            llm_csv_path=str(latest_llm),
            ground_truth_path=str(ground_truth_path),
            ground_truth_type=ground_truth_type,
            output_dir=str(output_dir)
        )
        
        print(f"\n[OK] LLM evaluation completed")
        print(f"  Edge F1:         {metrics['edge_f1']:.1%}")
        print(f"  Directed F1:     {metrics['directed_f1']:.1%}")
        print(f"  Orientation Acc: {metrics['orientation_accuracy']:.1%}")
        print(f"  Full SHD:        {metrics['full_shd']}")
        
        return True
    
    except Exception as e:
        print(f"\n[ERROR] LLM evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_fci_evaluation():
    """Step 2: Evaluate FCI results (optional, already done in main_fci.py)"""
    refactored_dir = Path('../refactored')
    
    success = run_command(
        f"{sys.executable} evaluate_fci.py",
        cwd=refactored_dir,
        description="STEP 2: EVALUATING FCI RESULTS"
    )
    
    if success:
        print("\n[OK] FCI evaluation completed")
    
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
        print("\n[OK] Export completed")
    
    return success


def run_training():
    """Step 4: Run neuro-symbolic training"""
    success = run_command(
        f"{sys.executable} train.py",
        cwd=None,  # Already in Neuro-Symbolic-Reasoning/
        description="STEP 4: RUNNING NEURO-SYMBOLIC TRAINING"
    )
    
    if success:
        print("\n[OK] Training completed")
    
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
        # FCI + LLM + export only
        print("\nMode: FCI only (skip training)")
        
        if not run_fci():
            sys.exit(1)
        
        # Run LLM if configured
        if not run_llm():
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
        
        # Step 2: LLM (if configured)
        if not run_llm():
            sys.exit(1)
        
        # Step 3: Export
        if not run_export():
            sys.exit(1)
        
        # Step 4: Training
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

