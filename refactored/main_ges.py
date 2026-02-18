"""
Main Program: GES
"""

import os
import time
from datetime import datetime

# Import config and utils
from config import get_output_dir, GROUND_TRUTH_PATH
from utils import get_active_data_loader, print_dataset_info

# Import modules
from modules.algorithms import GESAlgorithm
from modules.visualizers import GraphVisualizer
from modules.reporters import ReportGenerator


class GESPipeline:
    def __init__(self, data_loader, output_dir=None):
        print("=" * 60)
        print("Initializing GES Pipeline")
        print("=" * 60)

        self.output_dir = output_dir or get_output_dir()
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[OUTPUT] Results will be saved to: {self.output_dir}/")

        print("\n[1/4] Loading data...")
        self.data_loader = data_loader
        self.df, self.nodes = self.data_loader.load_csv()

        print("\n[2/4] Setting up GES algorithm...")
        self.algorithm = GESAlgorithm(self.df, self.nodes)

        print("\n[3/4] Setting up visualizer...")
        self.visualizer = GraphVisualizer(self.output_dir)

        print("\n[4/4] Setting up reporter...")
        self.reporter = ReportGenerator(self.output_dir)

        self.model_name = "ges"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.algorithm_runtime_seconds = None

        print("\n" + "=" * 60)
        print("Pipeline initialized successfully!")
        print("=" * 60)

    def run(self, score_func="local_score_BIC"):
        print(f"\n{'='*60}")
        print("Running GES Algorithm")
        print(f"Score function: {score_func}")
        print(f"{'='*60}\n")

        algo_start = time.perf_counter()
        self.graph = self.algorithm.run(score_func=score_func)
        self.algorithm_runtime_seconds = time.perf_counter() - algo_start

        print(f"\n{'='*60}")
        print("GES Algorithm Completed")
        print(f"{'='*60}")
        print(f"[TIME] GES algorithm runtime: {self.algorithm_runtime_seconds:.2f}s")
        self._print_statistics()
        self._save_results()

    def _print_statistics(self):
        print(f"\n{'='*60}")
        print("GRAPH STATISTICS")
        print(f"{'='*60}")
        print(f"Total nodes:          {self.graph.number_of_nodes()}")
        print(f"Total edges:          {self.graph.number_of_edges()}")

        directed = sum(1 for _, _, d in self.graph.edges(data=True) if d.get("type") != "undirected")
        undirected = sum(1 for _, _, d in self.graph.edges(data=True) if d.get("type") == "undirected")

        print("\nEdge Type Breakdown:")
        print(f"  Directed (->):      {directed:3d}")
        print(f"  Undirected (-):     {undirected:3d}  [stored as both directions]")
        print(f"{'='*60}")

    def _save_results(self):
        print(f"\n{'='*60}")
        print("Saving Results")
        print(f"{'='*60}")

        self.reporter.save_text_report(self.graph, model_name="GES")
        self.reporter.save_edge_list(self.graph, model_name="GES")

        filename = f"causal_graph_{self.model_name}_{self.timestamp}"
        self.visualizer.visualize(
            self.graph,
            title="Causal Graph (GES Algorithm)",
            filename=filename,
            save_only=False,
            node_color="lightgreen",
            edge_color="blue",
        )

        print(f"{'='*60}")


def main():
    """Main function - runs GES with config-compatible defaults."""
    total_start = time.perf_counter()
    evaluation_runtime_seconds = None

    print_dataset_info()

    # Allow optional dataset-specific override from root config.
    # If not provided, auto-select score function by data type.
    from config import get_current_dataset_config

    ds_cfg = get_current_dataset_config()
    data_type = str(ds_cfg.get("data_type", "discrete")).strip().lower()
    default_score_func = "local_score_BDeu" if data_type == "discrete" else "local_score_BIC"
    score_func = ds_cfg.get("ges_score_func", default_score_func)

    print("\nUsing parameters:")
    print(f"  Data type: {data_type}")
    print(f"  Score function: {score_func}")
    if data_type == "discrete" and str(score_func).strip().lower() in {"local_score_bic", "bic"}:
        print("[WARN] Discrete data with BIC may significantly hurt SHD.")
        print("[WARN] Consider setting ges_score_func='local_score_BDeu' in config.py.")

    data_loader = get_active_data_loader()
    pipeline = GESPipeline(data_loader)

    print("\nStarting GES algorithm...")
    pipeline.run(score_func=score_func)

    print("\n" + "=" * 60)
    print(f"GES completed! Results saved to {get_output_dir()}/")
    print("=" * 60)

    # === AUTO-EVALUATION ===
    print("\n" + "=" * 60)
    print("Running automatic evaluation...")
    print("=" * 60)

    try:
        eval_start = time.perf_counter()
        from pathlib import Path
        from evaluate_ges import evaluate_ges, find_latest_ges_csv

        latest_ges = find_latest_ges_csv(get_output_dir())
        gt_path = Path(GROUND_TRUTH_PATH)

        if latest_ges and gt_path.exists():
            print(f"\n[INFO] Evaluating: {latest_ges.name}")
            print(f"[INFO] Ground truth: {gt_path.name}\n")

            metrics = evaluate_ges(latest_ges, gt_path, output_dir=get_output_dir())

            print("\n" + "=" * 60)
            print("KEY METRICS (GES Only)")
            print("=" * 60)
            print(f"SHD:                  {metrics['shd']}")
            print(f"Unresolved Ratio:     {metrics['unresolved_ratio']*100:.1f}%")
            print(f"Edge F1:              {metrics['edge_f1']*100:.1f}%")
            print(f"Orientation Accuracy: {metrics['orientation_accuracy']*100:.1f}%")
            print("=" * 60)

        elif not latest_ges:
            print("[WARN] Could not find GES output for evaluation")
        elif not gt_path.exists():
            print(f"[WARN] Ground truth file not found: {gt_path}")
            print("Update GROUND_TRUTH_PATH in config.py to enable evaluation")
        evaluation_runtime_seconds = time.perf_counter() - eval_start
    except Exception as e:
        import traceback

        print(f"[ERROR] Evaluation failed: {e}")
        traceback.print_exc()
        print("\nYou can run 'python evaluate_ges.py' manually later.")
        evaluation_runtime_seconds = time.perf_counter() - eval_start

    print("\n" + "=" * 60)
    print("All done!")
    total_runtime_seconds = time.perf_counter() - total_start
    print(f"[TIME] Algorithm runtime:  {pipeline.algorithm_runtime_seconds:.2f}s")
    if evaluation_runtime_seconds is not None:
        print(f"[TIME] Evaluation runtime: {evaluation_runtime_seconds:.2f}s")
    print(f"[TIME] Total runtime:      {total_runtime_seconds:.2f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
