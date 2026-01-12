"""
Main Program: PC (Peter-Clark)

Pure baseline run of the PC algorithm to inspect PC outputs (CPDAG).
"""

import os
import argparse
from datetime import datetime

# Import config and utils
from config import get_output_dir, FCI_INDEPENDENCE_TEST, FCI_ALPHA, GROUND_TRUTH_PATH
from utils import get_active_data_loader, print_dataset_info

# Import modules
from modules.algorithms import PCAlgorithm
from modules.visualizers import GraphVisualizer
from modules.reporters import ReportGenerator


class PCPipeline:
    def __init__(self, data_loader, output_dir=None):
        print("=" * 60)
        print("Initializing PC Pipeline")
        print("=" * 60)

        self.output_dir = output_dir or get_output_dir()
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[OUTPUT] Results will be saved to: {self.output_dir}/")

        print("\n[1/4] Loading data...")
        self.data_loader = data_loader
        self.df, self.nodes = self.data_loader.load_csv()

        print("\n[2/4] Setting up PC algorithm...")
        self.algorithm = PCAlgorithm(self.df, self.nodes)

        print("\n[3/4] Setting up visualizer...")
        self.visualizer = GraphVisualizer(self.output_dir)

        print("\n[4/4] Setting up reporter...")
        self.reporter = ReportGenerator(self.output_dir)

        self.model_name = "pc"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print("\n" + "=" * 60)
        print("Pipeline initialized successfully!")
        print("=" * 60)

    def run(self, independence_test='chisq', alpha=0.05, stable=True):
        print(f"\n{'='*60}")
        print("Running PC Algorithm")
        print(f"Independence test: {independence_test}")
        print(f"Significance level: {alpha}")
        print(f"Stable-PC: {stable}")
        print(f"{'='*60}\n")

        self.graph = self.algorithm.run(
            independence_test=independence_test,
            alpha=alpha,
            stable=stable,
        )

        print(f"\n{'='*60}")
        print("PC Algorithm Completed")
        print(f"{'='*60}")
        self._print_statistics()
        self._save_results()

    def _print_statistics(self):
        print(f"\n{'='*60}")
        print("GRAPH STATISTICS (CPDAG)")
        print(f"{'='*60}")
        print(f"Total nodes:          {self.graph.number_of_nodes()}")
        print(f"Total edges:          {self.graph.number_of_edges()}")

        directed = sum(1 for _, _, d in self.graph.edges(data=True)
                       if d.get('type') == 'directed')
        undirected = sum(1 for _, _, d in self.graph.edges(data=True)
                         if d.get('type') == 'undirected')

        print("\nEdge Type Breakdown:")
        print(f"  Directed (->):      {directed:3d}")
        print(f"  Undirected (-):     {undirected:3d}  [stored as both directions]")
        print(f"{'='*60}")

    def _save_results(self):
        print(f"\n{'='*60}")
        print("Saving Results")
        print(f"{'='*60}")

        self.reporter.save_text_report(self.graph, model_name="PC")
        self.reporter.save_edge_list(self.graph, model_name="PC")

        filename = f"causal_graph_{self.model_name}_{self.timestamp}"
        self.visualizer.visualize(
            self.graph,
            title="Causal Graph (PC Algorithm - CPDAG)",
            filename=filename,
            save_only=False,
            node_color='lightcyan',
            edge_color='black',
        )

        print(f"{'='*60}")


def main():
    """
    Main function.
    Defaults to using FCI settings from unified config for convenience.
    """
    parser = argparse.ArgumentParser(description="Run PC baseline (pure PC)")
    parser.add_argument("--alpha", type=float, default=FCI_ALPHA,
                        help="Significance level alpha (default from config.py)")
    parser.add_argument("--independence-test", type=str, default=FCI_INDEPENDENCE_TEST,
                        choices=["chisq", "gsq", "fisherz"],
                        help="Independence test (default from config.py)")
    parser.add_argument("--stable", action="store_true", default=True,
                        help="Use stable-PC (default: True)")
    parser.add_argument("--no-stable", action="store_false", dest="stable",
                        help="Disable stable-PC")
    args = parser.parse_args()

    print_dataset_info()

    print("\nUsing parameters:")
    print(f"  Independence test: {args.independence_test}")
    print(f"  Significance level: {args.alpha}")
    print(f"  Stable-PC: {args.stable}")

    data_loader = get_active_data_loader()
    pipeline = PCPipeline(data_loader)
    pipeline.run(independence_test=args.independence_test, alpha=args.alpha, stable=args.stable)

    print("\n" + "=" * 60)
    print(f"PC completed! Results saved to {get_output_dir()}/")
    print("=" * 60 + "\n")

    # === AUTO-EVALUATION ===
    print("\n" + "=" * 60)
    print("Running automatic evaluation...")
    print("=" * 60)

    try:
        from evaluate_pc import evaluate_pc, find_latest_pc_csv
        from pathlib import Path

        latest_pc = find_latest_pc_csv(get_output_dir())
        gt_path = Path(GROUND_TRUTH_PATH)

        if latest_pc and gt_path.exists():
            print(f"\n[INFO] Evaluating: {latest_pc.name}")
            print(f"[INFO] Ground truth: {gt_path.name}\n")

            metrics = evaluate_pc(latest_pc, gt_path, output_dir=get_output_dir())

            print("\n" + "=" * 60)
            print("KEY METRICS (PC Only)")
            print("=" * 60)
            print(f"SHD:                  {metrics['shd']}")
            print(f"Unresolved Ratio:     {metrics['unresolved_ratio']*100:.1f}%")
            print(f"Edge F1:              {metrics['edge_f1']*100:.1f}%")
            print(f"Orientation Accuracy: {metrics['orientation_accuracy']*100:.1f}%")
            print("=" * 60)

        elif not latest_pc:
            print("[WARN] Could not find PC output for evaluation")
        elif not gt_path.exists():
            print(f"[WARN] Ground truth file not found: {gt_path}")
            print("Update GROUND_TRUTH_PATH in config.py to enable evaluation")
    except Exception as e:
        import traceback
        print(f"[ERROR] Evaluation failed: {e}")
        traceback.print_exc()
        print("\nYou can run 'python evaluate_pc.py' manually later.")


if __name__ == "__main__":
    main()

