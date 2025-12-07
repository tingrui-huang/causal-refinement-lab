"""
Main Program: FCI
"""

import sys
import os
from datetime import datetime

# Import config and utils
from config import get_output_dir
from utils import get_active_data_loader, print_dataset_info

# Import modules from the modules package
from modules.algorithms import FCIAlgorithm
from modules.visualizers import GraphVisualizer
from modules.reporters import ReportGenerator


class FCIPipeline:
    def __init__(self, data_loader, output_dir=None):
        print("=" * 60)
        print("Initializing FCI Pipeline")
        print("=" * 60)

        self.output_dir = output_dir or get_output_dir()
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[OUTPUT] Results will be saved to: {self.output_dir}/")

        print("\n[1/4] Loading data...")
        self.data_loader = data_loader
        self.df, self.nodes = self.data_loader.load_csv()

        print("\n[2/4] Setting up FCI algorithm...")
        self.algorithm = FCIAlgorithm(self.df, self.nodes)

        print("\n[3/4] Setting up visualizer...")
        self.visualizer = GraphVisualizer(self.output_dir)

        print("\n[4/4] Setting up reporter...")
        self.reporter = ReportGenerator(self.output_dir)

        self.model_name = "fci"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "=" * 60)
        print("Pipeline initialized successfully!")
        print("=" * 60)
    
    def run(self, independence_test='chisq', alpha=0.05):
        print(f"\n{'='*60}")
        print(f"Running FCI Algorithm")
        print(f"Independence test: {independence_test}")
        print(f"Significance level: {alpha}")
        print(f"{'='*60}\n")

        self.graph = self.algorithm.run(
            independence_test=independence_test,
            alpha=alpha
        )
        
        print(f"\n{'='*60}")
        print("FCI Algorithm Completed")
        print(f"{'='*60}")
        self._print_statistics()
        self._save_results()
    
    def _print_statistics(self):
        print(f"\n{'='*60}")
        print("GRAPH STATISTICS (PAG - Partial Ancestral Graph)")
        print(f"{'='*60}")
        print(f"Total nodes:          {self.graph.number_of_nodes()}")
        print(f"Total edges:          {self.graph.number_of_edges()}")
        
        # Count all edge types
        directed = sum(1 for u, v, d in self.graph.edges(data=True) 
                      if d.get('type') == 'directed')
        bidirected = sum(1 for u, v, d in self.graph.edges(data=True) 
                        if d.get('type') == 'bidirected')
        partial = sum(1 for u, v, d in self.graph.edges(data=True) 
                     if d.get('type') == 'partial')
        undirected = sum(1 for u, v, d in self.graph.edges(data=True) 
                        if d.get('type') == 'undirected')
        tail_tail = sum(1 for u, v, d in self.graph.edges(data=True) 
                       if d.get('type') == 'tail-tail')
        
        print(f"\nEdge Type Breakdown:")
        print(f"  Directed (->):      {directed:3d}  [certain causal direction]")
        print(f"  Bidirected (<->):   {bidirected:3d}  [latent confounder]")
        print(f"  Partial (o->/-o):   {partial:3d}  [ambiguous direction]")
        print(f"  Undirected (o-o):   {undirected:3d}  [completely ambiguous]")
        print(f"  Tail-tail (--):     {tail_tail:3d}  [no clear direction]")
        print(f"{'='*60}")
    
    def _save_results(self):
        print(f"\n{'='*60}")
        print("Saving Results")
        print(f"{'='*60}")

        self.reporter.save_text_report(self.graph, model_name="FCI")

        self.reporter.save_edge_list(self.graph, model_name="FCI")
        filename = f"causal_graph_{self.model_name}_{self.timestamp}"
        self.visualizer.visualize(self.graph, 
                                 title="Causal Graph (FCI Algorithm - PAG)",
                                 filename=filename,
                                 save_only=False,
                                 node_color='lightyellow',
                                 edge_color='red')
        
        print(f"{'='*60}")


def main():
    print_dataset_info()
    
    test_input = input("\nChoose test (default: chisq): ").strip().lower()
    independence_test = test_input if test_input in ['chisq', 'gsq', 'fisherz'] else 'chisq'
    
    # Get alpha
    alpha_input = input("\nSignificance level (default: 0.05): ").strip()
    alpha = float(alpha_input) if alpha_input else 0.05
    
    # Initialize pipeline
    data_loader = get_active_data_loader()
    pipeline = FCIPipeline(data_loader)
    
    # Run pipeline
    print(f"\nStarting FCI algorithm with {independence_test} test...")
    pipeline.run(independence_test=independence_test, alpha=alpha)
    
    print("\n" + "=" * 60)
    print(f"All done! Check {get_output_dir()}/ for results.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

