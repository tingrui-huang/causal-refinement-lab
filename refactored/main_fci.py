"""
Main Program: FCI
"""

import sys
import os
from datetime import datetime

# Import modules from the modules package
from modules.data_loader import LUCASDataLoader
from modules.algorithms import FCIAlgorithm
from modules.visualizers import GraphVisualizer
from modules.reporters import ReportGenerator


class FCIPipeline:
    def __init__(self, data_path, output_dir="outputs"):
        print("=" * 60)
        print("Initializing FCI Pipeline")
        print("=" * 60)

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[OUTPUT] Results will be saved to: {self.output_dir}/")

        print("\n[1/4] Loading data...")
        self.data_loader = LUCASDataLoader(data_path)
        self.df, self.nodes = self.data_loader.load_csv()

        print("\n[2/4] Setting up FCI algorithm...")
        self.algorithm = FCIAlgorithm(self.df, self.nodes)

        print("\n[3/4] Setting up visualizer...")
        self.visualizer = GraphVisualizer(output_dir)

        print("\n[4/4] Setting up reporter...")
        self.reporter = ReportGenerator(output_dir)

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
        print("GRAPH STATISTICS (PAG)")
        print(f"{'='*60}")
        print(f"Total nodes:          {self.graph.number_of_nodes()}")
        print(f"Total edges:          {self.graph.number_of_edges()}")
        
        # Count edge types
        directed = sum(1 for u, v, d in self.graph.edges(data=True) 
                      if d.get('type') == 'directed')
        partial = sum(1 for u, v, d in self.graph.edges(data=True) 
                     if d.get('type') == 'partial')
        bidirected = sum(1 for u, v, d in self.graph.edges(data=True) 
                        if d.get('type') == 'bidirected')
        
        print(f"Directed edges:       {directed}")
        print(f"Partial edges:        {partial}")
        print(f"Bidirected edges:     {bidirected}")
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
    test_input = input("\nChoose test (default: chisq): ").strip().lower()
    independence_test = test_input if test_input in ['chisq', 'gsq', 'fisherz'] else 'chisq'
    
    # Get alpha
    alpha_input = input("\nSignificance level (default: 0.05): ").strip()
    alpha = float(alpha_input) if alpha_input else 0.05
    
    # Initialize pipeline
    data_path = "../lucas0_train.csv"
    pipeline = FCIPipeline(data_path, output_dir="../outputs")
    
    # Run pipeline
    print(f"\nStarting FCI algorithm with {independence_test} test...")
    pipeline.run(independence_test=independence_test, alpha=alpha)
    
    print("\n" + "=" * 60)
    print("All done! Check the outputs/ directory for results.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

