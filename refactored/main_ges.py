"""
Main Program: GES
"""

import sys
import os
from datetime import datetime

# Import modules from the modules package
from modules.data_loader import LUCASDataLoader
from modules.algorithms import GESAlgorithm
from modules.visualizers import GraphVisualizer
from modules.reporters import ReportGenerator


class GESPipeline:
    def __init__(self, data_path, output_dir="outputs"):
        print("=" * 60)
        print("Initializing GES Pipeline")
        print("=" * 60)

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[OUTPUT] Results will be saved to: {self.output_dir}/")

        print("\n[1/4] Loading data...")
        self.data_loader = LUCASDataLoader(data_path)
        self.df, self.nodes = self.data_loader.load_csv()

        print("\n[2/4] Setting up GES algorithm...")
        self.algorithm = GESAlgorithm(self.df, self.nodes)

        print("\n[3/4] Setting up visualizer...")
        self.visualizer = GraphVisualizer(output_dir)

        print("\n[4/4] Setting up reporter...")
        self.reporter = ReportGenerator(output_dir)

        self.model_name = "ges"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "=" * 60)
        print("Pipeline initialized successfully!")
        print("=" * 60)
    
    def run(self, score_func='local_score_BIC'):
        print(f"\n{'='*60}")
        print(f"Running GES Algorithm")
        print(f"Score function: {score_func}")
        print(f"{'='*60}\n")

        self.graph = self.algorithm.run(score_func=score_func)
        
        print(f"\n{'='*60}")
        print("GES Algorithm Completed")
        print(f"{'='*60}")

        self._print_statistics()

        self._save_results()
    
    def _print_statistics(self):
        print(f"\n{'='*60}")
        print("GRAPH STATISTICS")
        print(f"{'='*60}")
        print(f"Total nodes:          {self.graph.number_of_nodes()}")
        print(f"Total edges:          {self.graph.number_of_edges()}")
        
        # Count edge types
        directed = sum(1 for u, v, d in self.graph.edges(data=True) 
                      if d.get('type') != 'undirected')
        undirected = sum(1 for u, v, d in self.graph.edges(data=True) 
                        if d.get('type') == 'undirected')
        
        print(f"Directed edges:       {directed}")
        print(f"Undirected edges:     {undirected}")
        print(f"{'='*60}")
    
    def _save_results(self):
        print(f"\n{'='*60}")
        print("Saving Results")
        print(f"{'='*60}")

        self.reporter.save_text_report(self.graph, model_name="GES")

        self.reporter.save_edge_list(self.graph, model_name="GES")

        filename = f"causal_graph_{self.model_name}_{self.timestamp}"
        self.visualizer.visualize(self.graph, 
                                 title="Causal Graph (GES Algorithm)",
                                 filename=filename,
                                 save_only=False,
                                 node_color='lightgreen',
                                 edge_color='blue')
        
        print(f"{'='*60}")


def main():
    data_path = "../lucas0_train.csv"
    pipeline = GESPipeline(data_path, output_dir="../outputs")

    print("\nStarting GES algorithm...")
    pipeline.run(score_func='local_score_BIC')
    
    print("\n" + "=" * 60)
    print("All done! Check the outputs/ directory for results.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

