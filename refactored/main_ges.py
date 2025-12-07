"""
Main Program: GES
"""

import sys
import os
from datetime import datetime

# Import config and utils
from config import get_output_dir
from utils import get_active_data_loader, print_dataset_info

# Import modules from the modules package
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
    print_dataset_info()
    
    data_loader = get_active_data_loader()
    pipeline = GESPipeline(data_loader)

    print("\nStarting GES algorithm...")
    pipeline.run(score_func='local_score_BIC')
    
    print("\n" + "=" * 60)
    print(f"All done! Check {get_output_dir()}/ for results.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

