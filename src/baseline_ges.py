import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from causallearn.search.ScoreBased.GES import ges
from report_generator import save_text_report, save_edge_list


class GESBaseline:
    def __init__(self, data_path, output_dir="outputs"):
        # Load Data
        self.df = pd.read_csv(data_path)
        self.nodes = self.df.columns.tolist()
        
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[OUTPUT] Results will be saved to: {self.output_dir}/")
        
        # Model metadata
        self.model_name = "baseline_ges"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"Loaded data: {self.df.shape}")
        print(f"Variables: {self.nodes}")
        
        # Convert to numpy array for causal-learn
        self.data_matrix = self.df.values

    def run_ges(self):
        """
        Run GES (Greedy Equivalence Search) algorithm
        """
        print("\nRunning GES algorithm...")
        print("This may take a while depending on data size...")
        
        # Run GES
        Record = ges(self.data_matrix)
        
        # Get the learned graph
        self.graph_matrix = Record['G'].graph
        
        print(f"GES completed!")
        print(f"Number of edges found: {np.sum(self.graph_matrix != 0) / 2}")
        
        return Record

    def convert_to_networkx(self):
        """
        Convert adjacency matrix to NetworkX graph
        """
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.nodes)
        
        n = len(self.nodes)
        edge_count = 0
        
        for i in range(n):
            for j in range(n):
                if self.graph_matrix[i, j] == -1 and self.graph_matrix[j, i] == 1:
                    # i -> j
                    self.graph.add_edge(self.nodes[i], self.nodes[j])
                    edge_count += 1
                elif self.graph_matrix[i, j] == 1 and self.graph_matrix[j, i] == -1:
                    # j -> i (already handled in reverse)
                    pass
                elif self.graph_matrix[i, j] == -1 and self.graph_matrix[j, i] == -1:
                    # undirected edge (both directions)
                    if i < j:  # avoid duplicate
                        self.graph.add_edge(self.nodes[i], self.nodes[j])
                        self.graph.add_edge(self.nodes[j], self.nodes[i])
                        edge_count += 1
        
        print(f"Converted to NetworkX graph with {edge_count} directed edges")

    def visualize(self, title="GES Causal Discovery Result", save_only=False):
        """
        Visualize the learned causal graph
        """
        pos = nx.circular_layout(self.graph)
        
        plt.figure(figsize=(12, 10))
        nx.draw_networkx_nodes(self.graph, pos, node_size=2000, node_color='lightgreen')
        nx.draw_networkx_labels(self.graph, pos, font_size=10)
        
        edges = self.graph.edges()
        nx.draw_networkx_edges(self.graph, pos, edgelist=edges, 
                              edge_color='blue', width=2, arrowsize=50, 
                              alpha=0.7, arrowstyle='->')
        
        plt.title(title, fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.output_dir, f"causal_graph_{self.model_name}_{self.timestamp}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"[OUTPUT] Graph saved to: {fig_path}")
        
        if not save_only:
            plt.show()
        else:
            plt.close()

    def print_edges(self):
        """
        Print all edges in the learned graph
        """
        print("\n" + "="*50)
        print("Learned Causal Edges:")
        print("="*50)
        
        edges = list(self.graph.edges())
        if not edges:
            print("No edges found!")
        else:
            for i, (source, target) in enumerate(edges, 1):
                print(f"{i}. {source} -> {target}")
        
        print("="*50)


if __name__ == "__main__":
    print("="*60)
    print("GES (Greedy Equivalence Search) Baseline")
    print("Pure data-driven causal discovery without LLM")
    print("="*60)
    
    baseline = GESBaseline("../lucas0_train.csv")
    
    Record = baseline.run_ges()
    
    baseline.convert_to_networkx()
    
    baseline.print_edges()
    
    # Generate reports
    save_text_report(baseline.graph, model_name="GES", output_dir=baseline.output_dir)
    save_edge_list(baseline.graph, model_name="GES", output_dir=baseline.output_dir)
    
    baseline.visualize(title="GES Algorithm - Causal Graph")

