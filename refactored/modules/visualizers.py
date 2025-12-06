"""
Visualizer Module

Handles graph visualization and saving to files.
"""

import networkx as nx
import matplotlib.pyplot as plt
import os


class GraphVisualizer:
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def visualize(self, graph, title="Causal Graph", 
                  filename=None, save_only=False,
                  node_color='lightblue', edge_color='green'):
        # Filter out rejected edges
        valid_edges = [(u, v) for u, v, d in graph.edges(data=True) 
                       if d.get('type') != 'rejected']
        
        # Layout
        pos = nx.circular_layout(graph)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_size=2000, node_color=node_color)
        
        # Draw labels
        nx.draw_networkx_labels(graph, pos, font_size=10)
        
        # Draw edges
        nx.draw_networkx_edges(graph, pos, edgelist=valid_edges, 
                              edge_color=edge_color, width=2, arrowsize=50)
        
        plt.title(title, fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        # Save figure
        if filename:
            fig_path = os.path.join(self.output_dir, f"{filename}.png")
        else:
            fig_path = os.path.join(self.output_dir, "causal_graph.png")
        
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"[OUTPUT] Graph saved to: {fig_path}")

        if not save_only:
            plt.show()
        else:
            plt.close()
        
        return fig_path

