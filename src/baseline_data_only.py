import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class BaselineGraph:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.nodes = self.df.columns.tolist()
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.nodes)

        print(f"Loaded data: {self.df.shape}")
        print(f"Variables: {self.nodes}")

    def build_correlation_graph(self, threshold=0.3):
        corr_matrix = self.df.corr().abs()

        edge_count = 0
        for i, node_a in enumerate(self.nodes):
            for j, node_b in enumerate(self.nodes):
                if i >= j: continue

                score = corr_matrix.loc[node_a, node_b]
                if score > threshold:
                    self.graph.add_edge(node_a, node_b, weight=score)
                    self.graph.add_edge(node_b, node_a, weight=score)
                    edge_count += 1

        print(f"Added {edge_count} bidirectional edges based on correlation > {threshold}")

    def build_simple_dag(self, threshold=0.3):
        """
        Build DAG：letter order
        """
        corr_matrix = self.df.corr().abs()

        edge_count = 0
        for i, node_a in enumerate(self.nodes):
            for j, node_b in enumerate(self.nodes):
                if i >= j: continue

                score = corr_matrix.loc[node_a, node_b]
                if score > threshold:
                    # letter order
                    if node_a < node_b:
                        self.graph.add_edge(node_a, node_b, weight=score)
                    else:
                        self.graph.add_edge(node_b, node_a, weight=score)
                    edge_count += 1

        print(f"Added {edge_count} directed edges based on correlation > {threshold}")

    def visualize(self, title="Baseline Correlation Graph"):
        pos = nx.circular_layout(self.graph)

        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(self.graph, pos, node_size=2000, node_color='lightcoral')
        nx.draw_networkx_labels(self.graph, pos)

        edges = self.graph.edges()
        nx.draw_networkx_edges(self.graph, pos, edgelist=edges, edge_color='gray', width=1, arrowsize=50, alpha=0.6)
        
        plt.title(title)
        plt.show()


# --- Execution ---
if __name__ == "__main__":
    baseline = BaselineGraph("../lucas0_train.csv")

    baseline.build_simple_dag(threshold=0.3)
    baseline.visualize(title="Baseline DAG (Data Only, Correlation > 0.3)")

