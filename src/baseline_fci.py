import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import chisq, fisherz


class FCIBaseline:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.nodes = self.df.columns.tolist()
        
        print(f"Loaded data: {self.df.shape}")
        print(f"Variables: {self.nodes}")
        
        self.data_matrix = self.df.values

    def run_fci(self, independence_test='chisq', alpha=0.05):
        """
        Run FCI (Fast Causal Inference) algorithm
        
        Parameters:
        - independence_test: 'chisq' for discrete data, 'fisherz' for continuous
        - alpha: significance level for independence tests
        """
        print(f"\nRunning FCI algorithm...")
        print(f"Independence test: {independence_test}")
        print(f"Alpha (significance level): {alpha}")
        print("This may take a while depending on data size...")
        
        if independence_test == 'chisq':
            test_method = chisq
        elif independence_test == 'fisherz':
            test_method = fisherz
        else:
            raise ValueError("independence_test must be 'chisq' or 'fisherz'")
        
        G, edges = fci(
            self.data_matrix, 
            independence_test_method=test_method,
            alpha=alpha,
            labels=self.nodes
        )
        
        self.graph_obj = G
        self.edges = edges
        
        print(f"FCI completed!")
        print(f"Number of edges found: {len(edges)}")
        
        return G, edges

    def convert_to_networkx(self):
        """
        Convert FCI result to NetworkX graph
        FCI returns PAG (Partial Ancestral Graph) with different edge types
        """
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.nodes)
        
        graph_matrix = self.graph_obj.graph
        n = len(self.nodes)
        edge_count = 0
        
        for i in range(n):
            for j in range(n):
                if i >= j:
                    continue
                
                edge_i_j = graph_matrix[i, j]
                edge_j_i = graph_matrix[j, i]
                
                if edge_i_j == -1 and edge_j_i == 1:
                    # i -> j (directed edge)
                    self.graph.add_edge(self.nodes[i], self.nodes[j], edge_type='directed')
                    edge_count += 1
                elif edge_i_j == 1 and edge_j_i == -1:
                    # j -> i (directed edge)
                    self.graph.add_edge(self.nodes[j], self.nodes[i], edge_type='directed')
                    edge_count += 1
                elif edge_i_j == 1 and edge_j_i == 1:
                    # i <-> j (bidirected edge, indicates latent confounder)
                    self.graph.add_edge(self.nodes[i], self.nodes[j], edge_type='bidirected')
                    self.graph.add_edge(self.nodes[j], self.nodes[i], edge_type='bidirected')
                    edge_count += 1
                elif edge_i_j == 2 and edge_j_i == 1:
                    # i o-> j (circle-arrow, partially oriented)
                    self.graph.add_edge(self.nodes[i], self.nodes[j], edge_type='partial')
                    edge_count += 1
                elif edge_i_j == 1 and edge_j_i == 2:
                    # j o-> i (circle-arrow, partially oriented)
                    self.graph.add_edge(self.nodes[j], self.nodes[i], edge_type='partial')
                    edge_count += 1
                elif edge_i_j == 2 and edge_j_i == 2:
                    # i o-o j (undirected/unoriented)
                    self.graph.add_edge(self.nodes[i], self.nodes[j], edge_type='undirected')
                    self.graph.add_edge(self.nodes[j], self.nodes[i], edge_type='undirected')
                    edge_count += 1
        
        print(f"Converted to NetworkX graph with {edge_count} edges")

    def visualize(self, title="FCI Causal Discovery Result"):
        """
        Visualize the learned PAG (Partial Ancestral Graph)
        Different edge types shown in different colors
        """
        pos = nx.circular_layout(self.graph)
        
        plt.figure(figsize=(12, 10))
        nx.draw_networkx_nodes(self.graph, pos, node_size=2000, node_color='lightyellow')
        nx.draw_networkx_labels(self.graph, pos, font_size=9)
        
        directed_edges = [(u, v) for u, v, d in self.graph.edges(data=True) 
                         if d.get('edge_type') == 'directed']
        bidirected_edges = [(u, v) for u, v, d in self.graph.edges(data=True) 
                           if d.get('edge_type') == 'bidirected']
        partial_edges = [(u, v) for u, v, d in self.graph.edges(data=True) 
                        if d.get('edge_type') == 'partial']
        undirected_edges = [(u, v) for u, v, d in self.graph.edges(data=True) 
                           if d.get('edge_type') == 'undirected']
        
        if directed_edges:
            nx.draw_networkx_edges(self.graph, pos, edgelist=directed_edges,
                                  edge_color='blue', width=2, arrowsize=50,
                                  alpha=0.7, arrowstyle='->', label='Directed')
        
        if bidirected_edges:
            nx.draw_networkx_edges(self.graph, pos, edgelist=bidirected_edges,
                                  edge_color='red', width=2, arrowsize=50,
                                  alpha=0.7, arrowstyle='<->', label='Bidirected (Confounder)')
        
        if partial_edges:
            nx.draw_networkx_edges(self.graph, pos, edgelist=partial_edges,
                                  edge_color='orange', width=2, arrowsize=50,
                                  alpha=0.7, arrowstyle='->', style='dashed', label='Partial')
        
        if undirected_edges:
            nx.draw_networkx_edges(self.graph, pos, edgelist=undirected_edges,
                                  edge_color='gray', width=1.5, arrowsize=50,
                                  alpha=0.5, arrowstyle='-', label='Undirected')
        
        plt.title(title, fontsize=16)
        plt.legend(loc='upper right')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def print_edges(self):
        """
        Print all edges with their types
        """
        print("\n" + "="*60)
        print("Learned Causal Edges (PAG - Partial Ancestral Graph):")
        print("="*60)
        
        edge_dict = {}
        for u, v, d in self.graph.edges(data=True):
            edge_type = d.get('edge_type', 'unknown')
            if edge_type not in edge_dict:
                edge_dict[edge_type] = []
            edge_dict[edge_type].append((u, v))
        
        for edge_type, edges in edge_dict.items():
            print(f"\n{edge_type.upper()} edges ({len(edges)}):")
            for i, (source, target) in enumerate(edges, 1):
                if edge_type == 'directed':
                    print(f"  {i}. {source} -> {target}")
                elif edge_type == 'bidirected':
                    print(f"  {i}. {source} <-> {target}")
                elif edge_type == 'partial':
                    print(f"  {i}. {source} o-> {target}")
                elif edge_type == 'undirected':
                    print(f"  {i}. {source} o-o {target}")
        
        print("="*60)
        print("\nLegend:")
        print("  -> : Directed edge (A causes B)")
        print("  <->: Bidirected edge (latent confounder)")
        print("  o->: Partially oriented (uncertain direction)")
        print("  o-o: Undirected (no orientation)")
        print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("FCI (Fast Causal Inference) Baseline")
    print("Constraint-based causal discovery with latent confounders")
    print("="*60)
    
    baseline = FCIBaseline("../lucas0_train.csv")
    
    print("\nNote: LUCAS dataset is discrete (binary), using chi-square test")
    print("For continuous data, use independence_test='fisherz' instead")
    
    G, edges = baseline.run_fci(independence_test='chisq', alpha=0.05)
    
    baseline.convert_to_networkx()
    
    baseline.print_edges()
    
    baseline.visualize(title="FCI Algorithm - Partial Ancestral Graph (PAG)")

