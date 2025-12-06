"""
Algorithm Module

Implements various causal discovery algorithms (GES, FCI, PC, etc.)
These are baseline methods that don't use LLMs.
"""

import numpy as np
import networkx as nx


class BaseAlgorithm:
    def __init__(self, dataframe, nodes):
        self.df = dataframe
        self.nodes = nodes
        self.data_matrix = dataframe.values
        self.graph = None
    
    def run(self):
        raise NotImplementedError("Subclasses must implement run()")


class GESAlgorithm(BaseAlgorithm):
    def __init__(self, dataframe, nodes):
        super().__init__(dataframe, nodes)
        print("[ALGORITHM] GES (Greedy Equivalence Search)")
        print("[ALGORITHM] Assumes: No latent confounders")
    
    def run(self, score_func='local_score_BIC'):
        try:
            from causallearn.search.ScoreBased.GES import ges
            
            print(f"[GES] Running with score function: {score_func}")
            print(f"[GES] Data shape: {self.data_matrix.shape}")
            
            # Run GES
            record = ges(self.data_matrix, score_func=score_func)
            
            # Extract graph
            learned_graph = record['G']
            
            print(f"[GES] Algorithm completed")
            print(f"[GES] Found {learned_graph.num_edges} edges")
            
            # Convert to NetworkX DiGraph
            self.graph = self._convert_to_networkx(learned_graph)
            
            return self.graph
            
        except ImportError:
            print("[ERROR] causal-learn not installed!")
            print("[ERROR] Install with: pip install causal-learn")
            raise
        except Exception as e:
            print(f"[ERROR] GES failed: {e}")
            raise
    
    def _convert_to_networkx(self, causallearn_graph):
        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)
        
        # Get adjacency matrix
        adj_matrix = causallearn_graph.graph
        
        # Add edges
        edge_count = 0
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                if i != j and adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                    # Directed edge: i -> j
                    graph.add_edge(self.nodes[i], self.nodes[j])
                    edge_count += 1
                elif i < j and adj_matrix[i, j] == -1 and adj_matrix[j, i] == -1:
                    # Undirected edge (in CPDAG): add both directions
                    graph.add_edge(self.nodes[i], self.nodes[j], type='undirected')
                    graph.add_edge(self.nodes[j], self.nodes[i], type='undirected')
                    edge_count += 1
        
        print(f"[GES] Converted to NetworkX: {edge_count} directed edges")
        
        return graph


class FCIAlgorithm(BaseAlgorithm):
    def __init__(self, dataframe, nodes):
        super().__init__(dataframe, nodes)
        print("[ALGORITHM] FCI (Fast Causal Inference)")
        print("[ALGORITHM] Allows: Latent confounders")
    
    def run(self, independence_test='fisherz', alpha=0.05):
        try:
            from causallearn.search.ConstraintBased.FCI import fci
            
            print(f"[FCI] Running with test: {independence_test}, alpha: {alpha}")
            print(f"[FCI] Data shape: {self.data_matrix.shape}")
            
            # Run FCI
            # For discrete data (like LUCAS), use 'chisq' or 'gsq'
            graph_result, edges = fci(
                self.data_matrix,
                independence_test_method=independence_test,
                alpha=alpha
            )
            
            print(f"[FCI] Algorithm completed")
            print(f"[FCI] Found {len(edges)} edges")
            
            # Convert to NetworkX DiGraph
            self.graph = self._convert_to_networkx(graph_result)
            
            return self.graph
            
        except ImportError:
            print("[ERROR] causal-learn not installed!")
            print("[ERROR] Install with: pip install causal-learn")
            raise
        except Exception as e:
            print(f"[ERROR] FCI failed: {e}")
            raise
    
    def _convert_to_networkx(self, causallearn_graph):
        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)
        
        # Get adjacency matrix
        adj_matrix = causallearn_graph.graph
        
        # Add edges
        edge_count = 0
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                if i != j and adj_matrix[i, j] != 0:
                    # Various edge types in PAG:
                    # -1: arrowhead, 1: tail, 2: circle
                    
                    if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                        # i -> j (directed)
                        graph.add_edge(self.nodes[i], self.nodes[j], type='directed')
                        edge_count += 1
                    elif adj_matrix[i, j] == 2 and adj_matrix[j, i] == 1:
                        # i o-> j (partially directed)
                        graph.add_edge(self.nodes[i], self.nodes[j], type='partial')
                        edge_count += 1
                    elif i < j and adj_matrix[i, j] == 2 and adj_matrix[j, i] == 2:
                        # i o-o j (undirected with circles)
                        graph.add_edge(self.nodes[i], self.nodes[j], type='bidirected')
                        graph.add_edge(self.nodes[j], self.nodes[i], type='bidirected')
                        edge_count += 1
        
        print(f"[FCI] Converted to NetworkX: {edge_count} edges")
        
        return graph

