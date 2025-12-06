"""
Data Loading Module

Handles loading and preprocessing of datasets for causal discovery.
"""

import pandas as pd
import networkx as nx


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.nodes = None
        
    def load_csv(self):
        self.df = pd.read_csv(self.data_path)
        self.nodes = self.df.columns.tolist()
        
        print(f"[DATA] Loaded: {self.df.shape}")
        print(f"[DATA] Variables: {self.nodes}")
        
        return self.df, self.nodes
    
    def create_empty_graph(self):
        if self.nodes is None:
            raise ValueError("Data not loaded. Call load_csv() first.")
        
        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)
        
        return graph


class LUCASDataLoader(DataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        
        # Domain-specific variable descriptions
        self.descriptions = {
            "Smoking": "A binary variable indicating if the patient smokes.",
            "Lung_cancer": "A binary variable indicating if the patient has lung cancer.",
            "Genetics": "Genetic predisposition to cancer.",
            "Yellow_Fingers": "Discoloration of fingers, often associated with tar.",
            "Anxiety": "Mental health status.",
            "Peer_Pressure": "Social influence to smoke.",
        }
    
    def get_description(self, node_name):
        return self.descriptions.get(node_name, "A medical variable.")

