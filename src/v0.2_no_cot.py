import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import openai
import sys


class CausalDiscoveryAgent:
    def __init__(self, data_path, api_key):
        self.df = pd.read_csv(data_path)
        self.nodes = self.df.columns.tolist()
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.nodes)

        self.client = openai.OpenAI(api_key=api_key)

        print(f"Loaded data: {self.df.shape}")

        self.descriptions = {
            "Smoking": "A binary variable indicating if the patient smokes.",
            "Lung_cancer": "A binary variable indicating if the patient has lung cancer.",
            "Genetics": "Genetic predisposition to cancer.",
            "Yellow_Fingers": "Discoloration of fingers, often associated with tar.",
            "Anxiety": "Mental health status.",
            "Peer_Pressure": "Social influence to smoke.",
        }

    def scanner(self, threshold=0.1):
        """
        [Scanner Module]
        """
        corr_matrix = self.df.corr().abs()

        candidates = []
        for i, node_a in enumerate(self.nodes):
            for j, node_b in enumerate(self.nodes):
                if i >= j: continue

                if self.graph.has_edge(node_a, node_b) or self.graph.has_edge(node_b, node_a):
                    continue

                score = corr_matrix.loc[node_a, node_b]
                if score > threshold:
                    candidates.append((node_a, node_b, score))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates

    def generate_prompt(self, node_a, node_b):
        """
        [Translator A: Graph -> Text]
        NO Chain-of-Thought - Direct question only
        """
        desc_a = self.descriptions.get(node_a, "A medical variable.")
        desc_b = self.descriptions.get(node_b, "A medical variable.")

        prompt = f"""You are an expert in Medical Science performing Causal Discovery.

Analyze the relationship between two variables:
Variable A: {node_a} ({desc_a})
Variable B: {node_b} ({desc_b})

The data shows a strong statistical correlation between them.

Determine the causal direction: Does A cause B, B cause A, or is there no direct causal link?

Answer in this exact format: "Direction: A->B", "Direction: B->A", or "Direction: None"."""
        return prompt

    def ask_expert(self, node_a, node_b):
        """
        [Expert Module]
        """
        prompt = self.generate_prompt(node_a, node_b)

        print(f"\n--- Asking LLM about {node_a} vs {node_b} ---")

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in medical causal inference."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            response_text = response.choices[0].message.content
            print(f"LLM Response: {response_text[:100]}...")
            return response_text
        except Exception as e:
            print(f"Error calling API: {e}")
            return "Direction: None"

    def parse_response(self, response_text, node_a, node_b):
        """
        [Translator B: Text -> Graph]
        """
        clean_text = response_text.replace(" ", "")
        
        if "Direction:A->B" in clean_text or "A->B" in clean_text:
            return (node_a, node_b)
        
        elif "Direction:B->A" in clean_text or "B->A" in clean_text:
            return (node_b, node_a)
            
        elif "Direction:A<->B" in clean_text:
            return None
            
        else:
            return None

    def run_loop(self, max_steps=5):
        """
        [The Loop] Algorithm 4 Main Loop
        """
        step = 0
        while step < max_steps:
            candidates = self.scanner()
            if not candidates:
                print("No more suspicious pairs found.")
                break

            node_a, node_b, score = candidates[0]

            response = self.ask_expert(node_a, node_b)
            print(f"Expert Response: {response}")

            edge = self.parse_response(response, node_a, node_b)
            if edge:
                print(f"Action: Adding edge {edge[0]} -> {edge[1]}")
                self.graph.add_edge(edge[0], edge[1])
            else:
                print("Action: No edge added (Expert rejected or unsure).")
                self.graph.add_edge(node_a, node_b, type='rejected')

            step += 1
            print("-" * 30)

    def visualize(self):
        pos = nx.circular_layout(self.graph)
        valid_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get('type') != 'rejected']

        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(self.graph, pos, node_size=2000, node_color='lightblue')
        nx.draw_networkx_labels(self.graph, pos)
        nx.draw_networkx_edges(self.graph, pos, edgelist=valid_edges, edge_color='green', width=2, arrowsize=50)
        plt.title("Expert-in-the-Loop Causal Graph (v0.2 - NO CoT)")
        plt.show()


if __name__ == "__main__":
    print("=" * 50)
    print("Causal Discovery with LLM Expert (NO CoT)")
    print("=" * 50)
    
    api_key = input("\nPlease enter your OpenAI API key: ").strip()
    
    if not api_key:
        print("Error: API key cannot be empty!")
        sys.exit(1)
    
    print("\nInitializing agent...")
    agent = CausalDiscoveryAgent("../lucas0_train.csv", api_key)
    
    max_steps_input = input("\nHow many iterations to run? (default: 30): ").strip()
    max_steps = int(max_steps_input) if max_steps_input else 50
    
    print(f"\nStarting causal discovery loop with max {max_steps} steps...")
    agent.run_loop(max_steps=max_steps)
    agent.visualize()

