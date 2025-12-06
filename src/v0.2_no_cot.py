import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import openai
import sys
from sklearn.metrics import mutual_info_score
from scipy.stats import chi2_contingency
from report_generator import save_text_report, save_edge_list


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
        
        # Statistics for validation
        self.validation_passed = 0
        self.validation_failed = 0

    def scanner(self, threshold=0.05):
        """
        [Scanner Module v2.0]
        Use Mutual Information for discrete/binary data
        """
        candidates = []
        for i, node_a in enumerate(self.nodes):
            for j, node_b in enumerate(self.nodes):
                if i >= j: continue

                if self.graph.has_edge(node_a, node_b) or self.graph.has_edge(node_b, node_a):
                    continue

                mi_score = mutual_info_score(self.df[node_a], self.df[node_b])
                if mi_score > threshold:
                    candidates.append((node_a, node_b, mi_score))

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

    def validate_edge_with_data(self, node_a, node_b, significance_level=0.05):
        """
        [Data-Driven Validation]
        Verify if LLM's suggestion is supported by data using Chi-Square test.
        
        Returns:
        --------
        tuple: (is_valid, p_value)
            is_valid: True if data supports the edge (p < significance_level)
            p_value: statistical significance
        """
        # Create contingency table
        contingency_table = pd.crosstab(self.df[node_a], self.df[node_b])
        
        # Chi-square test for independence
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        is_valid = p_value < significance_level
        
        return is_valid, p_value
    
    def parse_response(self, response_text, node_a, node_b):
        """
        [Translator B: Text -> Graph]
        Enhanced version: handles variable names, A/B notation, arrows, etc.
        """
        text = response_text.lower().replace(" ", "").replace("\n", "")
        
        # normalize arrows
        text = text.replace("→", "->").replace("->", "->")
        
        # match patterns like "smoking->lung_cancer"
        pattern_ab = f"{node_a.lower()}->{node_b.lower()}"
        pattern_ba = f"{node_b.lower()}->{node_a.lower()}"
        
        if pattern_ab in text:
            return (node_a, node_b)
        elif pattern_ba in text:
            return (node_b, node_a)
        
        # match A->B or B->A
        elif "direction:a->b" in text or "a->b" in text:
            return (node_a, node_b)
        elif "direction:b->a" in text or "b->a" in text:
            return (node_b, node_a)
        elif "direction:a<->b" in text:
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
                # Data-Driven Validation
                is_valid, p_value = self.validate_edge_with_data(edge[0], edge[1])
                
                if is_valid:
                    print(f"[PASS] Data Validated: {edge[0]}->{edge[1]} (p={p_value:.4f})")
                    self.graph.add_edge(edge[0], edge[1])
                    self.validation_passed += 1
                else:
                    print(f"[FAIL] Data Conflict: LLM said yes, but data says independent (p={p_value:.4f})")
                    print(f"       -> Edge {edge[0]}->{edge[1]} REJECTED by validation")
                    self.graph.add_edge(edge[0], edge[1], type='rejected')
                    self.validation_failed += 1
            else:
                print("Action: No edge added (Expert rejected or unsure).")
                self.graph.add_edge(node_a, node_b, type='rejected')

            step += 1
            print("-" * 30)
        
        # Print validation statistics
        print("\n" + "="*50)
        print("VALIDATION STATISTICS")
        print("="*50)
        print(f"Edges passed validation:  {self.validation_passed}")
        print(f"Edges failed validation:  {self.validation_failed}")
        if (self.validation_passed + self.validation_failed) > 0:
            pass_rate = (self.validation_passed / (self.validation_passed + self.validation_failed)) * 100
            print(f"Validation pass rate:     {pass_rate:.1f}%")
        print("="*50)

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
    
    # Generate reports
    save_text_report(agent.graph, model_name="GPT-3.5-NoCoT", output_dir=".")
    save_edge_list(agent.graph, model_name="GPT-3.5-NoCoT", output_dir=".")
    
    agent.visualize()
