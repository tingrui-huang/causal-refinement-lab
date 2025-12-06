import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys
import json
from datetime import datetime
from huggingface_hub import InferenceClient


class CausalDiscoveryAgentZephyr:
    def __init__(self, data_path, hf_token):
        self.df = pd.read_csv(data_path)
        self.nodes = self.df.columns.tolist()
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.nodes)

        print(f"Loaded data: {self.df.shape}")
        print(f"\nInitializing Hugging Face Inference API...")
        
        # Using Zephyr-7B (proven to work with text_generation on HF Inference)
        # Many newer models get routed to "together" provider which only supports chat
        model_id = "HuggingFaceH4/zephyr-7b-beta"
        
        self.client = InferenceClient(
            model=model_id,
            token=hf_token
        )
        
        # Initialize CoT reasoning log
        self.cot_log = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"cot_reasoning_zephyr_{timestamp}.json"
        
        print(f"Inference client ready! Using {model_id}")
        print("(No GPU needed, running on HF cloud)")
        print(f"[LOG] CoT reasoning will be saved to: {self.log_file}")

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
        WITH Chain-of-Thought Prompting
        """
        desc_a = self.descriptions.get(node_a, "A medical variable.")
        desc_b = self.descriptions.get(node_b, "A medical variable.")

        prompt = f"""[INST] You are an expert in Medical Science performing Causal Discovery.

Task: Analyze the relationship between two variables from a dataset:
Variable A: {node_a} ({desc_a})
Variable B: {node_b} ({desc_b})

Context: The data shows a strong statistical correlation between them.

Instructions:
1. Think step-by-step about the biological or logical mechanism.
2. Determine if A causes B, B causes A, or if they are likely confounded (no direct link).
3. Provide your reasoning briefly.
4. Final Answer format: "Direction: A->B", "Direction: B->A", or "Direction: None".

Reasoning: [/INST]"""
        return prompt

    def ask_expert(self, node_a, node_b):
        """
        [Expert Module] - Using chat_completion (works with all providers)
        """
        prompt = self.generate_prompt(node_a, node_b)

        print(f"\n--- Asking Zephyr about {node_a} vs {node_b} ---")

        try:
            # Use chat_completion instead of text_generation
            # This works with all HF Inference providers
            response = self.client.chat_completion(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=128,
                temperature=0.7,
            )
            
            response_text = response.choices[0].message.content
            print(f"Zephyr Response: {response_text[:100]}...")
            
            # Log CoT reasoning
            self.cot_log.append({
                "timestamp": datetime.now().isoformat(),
                "model": "HuggingFaceH4/zephyr-7b-beta",
                "node_a": node_a,
                "node_b": node_b,
                "prompt": prompt,
                "response": response_text,
                "has_cot": True
            })
            
            return response_text
        except Exception as e:
            import traceback
            print(f"\n✗ Error calling HF Inference API:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {repr(e)}")
            print("\nFull traceback:")
            traceback.print_exc()
            return "Direction: None"

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
                print(f"Action: Adding edge {edge[0]} -> {edge[1]}")
                self.graph.add_edge(edge[0], edge[1])
            else:
                print("Action: No edge added (Expert rejected or unsure).")
                self.graph.add_edge(node_a, node_b, type='rejected')

            step += 1
            print("-" * 30)
        
        # Save CoT log at the end
        self.save_cot_log()
    
    def save_cot_log(self):
        """
        Save CoT reasoning log to JSON file
        """
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.cot_log, f, indent=2, ensure_ascii=True)
        print(f"\n[LOG] CoT reasoning saved to: {self.log_file}")
        print(f"[LOG] Total reasoning steps: {len(self.cot_log)}")

    def visualize(self):
        pos = nx.circular_layout(self.graph)
        valid_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get('type') != 'rejected']

        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(self.graph, pos, node_size=2000, node_color='lightcoral')
        nx.draw_networkx_labels(self.graph, pos)
        nx.draw_networkx_edges(self.graph, pos, edgelist=valid_edges, edge_color='purple', width=2, arrowsize=50)
        plt.title("Expert-in-the-Loop Causal Graph (Zephyr-7B via Inference API)")
        plt.show()


if __name__ == "__main__":
    print("="*60)
    print("Causal Discovery with Zephyr-7B-Beta")
    print("Using Hugging Face Inference API (NO GPU NEEDED)")
    print("="*60)
    
    print("\n" + "!"*60)
    print("IMPORTANT: You need a Hugging Face token")
    print("1. Get your token from https://huggingface.co/settings/tokens")
    print("2. Zephyr-7B is open-source, no special access needed")
    print("3. Proven to work with HF Inference text_generation")
    print("\nAdvantages:")
    print("  - No GPU required")
    print("  - No model download")
    print("  - Runs on HF cloud")
    print("  - No provider routing issues")
    print("!"*60 + "\n")
    
    hf_token = input("Please enter your Hugging Face token: ").strip()
    
    if not hf_token:
        print("Error: Token cannot be empty!")
        sys.exit(1)
    
    print("\nInitializing agent with Zephyr-7B Inference API...")
    agent = CausalDiscoveryAgentZephyr("../lucas0_train.csv", hf_token=hf_token)
    
    max_steps_input = input("\nHow many iterations to run? (default: 50): ").strip()
    max_steps = int(max_steps_input) if max_steps_input else 50
    
    print(f"\nStarting causal discovery loop with max {max_steps} steps...")
    agent.run_loop(max_steps=max_steps)
    agent.visualize()

