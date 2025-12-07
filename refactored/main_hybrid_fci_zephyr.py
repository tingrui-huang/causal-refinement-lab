"""
Main Program: Hybrid FCI + Zephyr

Strategy:
1. FCI does the heavy lifting (finds skeleton, handles confounders)
2. Zephyr-7B resolves ambiguous edges (o-o or o->)

This combines statistical rigor with open-source LLM reasoning!
"""

import sys
import os
from datetime import datetime

from modules.data_loader import LUCASDataLoader
from modules.algorithms import FCIAlgorithm
from modules.api_clients import ZephyrClient
from modules.prompt_generators import ZephyrCoTPromptGenerator
from modules.parsers import RobustDirectionParser
from modules.validators import ChiSquareValidator
from modules.visualizers import GraphVisualizer
from modules.reporters import ReportGenerator


class HybridFCIZephyrPipeline:
    def __init__(self, data_path, hf_token, output_dir="outputs"):
        print("=" * 60)
        print("Initializing Hybrid FCI + Zephyr Pipeline")
        print("=" * 60)
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[OUTPUT] Results will be saved to: {self.output_dir}/")

        print("\n[1/6] Loading data...")
        self.data_loader = LUCASDataLoader(data_path)
        self.df, self.nodes = self.data_loader.load_csv()

        print("\n[2/6] Setting up FCI algorithm...")
        self.fci_algo = FCIAlgorithm(self.df, self.nodes)

        print("\n[3/6] Connecting to Zephyr-7B API...")
        self.llm_client = ZephyrClient(hf_token)

        print("\n[4/6] Setting up prompt generator...")
        self.prompt_generator = ZephyrCoTPromptGenerator(self.data_loader)

        print("\n[5/6] Setting up parser...")
        self.parser = RobustDirectionParser()

        print("\n[6/6] Setting up validator...")
        self.validator = ChiSquareValidator(self.df)

        self.visualizer = GraphVisualizer(output_dir)
        self.reporter = ReportGenerator(output_dir)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "=" * 60)
        print("Pipeline initialized successfully!")
        print("=" * 60)
    
    def run(self, fci_alpha=0.05, validation_alpha=0.05):
        print(f"\n{'='*60}")
        print(f"Starting Hybrid FCI + Zephyr Pipeline")
        print(f"FCI alpha: {fci_alpha}")
        print(f"Validation alpha: {validation_alpha}")
        print(f"{'='*60}\n")
        
        # Step 1: Run FCI
        print("\n" + "=" * 60)
        print("STEP 1: Running FCI Algorithm")
        print("=" * 60)
        
        # Use chisq for discrete data
        self.graph = self.fci_algo.run(independence_test='chisq', alpha=fci_alpha)
        
        print(f"\n[FCI] Initial graph has {self.graph.number_of_edges()} edges")
        
        # Step 2: Extract ambiguous edges
        print("\n" + "=" * 60)
        print("STEP 2: Extracting Ambiguous Edges")
        print("=" * 60)
        
        ambiguous_edges = self.fci_algo.get_ambiguous_edges()
        
        if not ambiguous_edges:
            print("[INFO] No ambiguous edges found! FCI resolved everything.")
            print("[INFO] Skipping LLM consultation.")
        else:
            print(f"\n[INFO] Found {len(ambiguous_edges)} ambiguous edges:")
            for node_a, node_b, edge_type in ambiguous_edges:
                print(f"  - {node_a} {edge_type} {node_b}")
        
        # Step 3: LLM arbitration
        if ambiguous_edges:
            print("\n" + "=" * 60)
            print("STEP 3: Zephyr Arbitration")
            print("=" * 60)
            
            self._llm_arbitration(ambiguous_edges, validation_alpha)
        
        # Step 4: Save results
        self._save_results()
    
    def _llm_arbitration(self, ambiguous_edges, validation_alpha):
        resolved_count = 0
        validated_count = 0
        rejected_count = 0
        
        for idx, (node_a, node_b, edge_type) in enumerate(ambiguous_edges):
            print(f"\n--- Edge {idx + 1}/{len(ambiguous_edges)} ---")
            print(f"[AMBIGUOUS] {node_a} {edge_type} {node_b}")

            prompt = self.prompt_generator.generate(node_a, node_b)

            print(f"[LLM] Consulting Zephyr-7B...")
            response = self.llm_client.call(prompt)
            print(f"[LLM] Response: {response[:100]}...")

            edge = self.parser.parse(response, node_a, node_b)
            
            if edge:
                print(f"[LLM] Suggested direction: {edge[0]} -> {edge[1]}")

                is_valid, p_value = self.validator.validate(edge[0], edge[1], 
                                                            validation_alpha)
                
                if is_valid:
                    print(f"[PASS] Data validation passed (p={p_value:.4f})")

                    if self.graph.has_edge(node_a, node_b):
                        self.graph.remove_edge(node_a, node_b)
                    if self.graph.has_edge(node_b, node_a):
                        self.graph.remove_edge(node_b, node_a)
                    
                    self.graph.add_edge(edge[0], edge[1], type='llm_resolved')
                    
                    resolved_count += 1
                    validated_count += 1
                else:
                    print(f"[FAIL] Data validation failed (p={p_value:.4f})")
                    print(f"       Keeping original FCI edge")
                    rejected_count += 1
            else:
                print("[LLM] No clear direction suggested")
                print("       Keeping original FCI edge")
            
            print("-" * 60)

        print(f"\n{'='*60}")
        print("ZEPHYR ARBITRATION STATISTICS")
        print(f"{'='*60}")
        print(f"Total ambiguous edges:    {len(ambiguous_edges)}")
        print(f"Zephyr resolved:          {resolved_count}")
        print(f"Data validated:           {validated_count}")
        print(f"Data rejected:            {rejected_count}")
        print(f"Kept as-is:               {len(ambiguous_edges) - resolved_count}")
        print(f"{'='*60}")
    
    def _save_results(self):
        print(f"\n{'='*60}")
        print("Saving Results")
        print(f"{'='*60}")
        
        # Save LLM call log
        self.reporter.save_cot_log(self.llm_client.call_log, "hybrid_fci_zephyr")
        
        # Save text report
        self.reporter.save_text_report(self.graph, 
                                      model_name="Hybrid_FCI_Zephyr",
                                      cot_log=self.llm_client.call_log)
        
        # Save edge list
        self.reporter.save_edge_list(self.graph, model_name="Hybrid_FCI_Zephyr")
        
        # Save visualization
        filename = f"causal_graph_hybrid_fci_zephyr_{self.timestamp}"
        self.visualizer.visualize(self.graph, 
                                 title="Causal Graph (Hybrid: FCI + Zephyr)",
                                 filename=filename,
                                 save_only=False,
                                 node_color='lightblue',
                                 edge_color='darkblue')
        
        print(f"{'='*60}")


def main():
    hf_token = input("\nPlease enter your Hugging Face token: ").strip()
    
    if not hf_token:
        print("[ERROR] Hugging Face token cannot be empty!")
        sys.exit(1)
    
    # Initialize pipeline
    data_path = "../lucas0_train.csv"
    pipeline = HybridFCIZephyrPipeline(data_path, hf_token, output_dir="../outputs")
    
    # Run pipeline
    pipeline.run(fci_alpha=0.05, validation_alpha=0.05)
    
    print("\n" + "=" * 60)
    print("All done! Check the outputs/ directory for results.")
    print("Compare with GPT-3.5 hybrid and pure approaches!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

