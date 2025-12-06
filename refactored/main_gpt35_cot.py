"""
Main Program: GPT-3.5 with Chain-of-Thought
"""

import sys
import os
from datetime import datetime

# Import modules from the modules package
from modules.data_loader import LUCASDataLoader
from modules.api_clients import GPT35Client
from modules.scanners import MutualInformationScanner
from modules.prompt_generators import CoTPromptGenerator
from modules.validators import ChiSquareValidator
from modules.parsers import RobustDirectionParser
from modules.visualizers import GraphVisualizer
from modules.reporters import ReportGenerator


class CausalDiscoveryPipeline:
    """
    Main pipeline orchestrating all modules
    """
    
    def __init__(self, data_path, api_key, output_dir="outputs"):
        print("=" * 60)
        print("Initializing Causal Discovery Pipeline")
        print("=" * 60)

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[OUTPUT] Results will be saved to: {self.output_dir}/")

        print("\n[1/8] Loading data...")
        self.data_loader = LUCASDataLoader(data_path)
        self.df, self.nodes = self.data_loader.load_csv()
        self.graph = self.data_loader.create_empty_graph()

        print("\n[2/8] Connecting to API...")
        self.api_client = GPT35Client(api_key)

        print("\n[3/8] Setting up scanner...")
        self.scanner = MutualInformationScanner(self.df, self.graph, self.nodes)
        print("[SCANNER] Using Mutual Information (ideal for binary data)")

        print("\n[4/8] Setting up prompt generator...")
        self.prompt_generator = CoTPromptGenerator(self.data_loader)
        print("[PROMPT] Using Chain-of-Thought strategy")

        print("\n[5/8] Setting up validator...")
        self.validator = ChiSquareValidator(self.df)
        print("[VALIDATOR] Using Chi-Square test (ideal for categorical data)")

        print("\n[6/8] Setting up parser...")
        self.parser = RobustDirectionParser()
        print("[PARSER] Using robust multi-format parser")

        print("\n[7/8] Setting up visualizer...")
        self.visualizer = GraphVisualizer(output_dir)

        print("\n[8/8] Setting up reporter...")
        self.reporter = ReportGenerator(output_dir)

        self.validation_passed = 0
        self.validation_failed = 0
        
        # Metadata
        self.model_name = "gpt35_cot"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "=" * 60)
        print("Pipeline initialized successfully!")
        print("=" * 60)
    
    def run(self, max_steps=50, mi_threshold=0.05, significance_level=0.05):
        print(f"\n{'='*60}")
        print(f"Starting Causal Discovery Loop")
        print(f"Max iterations: {max_steps}")
        print(f"MI threshold: {mi_threshold}")
        print(f"Validation threshold: {significance_level}")
        print(f"{'='*60}\n")
        
        step = 0
        
        while step < max_steps:
            print(f"\n--- Iteration {step + 1}/{max_steps} ---")
            
            # Step 1: Scan for candidates
            candidates = self.scanner.scan(threshold=mi_threshold)
            
            if not candidates:
                print("[SCAN] No more candidate pairs found.")
                break
            
            # Step 2: Pick best candidate (RankedExpand)
            node_a, node_b, score = candidates[0]
            print(f"[SCAN] Best candidate: {node_a} <-> {node_b} (MI={score:.4f})")
            
            # Step 3: Generate prompt
            prompt = self.prompt_generator.generate(node_a, node_b)
            
            # Step 4: Ask expert (LLM)
            print(f"[LLM] Querying {self.api_client.model_name}...")
            response = self.api_client.call(prompt)
            print(f"[LLM] Response: {response[:100]}...")
            
            # Step 5: Parse response
            edge = self.parser.parse(response, node_a, node_b)
            
            if edge:
                print(f"[PARSE] Extracted edge: {edge[0]} -> {edge[1]}")
                
                # Step 6: Validate with data
                is_valid, p_value = self.validator.validate(edge[0], edge[1], 
                                                            significance_level)
                
                if is_valid:
                    print(f"[PASS] Data validation passed (p={p_value:.4f})")
                    self.graph.add_edge(edge[0], edge[1])
                    self.validation_passed += 1
                else:
                    print(f"[FAIL] Data validation failed (p={p_value:.4f})")
                    print(f"       Edge {edge[0]}->{edge[1]} REJECTED")
                    self.graph.add_edge(edge[0], edge[1], type='rejected')
                    self.validation_failed += 1
            else:
                print("[PARSE] No clear edge detected (LLM rejected or unsure)")
                self.graph.add_edge(node_a, node_b, type='rejected')
            
            step += 1
            print("-" * 60)
        
        print(f"\n{'='*60}")
        print("Causal Discovery Loop Completed")
        print(f"{'='*60}")
        
        # Print validation statistics
        self._print_statistics()
        
        # Save results
        self._save_results()
    
    def _print_statistics(self):
        """
        Print validation statistics
        """
        print(f"\n{'='*60}")
        print("VALIDATION STATISTICS")
        print(f"{'='*60}")
        print(f"Edges passed validation:  {self.validation_passed}")
        print(f"Edges failed validation:  {self.validation_failed}")
        
        total = self.validation_passed + self.validation_failed
        if total > 0:
            pass_rate = (self.validation_passed / total) * 100
            print(f"Validation pass rate:     {pass_rate:.1f}%")
        
        print(f"{'='*60}")
    
    def _save_results(self):
        """
        Save all results (logs, reports, visualizations)
        """
        print(f"\n{'='*60}")
        print("Saving Results")
        print(f"{'='*60}")
        
        # Save CoT log
        self.reporter.save_cot_log(self.api_client.call_log, self.model_name)
        
        # Save text report
        self.reporter.save_text_report(self.graph, 
                                      model_name="GPT-3.5-CoT",
                                      cot_log=self.api_client.call_log)
        
        # Save edge list
        self.reporter.save_edge_list(self.graph, model_name="GPT-3.5-CoT")
        
        # Save visualization
        filename = f"causal_graph_{self.model_name}_{self.timestamp}"
        self.visualizer.visualize(self.graph, 
                                 title="Causal Graph (GPT-3.5 + CoT)",
                                 filename=filename,
                                 save_only=False)
        
        print(f"{'='*60}")


def main():
    # Get API key
    api_key = input("\nPlease enter your OpenAI API key: ").strip()
    
    if not api_key:
        print("[ERROR] API key cannot be empty!")
        sys.exit(1)
    
    # Get max iterations
    max_steps_input = input("\nHow many iterations to run? (default: 50): ").strip()
    max_steps = int(max_steps_input) if max_steps_input else 50
    
    # Initialize pipeline
    data_path = "../lucas0_train.csv"
    pipeline = CausalDiscoveryPipeline(data_path, api_key, output_dir="../outputs")
    
    # Run pipeline
    pipeline.run(max_steps=max_steps)
    
    print("\n" + "=" * 60)
    print("All done! Check the outputs/ directory for results.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

