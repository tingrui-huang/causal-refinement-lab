"""
Main Program: FCI + LLM (Zephyr-7B)

Strategy:
1. FCI does the heavy lifting (finds skeleton, handles confounders)
2. Zephyr-7B resolves ambiguous edges (o-o or o->)

This combines statistical rigor with open-source LLM reasoning!
"""

import sys
import os
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, will fall back to manual input
    pass

# Import config and utils
from config import get_output_dir
from utils import get_active_data_loader, print_dataset_info

# Import modules from the modules package
from modules.data_loader import DataLoader, LUCASDataLoader, ALARMDataLoader
from modules.algorithms import FCIAlgorithm
from modules.api_clients import ZephyrClient
from modules.prompt_generators import ZephyrCoTPromptGenerator
from modules.parsers import RobustDirectionParser
from modules.validators import ChiSquareValidator
from modules.visualizers import GraphVisualizer
from modules.reporters import ReportGenerator


class FCIZephyrPipeline:
    def __init__(self, data_loader, hf_token, output_dir=None):
        print("=" * 60)
        print("Initializing FCI + LLM (Zephyr-7B) Pipeline")
        print("=" * 60)
        
        self.output_dir = output_dir or get_output_dir()
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[OUTPUT] Results will be saved to: {self.output_dir}/")

        print("\n[1/6] Loading data...")
        self.data_loader = data_loader
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

        self.visualizer = GraphVisualizer(self.output_dir)
        self.reporter = ReportGenerator(self.output_dir)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "=" * 60)
        print("Pipeline initialized successfully!")
        print("=" * 60)
    
    def run(self, fci_alpha=0.05, validation_alpha=0.05):
        print(f"\n{'='*60}")
        print(f"Starting FCI + LLM (Zephyr-7B) Pipeline")
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
        self._print_fci_statistics()
        
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
    
    def _print_fci_statistics(self):
        """Print detailed statistics of FCI PAG structure"""
        print(f"\n{'='*60}")
        print("FCI PAG STRUCTURE")
        print(f"{'='*60}")
        
        # Count all edge types
        directed = sum(1 for u, v, d in self.graph.edges(data=True) 
                      if d.get('type') == 'directed')
        bidirected = sum(1 for u, v, d in self.graph.edges(data=True) 
                        if d.get('type') == 'bidirected')
        partial = sum(1 for u, v, d in self.graph.edges(data=True) 
                     if d.get('type') == 'partial')
        undirected = sum(1 for u, v, d in self.graph.edges(data=True) 
                        if d.get('type') == 'undirected')
        tail_tail = sum(1 for u, v, d in self.graph.edges(data=True) 
                       if d.get('type') == 'tail-tail')
        
        print(f"Edge Type Breakdown:")
        print(f"  Directed (->):      {directed:3d}  [certain causal direction]")
        print(f"  Bidirected (<->):   {bidirected:3d}  [latent confounder]")
        print(f"  Partial (o->/-o):   {partial:3d}  [ambiguous direction]")
        print(f"  Undirected (o-o):   {undirected:3d}  [completely ambiguous]")
        print(f"  Tail-tail (--):     {tail_tail:3d}  [no clear direction]")
        print(f"{'='*60}")
    
    def _save_results(self):
        print(f"\n{'='*60}")
        print("Saving Results")
        print(f"{'='*60}")
        
        # Save LLM call log
        self.reporter.save_cot_log(self.llm_client.call_log, "hybrid_fci_zephyr")
        
        # Save text report
        self.reporter.save_text_report(self.graph, 
                                      model_name="FCI_LLM_Zephyr",
                                      cot_log=self.llm_client.call_log)
        
        # Save edge list
        self.reporter.save_edge_list(self.graph, model_name="FCI_LLM_Zephyr")
        
        # Save visualization
        filename = f"causal_graph_fci_llm_zephyr_{self.timestamp}"
        self.visualizer.visualize(self.graph, 
                                 title="Causal Graph (FCI + LLM Zephyr-7B)",
                                 filename=filename,
                                 save_only=False,
                                 node_color='lightblue',
                                 edge_color='darkblue')
        
        print(f"{'='*60}")


def main():
    """Main function - runs FCI + Zephyr with parameters from config.py"""
    from config import FCI_ALPHA, VALIDATION_ALPHA
    
    print_dataset_info()
    
    # Get Hugging Face token from environment variable or user input
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    
    if hf_token:
        print("\n[INFO] Using Hugging Face token from environment variable")
    else:
        print("\n[WARN] HUGGINGFACE_TOKEN not found in environment")
        print("  Tip: Create a .env file with HUGGINGFACE_TOKEN=your_token")
        hf_token = input("\nPlease enter your Hugging Face token: ").strip()
        
        if not hf_token:
            print("[ERROR] Hugging Face token cannot be empty!")
            sys.exit(1)
    
    print(f"\nUsing parameters from config.py:")
    print(f"  FCI Alpha: {FCI_ALPHA}")
    print(f"  Validation Alpha: {VALIDATION_ALPHA}")
    
    # Initialize pipeline
    data_loader = get_active_data_loader()
    pipeline = FCIZephyrPipeline(data_loader, hf_token)
    
    # Run pipeline
    pipeline.run(fci_alpha=FCI_ALPHA, validation_alpha=VALIDATION_ALPHA)
    
    print("\n" + "=" * 60)
    print(f"All done! Check {get_output_dir()}/ for results.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

