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
from pathlib import Path

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
from modules.algorithms import FCIAlgorithm, RFCIAlgorithm
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

        # Select constraint-based algorithm (FCI vs RFCI) based on unified config.
        from config import get_current_dataset_config
        ds_cfg = get_current_dataset_config()
        self.constraint_algo = str(ds_cfg.get("constraint_algo", "fci")).lower()

        if self.constraint_algo == "rfci":
            print("\n[2/6] Setting up RFCI algorithm (Tetrad)...")
            self.fci_algo = RFCIAlgorithm(self.df, self.nodes, data_path=str(self.data_loader.data_path))
        else:
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
        
        # Step 1: Run constraint-based discovery (FCI or RFCI)
        print("\n" + "=" * 60)
        print(f"STEP 1: Running {self.constraint_algo.upper()} Algorithm")
        print("=" * 60)

        if self.constraint_algo == "rfci":
            from config import RFCI_ALPHA, RFCI_DEPTH, RFCI_MAX_DISC_PATH_LEN, RFCI_MAX_ROWS, VERBOSE as CFG_VERBOSE
            out_dir = Path(get_output_dir())

            # Reuse cached RFCI outputs if present (avoid expensive reruns).
            cached_csv = sorted(out_dir.glob("edges_RFCI_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            cached_report = sorted(out_dir.glob("report_RFCI_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)

            if cached_csv:
                print(f"[RFCI] Using cached edges CSV: {cached_csv[0].name}")
                self.graph = RFCIAlgorithm.load_graph_from_edges_csv(str(cached_csv[0]), nodes=self.nodes)
                # Ensure downstream ambiguous-edge extraction uses the cached graph
                self.fci_algo.graph = self.graph
            elif cached_report:
                print(f"[RFCI] Using cached report: {cached_report[0].name} (parsing edges)")
                self.graph = RFCIAlgorithm.load_graph_from_report_txt(str(cached_report[0]), nodes=self.nodes)
                # Ensure downstream ambiguous-edge extraction uses the cached graph
                self.fci_algo.graph = self.graph
            else:
                alpha = RFCI_ALPHA
                if abs(alpha - float(fci_alpha)) > 1e-12:
                    print(f"[INFO] RFCI_ALPHA={alpha} overrides fci_alpha={fci_alpha} for RFCI runs")
                self.graph = self.fci_algo.run(
                    alpha=float(alpha),
                    depth=int(RFCI_DEPTH),
                    max_disc_path_len=int(RFCI_MAX_DISC_PATH_LEN),
                    max_rows=RFCI_MAX_ROWS,
                    verbose=bool(CFG_VERBOSE),
                    output_edges_path=str(out_dir / f"edges_RFCI_{self.timestamp}.csv"),
                )
        else:
            from config import FCI_INDEPENDENCE_TEST
            self.graph = self.fci_algo.run(independence_test=FCI_INDEPENDENCE_TEST, alpha=fci_alpha)

        print(f"\n[{self.constraint_algo.upper()}] Initial graph has {self.graph.number_of_edges()} edges")
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
            from config import LLM_TEMPERATURE, LLM_MAX_TOKENS
            response = self.llm_client.call(prompt, temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS)
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

        # Name outputs based on constraint algorithm (FCI vs RFCI)
        algo_tag = "RFCI" if str(getattr(self, "constraint_algo", "fci")).lower() == "rfci" else "FCI"
        model_name = f"{algo_tag}_LLM_Zephyr"
        
        # Save LLM call log
        self.reporter.save_cot_log(self.llm_client.call_log, f"hybrid_{algo_tag.lower()}_zephyr")
        
        # Save text report
        self.reporter.save_text_report(self.graph, 
                                      model_name=model_name,
                                      cot_log=self.llm_client.call_log)
        
        # Save edge list
        self.reporter.save_edge_list(self.graph, model_name=model_name)
        
        # Save visualization
        filename = f"causal_graph_{algo_tag.lower()}_llm_zephyr_{self.timestamp}"
        self.visualizer.visualize(self.graph, 
                                 title=f"Causal Graph ({algo_tag} + LLM Zephyr-7B)",
                                 filename=filename,
                                 save_only=False,
                                 node_color='lightblue',
                                 edge_color='darkblue')
        
        print(f"{'='*60}")


def main():
    """Main function - runs FCI + Zephyr with parameters from config.py"""
    from config import FCI_ALPHA, VALIDATION_ALPHA, RANDOM_SEED

    # Best-effort reproducibility
    try:
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from reproducibility import set_global_seed

        set_global_seed(int(RANDOM_SEED))
    except Exception as e:
        print(f"[WARN] Could not set global seed: {e}")
    
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

