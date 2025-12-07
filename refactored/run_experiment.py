"""
Universal Experiment Runner

This script automatically loads the dataset specified in config.py
and runs the selected causal discovery method.

Usage:
    python run_experiment.py --method gpt35_cot
    python run_experiment.py --method hybrid_fci_llm
    python run_experiment.py --method ges
"""

import sys
import argparse
from config import DATASETS, ACTIVE_DATASET, MAX_ITERATIONS, FCI_ALPHA, VALIDATION_ALPHA, get_output_dir

# Import all data loaders
from modules.data_loader import DataLoader, LUCASDataLoader, ALARMDataLoader


def get_data_loader(dataset_key):
    """
    Get appropriate data loader based on dataset configuration
    """
    if dataset_key not in DATASETS:
        print(f"[ERROR] Dataset '{dataset_key}' not found in config.py")
        print(f"[ERROR] Available datasets: {list(DATASETS.keys())}")
        sys.exit(1)
    
    dataset_config = DATASETS[dataset_key]
    data_path = dataset_config["path"]
    loader_name = dataset_config["loader"]
    
    print(f"\n{'='*60}")
    print(f"Loading Dataset: {dataset_key.upper()}")
    print(f"{'='*60}")
    print(f"Description: {dataset_config['description']}")
    print(f"Path: {data_path}")
    print(f"Loader: {loader_name}")
    print(f"{'='*60}\n")
    
    # Instantiate the appropriate loader
    if loader_name == "LUCASDataLoader":
        return LUCASDataLoader(data_path)
    elif loader_name == "ALARMDataLoader":
        return ALARMDataLoader(data_path)
    else:
        return DataLoader(data_path, dataset_name=dataset_key.upper())


def run_gpt35_cot(api_key):
    """Run GPT-3.5 with CoT"""
    from modules.api_clients import GPT35Client
    from modules.scanners import MutualInformationScanner
    from modules.prompt_generators import CoTPromptGenerator
    from modules.validators import ChiSquareValidator
    from modules.parsers import RobustDirectionParser
    from modules.visualizers import GraphVisualizer
    from modules.reporters import ReportGenerator
    
    output_dir = get_output_dir()
    print(f"[OUTPUT] Results will be saved to: {output_dir}/")
    
    data_loader = get_data_loader(ACTIVE_DATASET)
    df, nodes = data_loader.load_csv()
    graph = data_loader.create_empty_graph()
    
    print("[INIT] Setting up GPT-3.5 with CoT...")
    api_client = GPT35Client(api_key)
    scanner = MutualInformationScanner(df, graph, nodes)
    prompt_generator = CoTPromptGenerator(data_loader)
    validator = ChiSquareValidator(df)
    parser = RobustDirectionParser()
    visualizer = GraphVisualizer(output_dir)
    reporter = ReportGenerator(output_dir)
    
    print(f"\n[RUN] Starting causal discovery loop (max {MAX_ITERATIONS} iterations)...")
    
    step = 0
    validation_passed = 0
    validation_failed = 0
    
    while step < MAX_ITERATIONS:
        print(f"\n--- Iteration {step + 1}/{MAX_ITERATIONS} ---")
        
        candidates = scanner.scan(threshold=0.05)
        if not candidates:
            print("[DONE] No more candidates")
            break
        
        node_a, node_b, score = candidates[0]
        print(f"[SCAN] {node_a} <-> {node_b} (MI={score:.4f})")
        
        prompt = prompt_generator.generate(node_a, node_b)
        response = api_client.call(prompt)
        edge = parser.parse(response, node_a, node_b)
        
        if edge:
            is_valid, p_value = validator.validate(edge[0], edge[1], VALIDATION_ALPHA)
            if is_valid:
                print(f"[PASS] {edge[0]} -> {edge[1]} (p={p_value:.4f})")
                graph.add_edge(edge[0], edge[1])
                validation_passed += 1
            else:
                print(f"[FAIL] Rejected (p={p_value:.4f})")
                graph.add_edge(edge[0], edge[1], type='rejected')
                validation_failed += 1
        else:
            print("[SKIP] No clear direction")
            graph.add_edge(node_a, node_b, type='rejected')
        
        step += 1
    
    print(f"\n[STATS] Passed: {validation_passed}, Failed: {validation_failed}")
    
    # Save results
    reporter.save_cot_log(api_client.call_log, f"gpt35_cot_{ACTIVE_DATASET}")
    reporter.save_text_report(graph, model_name=f"GPT35_CoT_{ACTIVE_DATASET}", cot_log=api_client.call_log)
    reporter.save_edge_list(graph, model_name=f"GPT35_CoT_{ACTIVE_DATASET}")
    visualizer.visualize(graph, title=f"Causal Graph (GPT-3.5 CoT - {ACTIVE_DATASET.upper()})", 
                        filename=f"causal_graph_gpt35_cot_{ACTIVE_DATASET}")


def run_hybrid_fci_llm(api_key):
    """Run Hybrid FCI + GPT-3.5"""
    from modules.algorithms import FCIAlgorithm
    from modules.api_clients import GPT35Client
    from modules.prompt_generators import CoTPromptGenerator
    from modules.validators import ChiSquareValidator
    from modules.parsers import RobustDirectionParser
    from modules.visualizers import GraphVisualizer
    from modules.reporters import ReportGenerator
    
    output_dir = get_output_dir()
    print(f"[OUTPUT] Results will be saved to: {output_dir}/")
    
    data_loader = get_data_loader(ACTIVE_DATASET)
    df, nodes = data_loader.load_csv()
    
    print("[INIT] Setting up Hybrid FCI + GPT-3.5...")
    fci_algo = FCIAlgorithm(df, nodes)
    api_client = GPT35Client(api_key)
    prompt_generator = CoTPromptGenerator(data_loader)
    validator = ChiSquareValidator(df)
    parser = RobustDirectionParser()
    visualizer = GraphVisualizer(output_dir)
    reporter = ReportGenerator(output_dir)
    
    print("\n[STEP 1] Running FCI...")
    graph = fci_algo.run(independence_test='chisq', alpha=FCI_ALPHA)
    print(f"[FCI] Found {graph.number_of_edges()} edges")
    
    print("\n[STEP 2] Extracting ambiguous edges...")
    ambiguous_edges = fci_algo.get_ambiguous_edges()
    print(f"[FCI] Found {len(ambiguous_edges)} ambiguous edges")
    
    if ambiguous_edges:
        print("\n[STEP 3] LLM arbitration...")
        for idx, (node_a, node_b, edge_type) in enumerate(ambiguous_edges):
            print(f"\n[{idx+1}/{len(ambiguous_edges)}] {node_a} {edge_type} {node_b}")
            
            prompt = prompt_generator.generate(node_a, node_b)
            response = api_client.call(prompt)
            edge = parser.parse(response, node_a, node_b)
            
            if edge:
                is_valid, p_value = validator.validate(edge[0], edge[1], VALIDATION_ALPHA)
                if is_valid:
                    print(f"[RESOLVED] {edge[0]} -> {edge[1]}")
                    if graph.has_edge(node_a, node_b):
                        graph.remove_edge(node_a, node_b)
                    if graph.has_edge(node_b, node_a):
                        graph.remove_edge(node_b, node_a)
                    graph.add_edge(edge[0], edge[1], type='llm_resolved')
    
    # Save results
    reporter.save_cot_log(api_client.call_log, f"hybrid_fci_llm_{ACTIVE_DATASET}")
    reporter.save_text_report(graph, model_name=f"Hybrid_FCI_LLM_{ACTIVE_DATASET}", cot_log=api_client.call_log)
    reporter.save_edge_list(graph, model_name=f"Hybrid_FCI_LLM_{ACTIVE_DATASET}")
    visualizer.visualize(graph, title=f"Causal Graph (Hybrid FCI+LLM - {ACTIVE_DATASET.upper()})", 
                        filename=f"causal_graph_hybrid_fci_llm_{ACTIVE_DATASET}")


def run_ges():
    """Run GES baseline"""
    from modules.algorithms import GESAlgorithm
    from modules.visualizers import GraphVisualizer
    from modules.reporters import ReportGenerator
    
    output_dir = get_output_dir()
    print(f"[OUTPUT] Results will be saved to: {output_dir}/")
    
    data_loader = get_data_loader(ACTIVE_DATASET)
    df, nodes = data_loader.load_csv()
    
    print("[INIT] Setting up GES...")
    ges_algo = GESAlgorithm(df, nodes)
    visualizer = GraphVisualizer(output_dir)
    reporter = ReportGenerator(output_dir)
    
    print("\n[RUN] Running GES algorithm...")
    graph = ges_algo.run(score_func='local_score_BIC')
    
    # Save results
    reporter.save_text_report(graph, model_name=f"GES_{ACTIVE_DATASET}", cot_log=[])
    reporter.save_edge_list(graph, model_name=f"GES_{ACTIVE_DATASET}")
    visualizer.visualize(graph, title=f"Causal Graph (GES - {ACTIVE_DATASET.upper()})", 
                        filename=f"causal_graph_ges_{ACTIVE_DATASET}")


def run_fci():
    """Run FCI baseline"""
    from modules.algorithms import FCIAlgorithm
    from modules.visualizers import GraphVisualizer
    from modules.reporters import ReportGenerator
    
    output_dir = get_output_dir()
    print(f"[OUTPUT] Results will be saved to: {output_dir}/")
    
    data_loader = get_data_loader(ACTIVE_DATASET)
    df, nodes = data_loader.load_csv()
    
    print("[INIT] Setting up FCI...")
    fci_algo = FCIAlgorithm(df, nodes)
    visualizer = GraphVisualizer(output_dir)
    reporter = ReportGenerator(output_dir)
    
    print("\n[RUN] Running FCI algorithm...")
    graph = fci_algo.run(independence_test='chisq', alpha=FCI_ALPHA)
    
    # Save results
    reporter.save_text_report(graph, model_name=f"FCI_{ACTIVE_DATASET}", cot_log=[])
    reporter.save_edge_list(graph, model_name=f"FCI_{ACTIVE_DATASET}")
    visualizer.visualize(graph, title=f"Causal Graph (FCI - {ACTIVE_DATASET.upper()})", 
                        filename=f"causal_graph_fci_{ACTIVE_DATASET}")


def main():
    parser = argparse.ArgumentParser(description="Run causal discovery experiments")
    parser.add_argument("--method", type=str, required=True,
                       choices=["gpt35_cot", "gpt35_no_cot", "gemini", "zephyr", 
                               "hybrid_fci_llm", "hybrid_fci_zephyr", "ges", "fci"],
                       help="Causal discovery method to run")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key (for LLM methods)")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("CAUSAL DISCOVERY EXPERIMENT RUNNER")
    print("="*60)
    print(f"Method: {args.method}")
    print(f"Dataset: {ACTIVE_DATASET} (configured in config.py)")
    print("="*60 + "\n")
    
    # Check if API key is needed
    llm_methods = ["gpt35_cot", "gpt35_no_cot", "gemini", "zephyr", "hybrid_fci_llm", "hybrid_fci_zephyr"]
    
    if args.method in llm_methods:
        if not args.api_key:
            api_key = input("Please enter your API key: ").strip()
            if not api_key:
                print("[ERROR] API key required for this method!")
                sys.exit(1)
        else:
            api_key = args.api_key
    
    # Run the selected method
    if args.method == "gpt35_cot":
        run_gpt35_cot(api_key)
    elif args.method == "hybrid_fci_llm":
        run_hybrid_fci_llm(api_key)
    elif args.method == "ges":
        run_ges()
    elif args.method == "fci":
        run_fci()
    else:
        print(f"[ERROR] Method {args.method} not yet implemented in this runner")
        print("[INFO] Use the individual main_*.py scripts instead")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED!")
    print(f"Results saved to: {get_output_dir()}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

