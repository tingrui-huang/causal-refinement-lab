"""
Tuebingen LLM Direction Resolver

Pure LLM-based direction resolution for Tuebingen continuous variable pairs.
Unlike FCI+LLM (which resolves ambiguous edges), this directly asks LLM
about the causal direction between two variables.

Strategy: Soft Initialization (方案 A)
- LLM provides initial bias (e.g., 0.6 vs 0.4)
- GSB (Gradient-based Symmetry Breaking) can still override if data says otherwise
- This gives correct direction a "head start" without forcing it
"""

import os
import sys
from pathlib import Path
import re
from typing import Tuple, Optional

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import existing API client infrastructure
from refactored.modules.api_clients import GPT35Client


class TuebingenLLMDirectionResolver:
    """
    LLM-based direction resolver for Tuebingen pairs
    
    Uses existing API client infrastructure from refactored/modules/api_clients.py
    Supports any model configured in config.py (GPT-3.5, GPT-4, Zephyr, etc.)
    """
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize LLM client using existing infrastructure
        
        Args:
            model: Model name (if None, reads from config.py)
        """
        # Try to load .env file from project root (in case it wasn't loaded yet)
        env_path = Path(__file__).parent.parent.parent / '.env'
        if env_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(dotenv_path=env_path, override=True)
            except ImportError:
                # Manually parse .env if python-dotenv not available
                try:
                    with open(env_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                os.environ[key.strip()] = value.strip()
                except Exception:
                    pass
        
        # Import config to get API key and model settings
        from config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
        
        # Use provided model or fall back to config
        self.model = model or LLM_MODEL
        if not self.model:
            raise ValueError("No LLM model specified. Set LLM_MODEL in config.py or pass model parameter.")
        
        # Get API key from environment (same as refactored pipeline)
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable.\n"
                "Tip: Create a .env file with OPENAI_API_KEY=your_key"
            )
        
        # Use existing GPT35Client infrastructure
        self.client = GPT35Client(self.api_key)
        
        # Store config settings
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
        self.call_log = []
        
        print(f"[LLM] Using existing API client infrastructure")
        print(f"[LLM] Model: {self.model}")
        print(f"[LLM] Temperature: {self.temperature}")
        print(f"[LLM] Max Tokens: {self.max_tokens}")
    
    def get_direction_prior(
        self, 
        var_x: str, 
        var_y: str, 
        pair_id: str = "unknown",
        var_x_description: Optional[str] = None,
        var_y_description: Optional[str] = None,
        context: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Ask LLM about causal direction between two variables
        
        Args:
            var_x: First variable name (e.g., "x")
            var_y: Second variable name (e.g., "y")
            pair_id: Pair identifier (for context)
            var_x_description: Semantic description (e.g., "altitude")
            var_y_description: Semantic description (e.g., "temperature")
            context: Additional context about the dataset
        
        Returns:
            (forward_weight, backward_weight) tuple
            e.g., (0.6, 0.4) means LLM thinks x->y is more likely
        
        Strategy (方案 C: 语义增强):
            - Only use LLM if semantic descriptions are available
            - If no semantics, return neutral (0.5, 0.5)
            - LLM confidence → ±0.1 weight adjustment
            - Example: confidence=0.8 → forward=0.6, backward=0.4
            - GSB can still override if data strongly disagrees
        """
        # Use descriptions if available, otherwise use variable names
        x_display = var_x_description if var_x_description else var_x
        y_display = var_y_description if var_y_description else var_y
        
        # Check if we have semantic info
        has_semantic = (
            var_x_description and 
            var_y_description and 
            var_x_description != var_x and 
            var_y_description != var_y
        )
        
        if not has_semantic:
            print(f"\n[LLM SKIP] No semantic information for {var_x}, {var_y}")
            print(f"[LLM SKIP] Variable names are too abstract for LLM judgment")
            print(f"[LLM SKIP] Returning neutral prior (0.5, 0.5)")
            return 0.5, 0.5
        
        print(f"\n[LLM] Consulting {self.model} for direction prior...")
        print(f"      Variables: {x_display[:60]}{'...' if len(x_display) > 60 else ''}")
        print(f"                 {y_display[:60]}{'...' if len(y_display) > 60 else ''}")
        print(f"      Pair: {pair_id}")
        
        # Generate prompt with semantic info
        prompt = self._generate_prompt(x_display, y_display, pair_id, context)
        
        # Call LLM using existing client infrastructure
        try:
            # Use the existing GPT35Client.call() method
            response_text = self.client.call(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Log the call
            self.call_log.append({
                "pair_id": pair_id,
                "var_x": var_x,
                "var_y": var_y,
                "prompt": prompt,
                "response": response_text
            })
            
            print(f"[LLM] Response received ({len(response_text)} chars)")
            
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            print(f"[LLM] Falling back to neutral prior (0.5, 0.5)")
            return 0.5, 0.5
        
        # Parse response
        direction, confidence = self._parse_response(response_text, var_x, var_y)
        
        print(f"[LLM] Parsed direction: {direction}")
        print(f"[LLM] Parsed confidence: {confidence:.2f}")
        
        # Convert to weights (方案 A: 温和初始化)
        forward, backward = self._convert_to_weights(direction, confidence, var_x, var_y)
        
        print(f"[LLM] Final weights: {var_x}->{var_y} = {forward:.2f}, {var_y}->{var_x} = {backward:.2f}")
        
        return forward, backward
    
    def _generate_prompt(self, var_x_desc: str, var_y_desc: str, pair_id: str, context: Optional[str] = None) -> str:
        """
        Generate prompt for LLM with semantic information
        
        Uses Chain-of-Thought (CoT) prompting for better reasoning.
        """
        context_str = f"\nDataset Context: {context}\n" if context else ""
        
        prompt = f"""You are an expert in causal inference analyzing variable pairs from real-world data.

Task: Determine the most plausible causal direction between two correlated variables.

Dataset: Tuebingen Pairs (Pair {pair_id}){context_str}
Variables:
- Variable X: {var_x_desc}
- Variable Y: {var_y_desc}

Context: These variables show statistical correlation in observational data. Your task is to determine which causal direction is more plausible based on domain knowledge, physical principles, and logical reasoning.

Instructions:
1. Think step-by-step about the nature of each variable
2. Consider which direction makes more physical/logical/temporal sense
3. Think about what could cause what (e.g., altitude causes temperature, not vice versa)
4. Provide your confidence level (0.5 = no preference, 1.0 = very confident)

Response format (IMPORTANT - follow exactly):
Direction: X->Y or Y->X or None
Confidence: [number between 0.5 and 1.0]
Reasoning: [brief explanation of your causal reasoning]

Example 1:
Direction: X->Y
Confidence: 0.90
Reasoning: Altitude (X) physically determines temperature (Y) due to atmospheric pressure gradient. Higher altitude leads to lower air pressure and thus lower temperature. The reverse (temperature causing altitude) makes no physical sense.

Example 2:
Direction: Y->X
Confidence: 0.85
Reasoning: Solar radiation (Y) causes air temperature (X) through heating. The sun's energy heats the air, not the other way around. Temperature cannot cause solar radiation.

Your analysis:"""
        
        return prompt
    
    def _parse_response(self, response_text: str, var_x: str, var_y: str) -> Tuple[str, float]:
        """
        Parse LLM response to extract direction and confidence
        
        Returns:
            (direction, confidence) tuple
            direction: "X->Y", "Y->X", or "None"
            confidence: float between 0.5 and 1.0
        """
        # Extract direction
        direction = "None"
        if re.search(r'Direction:\s*X->Y', response_text, re.IGNORECASE):
            direction = "X->Y"
        elif re.search(r'Direction:\s*Y->X', response_text, re.IGNORECASE):
            direction = "Y->X"
        elif re.search(r'Direction:\s*None', response_text, re.IGNORECASE):
            direction = "None"
        else:
            # Fallback: search for variable names in reasoning
            if var_x.lower() in response_text.lower() and "cause" in response_text.lower():
                if response_text.lower().find(var_x.lower()) < response_text.lower().find(var_y.lower()):
                    direction = "X->Y"
                else:
                    direction = "Y->X"
        
        # Extract confidence
        confidence = 0.5  # Default: no preference
        confidence_match = re.search(r'Confidence:\s*(0\.\d+|1\.0)', response_text, re.IGNORECASE)
        if confidence_match:
            confidence = float(confidence_match.group(1))
            # Clamp to [0.5, 1.0]
            confidence = max(0.5, min(1.0, confidence))
        
        return direction, confidence
    
    def _convert_to_weights(self, direction: str, confidence: float, var_x: str, var_y: str) -> Tuple[float, float]:
        """
        Convert LLM direction + confidence to initial weights
        
        Strategy (方案 A: 温和初始化):
            - Base weight: 0.5 (neutral)
            - LLM adjustment: ±(confidence - 0.5) * 0.2
            - Example: confidence=0.8 → adjustment = 0.3 * 0.2 = 0.06
            - Result: forward=0.56, backward=0.44 (modest 0.12 advantage)
        
        This gives correct direction a "起跑优势" but not overwhelming.
        GSB can still override if data strongly disagrees.
        """
        base_weight = 0.5
        
        # Scale confidence to adjustment
        # confidence ∈ [0.5, 1.0] → adjustment ∈ [0, 0.1]
        adjustment = (confidence - 0.5) * 0.2  # Max adjustment = 0.1
        
        if direction == "X->Y":
            forward = base_weight + adjustment
            backward = base_weight - adjustment
        elif direction == "Y->X":
            forward = base_weight - adjustment
            backward = base_weight + adjustment
        else:  # No preference
            forward = base_weight
            backward = base_weight
        
        return forward, backward
    
    def save_log(self, output_path: str):
        """Save call log to file for debugging"""
        import json
        with open(output_path, 'w') as f:
            json.dump(self.call_log, f, indent=2)
        print(f"[LLM] Call log saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Test the resolver
    print("Testing TuebingenLLMDirectionResolver...")
    print("=" * 80)
    
    try:
        resolver = TuebingenLLMDirectionResolver()
        
        # Test case: Altitude -> Temperature
        forward, backward = resolver.get_direction_prior("Altitude", "Temperature", "pair0001")
        
        print(f"\nResult:")
        print(f"  Altitude -> Temperature: {forward:.2f}")
        print(f"  Temperature -> Altitude: {backward:.2f}")
        print(f"  Advantage: {abs(forward - backward):.2f}")
        
        print("\n✓ Test successful!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()



