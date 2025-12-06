"""
API Client Module

Handles communication with different LLM APIs (OpenAI, Google, HuggingFace, etc.)
"""

import openai
from datetime import datetime


class BaseAPIClient:
    
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        self.call_log = []
    
    def call(self, prompt):
        raise NotImplementedError("Subclasses must implement call()")
    
    def log_call(self, prompt, response, metadata=None):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "prompt": prompt,
            "response": response,
        }
        
        if metadata:
            log_entry.update(metadata)
        
        self.call_log.append(log_entry)


class GPT35Client(BaseAPIClient):
    def __init__(self, api_key):
        """
        Initialize GPT-3.5 client
        
        Parameters:
        -----------
        api_key : str
            OpenAI API key
        """
        super().__init__(api_key, "gpt-3.5-turbo")
        self.client = openai.OpenAI(api_key=api_key)
        print(f"[API] Initialized: {self.model_name}")
    
    def call(self, prompt, temperature=0.7, max_tokens=1000):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert in medical causal inference."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            response_text = response.choices[0].message.content
            
            # Log the call
            self.log_call(prompt, response_text, {
                "temperature": temperature,
                "max_tokens": max_tokens
            })
            
            return response_text
            
        except Exception as e:
            error_msg = f"Error calling API: {e}"
            print(f"[API ERROR] {error_msg}")
            return f"Reasoning: Error occurred. Direction: None"


class GPT35ClientNoCoT(GPT35Client):
    def __init__(self, api_key):
        super().__init__(api_key)
        self.model_name = "gpt-3.5-turbo-no-cot"
        print(f"[API] Mode: No Chain-of-Thought")


class GeminiClient(BaseAPIClient):
    def __init__(self, api_key, model_name="models/gemini-2.0-flash-exp"):
        super().__init__(api_key, model_name)
        
        # Import here to avoid dependency issues
        from google import genai
        
        self.client = genai.Client(api_key=api_key)
        print(f"[API] Initialized: {self.model_name}")
        
        # Check available models
        print("[API] Checking available models...")
        try:
            models = self.client.models.list()
            available = [m.name for m in models][:5]
            print(f"[API] Available models (first 5): {available}")
        except Exception as e:
            print(f"[API] Could not list models: {e}")
    
    def call(self, prompt, temperature=0.7, max_tokens=1000):
        import time
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            response_text = response.text if response.text else ""
            
            # Log the call
            self.log_call(prompt, response_text, {
                "model": self.model_name
            })
            
            # Rate limiting: sleep to avoid hitting limits
            time.sleep(4)
            
            return response_text
            
        except Exception as e:
            error_msg = f"Error calling Gemini API: {e}"
            print(f"[API ERROR] {error_msg}")
            print("[API] Sleeping for 20s due to rate limit...")
            
            import time
            time.sleep(20)
            
            return "Reasoning: Error occurred. Direction: None"


class HuggingFaceInferenceClient(BaseAPIClient):
    def __init__(self, hf_token, model_id="HuggingFaceH4/zephyr-7b-beta"):
        super().__init__(hf_token, model_id)
        
        # Import here to avoid dependency issues
        from huggingface_hub import InferenceClient
        
        self.client = InferenceClient(
            model=model_id,
            token=hf_token
        )
        
        print(f"[API] Initialized: {self.model_name}")
        print("[API] Using Hugging Face Inference API (no GPU needed)")
    
    def call(self, prompt, temperature=0.7, max_tokens=1000):
        try:
            # Use chat_completion instead of text_generation
            # This works with all HF Inference providers (together, featherless-ai, etc.)
            response = self.client.chat_completion(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            response_text = response.choices[0].message.content
            
            # Log the call
            self.log_call(prompt, response_text, {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "model": self.model_name
            })
            
            return response_text
            
        except Exception as e:
            import traceback
            error_msg = f"Error calling HF Inference API: {e}"
            print(f"[API ERROR] {error_msg}")
            print(f"[API ERROR] Error type: {type(e).__name__}")
            print("[API ERROR] Full traceback:")
            traceback.print_exc()
            
            return "Reasoning: Error occurred. Direction: None"


class ZephyrClient(HuggingFaceInferenceClient):
    def __init__(self, hf_token):
        super().__init__(hf_token, model_id="HuggingFaceH4/zephyr-7b-beta")
        print("[API] Model: Zephyr-7B-Beta (open-source, instruction-tuned)")


class MistralClient(HuggingFaceInferenceClient):
    def __init__(self, hf_token):
        super().__init__(hf_token, model_id="mistralai/Mistral-7B-Instruct-v0.3")
        print("[API] Model: Mistral-7B-Instruct (open-source)")

