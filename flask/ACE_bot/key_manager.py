import google.generativeai as genai
import time

class KeyManager:
    # Defaulting to your preferred model version
    def __init__(self, keys, model_name="gemini-2.5-flash"):
        # Clean the list (remove None values if an env var is missing)
        self.keys = [k for k in keys if k]
        self.current_index = 0
        self.model_name = model_name
        
        if not self.keys:
            print("WARNING: No Backup Keys found in KeyManager!")
        else:
            print(f"Backup Manager initialized with {len(self.keys)} keys for {model_name}.")

    def _get_model(self):
        """Switches global config to the current key and returns model."""
        if not self.keys:
            raise ValueError("No API keys available.")
            
        api_key = self.keys[self.current_index]
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(self.model_name)

    def generate_content(self, contents):
        """
        Tries to generate content. If rate limited, rotates to next key.
        """
        attempts = 0
        max_attempts = len(self.keys)

        while attempts < max_attempts:
            try:
                # 1. Configure with current key
                model = self._get_model()
                
                # 2. Try generation
                response = model.generate_content(contents)
                return response

            except Exception as e:
                error_str = str(e)
                # Check for 429 (Rate Limit) or Resource Exhausted
                if "429" in error_str or "ResourceExhausted" in error_str:
                    print(f"⚠️ Backup Key {self.current_index + 1} exhausted. Switching...")
                    
                    # Rotate Index
                    self.current_index = (self.current_index + 1) % len(self.keys)
                    attempts += 1
                    time.sleep(0.5) 
                else:
                    # If it's a real error (like invalid prompt), raise it
                    raise e
        
        raise Exception("All backup keys are exhausted.")