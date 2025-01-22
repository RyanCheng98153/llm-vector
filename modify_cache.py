from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Dict, Tuple, List
import torch
import numpy as np

class KVCacheModifier:
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(self.device)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=f"./.cache/{model_name}").to(self.device)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=f"./.cache/{model_name}")
    
    def get_kv_cache(self, prompt: str) -> DynamicCache:
        """Using your provided get_kv_cache implementation"""
        embed_device = self.model.model.embed_tokens.weight.device
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(embed_device)
        
        past_key_values = DynamicCache()
        
        with torch.no_grad():
            outputs: CausalLMOutputWithPast = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False
            )
        
        return outputs.past_key_values
    
    def modify_kv_cache(self, 
                       past_key_values: DynamicCache,
                       old_word: str,
                       new_word: str) -> DynamicCache:
        """Modify KV cache by replacing old word with new word"""
        # Get token IDs for both words
        old_token_ids = self.tokenizer.encode(old_word, add_special_tokens=False)
        new_token_ids = self.tokenizer.encode(new_word, add_special_tokens=False)
        
        # Get new word embeddings
        embed_device = self.model.model.embed_tokens.weight.device
        new_input_ids = torch.tensor([new_token_ids], device=embed_device)
        
        # Generate new cache for the replacement word
        new_cache = DynamicCache()
        with torch.no_grad():
            new_outputs = self.model(
                input_ids=new_input_ids,
                past_key_values=new_cache,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False
            )
        new_cache = new_outputs.past_key_values
        
        # Modify each layer's cache
        modified_cache = DynamicCache()
        
        
        for layer_idx in range(len(past_key_values)):
            # Get devices for current layer
            k_device = past_key_values[layer_idx][0].device
            v_device = past_key_values[layer_idx][1].device
            
            # Copy original KV pairs
            layer_k = past_key_values[layer_idx][0].clone()
            layer_v = past_key_values[layer_idx][1].clone()
            
            # Find position of old token sequence
            seq_len = layer_k.size(-2)
            print(seq_len)
            found = False
            for pos in range(seq_len - len(old_token_ids) + 1):
                if torch.equal(
                    layer_k[..., pos:pos+len(old_token_ids), :].cpu(),
                    new_cache[layer_idx][0][..., :len(old_token_ids), :].cpu()
                ):
                    found = True
                    # Replace the KV pairs
                    layer_k[..., pos:pos+len(old_token_ids), :] = new_cache[layer_idx][0][..., :len(old_token_ids), :].to(k_device)
                    layer_v[..., pos:pos+len(old_token_ids), :] = new_cache[layer_idx][1][..., :len(old_token_ids), :].to(v_device)
                    break
            
            if not found:
                print(f"Warning: Could not find exact match for replacement in layer {layer_idx}")
            
            # Store modified KV pairs in new cache
            modified_cache[layer_idx] = (layer_k, layer_v)
        
        return modified_cache
    
    def cache_to_text(self, past_key_values: DynamicCache, prompt: str) -> str:
        """Convert KV cache back to text for testing"""
        embed_device = self.model.model.embed_tokens.weight.device
        
        # Create input ids for continuation
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(embed_device)
        
        # Generate from the cache
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids[:, -1:],  # Only use last token
                past_key_values=past_key_values,
                use_cache=True
            )
        
        # Get next token
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        
        # Decode the complete sequence
        full_ids = torch.cat([input_ids[0], next_token])

        return self.tokenizer.decode(full_ids)
    
    def debug_cache(self, cache: DynamicCache):
        """Debug function to print cache information"""
        print(f"Number of layers: {len(cache)}")
        for i, (k, v) in enumerate(cache):
            print(f"Layer {i}:")
            print(f"  Key shape: {k.shape}")
            print(f"  Value shape: {v.shape}")
    
    def run_experiment(self, prompt: str, old_word: str, new_word: str, debug: bool = True) -> str:
        """Run the complete experiment"""
        # Get original cache
        original_cache:DynamicCache = self.get_kv_cache(prompt)
        
        # print(np.array(original_cache).shape)
        
        if debug:
            print("Original cache info:")
            # self.debug_cache(original_cache)
            print()
        
        result = self.cache_to_text(original_cache, prompt)
        print(result)
        
        # Modify cache
        modified_cache = self.modify_kv_cache(original_cache, old_word, new_word)
        
        if debug:
            print("\nModified cache info:")
            # self.debug_cache(modified_cache)
            print()
        
        # Convert modified cache to text
        result = self.cache_to_text(modified_cache, prompt)
        
        return result

def main():
    # Example usage
    modifier = KVCacheModifier()
    
    prompt = "Jack has a dog. Its name is Tommy. it likes to play with balls."
    old_word = "Tommy"
    new_word = "Becky"
    
    result = modifier.run_experiment(prompt, old_word, new_word)
    print(f"\nModified text: {result}")

if __name__ == "__main__":
    main()