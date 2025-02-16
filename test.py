import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache
import numpy as np
from transformers.modeling_outputs import CausalLMOutputWithPast

# Load model directly
model_name = "meta-llama/Llama-3.1-8B-Instruct"
device = "cuda" # "cuda" for GPU usage or "cpu" for CPU usage

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=f"./.cache/{model_name}")
model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=f"./.cache/{model_name}").to(device)

torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])

def generate_with_cache(
    input_ids: torch.Tensor,
    model: PreTrainedModel,
    max_new_tokens: int = 20,
    past_key_values: DynamicCache = None
) -> torch.Tensor:
    
    # Get the device of the embedding layer
    embed_device = model.model.embed_tokens.weight.device
    
    origin_ids = input_ids
    # Move input to the same device as embedding layer
    input_ids = input_ids.to(embed_device)
    
    # Initialize output tensor on embedding device
    output_ids = input_ids.clone()
    next_token = input_ids
    
    # Main generation loop
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass with proper device placement
            outputs = model(
                input_ids=next_token,  # Only process last token
                past_key_values=past_key_values,
                use_cache=True
            )
            
            # Get next token prediction (logits will be on the last device)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
            
            # Move next token to embedding device for next iteration
            next_token = next_token.to(embed_device)
            
            # Update KV cache
            past_key_values = outputs.past_key_values
            
            # Append prediction
            output_ids = torch.cat([output_ids, next_token], dim=1)
            
            # Optional: Check for EOS token
            #print(next_token.item())
            #print(model.config.eos_token_id)
            if next_token.item() in model.config.eos_token_id:
                break
    
    # return output_ids[:,origin_ids.shape[-1]:]
    return output_ids[:,:]
    
def generate_text(
    input_ids: torch.Tensor,
    model: PreTrainedModel, 
    max_new_tokens: int = 20
) -> torch.Tensor:
    
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,  # Set the maximum length of the generated text
        do_sample=False,  # Ensures greedy decoding,
        temperature=None
    )
        
    # return output_ids[:,origin_ids.shape[-1]:]
    return output_ids[:,:]

class KVCacheModifier:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model:PreTrainedModel = model
        self.tokenizer:PreTrainedTokenizer = tokenizer
    
    def get_kv_cache(
        self,
        prompt: str,
        past_key_values: DynamicCache = None,
    ) -> DynamicCache:
        
        # Get embedding layer device
        embed_device = self.model.model.embed_tokens.weight.device
        
        # Encode and move input to embedding device
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(embed_device)
        
        if past_key_values is None:
            past_key_values = DynamicCache()
        
        # Generate KV cache with proper device placement
        with torch.no_grad():
            outputs:CausalLMOutputWithPast = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False
            )
        
        # The model's device mapping will automatically place each layer's 
        # KV cache on the correct device
        # print(outputs)
        return outputs.past_key_values
    
    @staticmethod
    def compare_cache(
        cache1: DynamicCache,
        cache2: DynamicCache,
        print_diff: bool = True
    ) -> bool:
        print("Cache 1 length", cache1.key_cache[0].shape[-2])
        print("Cache 2 length", cache2.key_cache[0].shape[-2])
        
        # check if test1_cache and test2_cache are equal
        if print_diff:
            for i in range(min(len(cache1.key_cache), len(cache2.key_cache))):
                if not torch.equal(cache1.key_cache[i], cache2.key_cache[i]):
                    print(f"Layer {i} key_cache is not equal")
                if not torch.equal(cache1.value_cache[i], cache2.value_cache[i]):
                    print(f"Layer {i} value_cache is not equal")
        
        for i in range(len(cache1.key_cache)):
            if not torch.equal(cache1.key_cache[i], cache2.key_cache[i]):
                return False
            if not torch.equal(cache1.value_cache[i], cache2.value_cache[i]):
                return False
        
        if print_diff:
            print("Cache 1 and Cache 2 are equal")
        
        return True

    
    def comparing_test1(self,
    ) -> bool:
        
        knowledge: str = "Jack has a dog named Max, and he loves to play with him."
        prompt: str = "What type of pet does Jack have?"
        # Get KV cache for the prompt
        past_key_values: DynamicCache = self.get_kv_cache(knowledge, DynamicCache())
        origin_len = past_key_values.key_cache[0].shape[-2]
        
        test1_cache: DynamicCache = self.get_kv_cache(prompt, past_key_values)
        test2_cache: DynamicCache = self.get_kv_cache(prompt, DynamicCache())
        
        
        for i in range(len(past_key_values.key_cache)):
            test1_cache.key_cache[i] = test1_cache.key_cache[i][:, :, origin_len:, :]
            test1_cache.value_cache[i] = test1_cache.value_cache[i][:, :, origin_len:, :]
        
        return self.compare_cache(test1_cache, test2_cache, print_diff=True)
    
    
    def comparing_test2(self,
    ) -> bool:
        
        prompt: str = "Jack has a dog named Max, and he loves to play with him."
        prompt_kv = self.get_kv_cache(prompt)
        
        past_prompt: str = "Jack has a dog"
        past_prompt_kv = self.get_kv_cache(past_prompt)
        
        # Compare if the KV cache is the same between the prompt and the past prompt
        origin_len = past_prompt_kv.key_cache[0].shape[-2]
        print("origin_len: ", origin_len)
        
        for i in range(len(prompt_kv.key_cache)):
            prompt_kv.key_cache[i] = prompt_kv.key_cache[i][:, :, :origin_len, :]
            prompt_kv.value_cache[i] = prompt_kv.value_cache[i][:, :, :origin_len, :]
        
        return self.compare_cache(prompt_kv, past_prompt_kv, print_diff=True)
        
    def comparing_test3(self,
    ) -> bool:
        """Modify KV cache by replacing old word with new word"""
        # Get token IDs for both words
        origin_prompt: str = "Jack has a dog named Max, and he loves to play with him."
        origin_prompt_kv = self.get_kv_cache(origin_prompt)
        origin_kvlen = origin_prompt_kv.key_cache[0].shape[-2]
        print("origin_kvlen: ", origin_kvlen)
        
        past_prompt: str = "Jack has a "
        past_prompt_kv = self.get_kv_cache(past_prompt)
        past_kvlen = past_prompt_kv.key_cache[0].shape[-2]
        print("past_kvlen: ", past_kvlen)
        
        dog_prompt: str = "Jack has a dog"
        dog_prompt_kv = self.get_kv_cache(dog_prompt)
        dog_kvlen = dog_prompt_kv.key_cache[0].shape[-2]
        print("dog_kvlen: ", dog_kvlen)
        
        cat_prompt: str = "Jack has a cat"
        cat_prompt_kv = self.get_kv_cache(cat_prompt)
        cat_kvlen = cat_prompt_kv.key_cache[0].shape[-2]
        print("cat_kvlen: ", cat_kvlen)
        
        verify_prompt: str = "Jack has a cat named Max, and he loves to play with him."
        verify_prompt_kvlen = self.get_kv_cache(verify_prompt)
        verify_kvlen = verify_prompt_kvlen.key_cache[0].shape[-2]
        print("verify_kvlen: ", verify_kvlen)
        
        # for i in range(len(prompt_kv.key_cache)):
        #     prompt_kv.key_cache[i] = prompt_kv.key_cache[i][:, :, :origin_len, :]
        #     prompt_kv.value_cache[i] = prompt_kv.value_cache[i][:, :, :origin_len, :]
        
        # return self.compare_cache(prompt_kv, past_prompt_kv, print_diff=True)
        
    
    def archieve_modify_kv_cache(self, 
        past_key_values: DynamicCache,
        old_word: str,
        new_word: str
    ) -> DynamicCache:
        """Modify KV cache by replacing old word with new word"""
        # Get token IDs for both words
        old_token_ids = self.tokenizer.encode(old_word, add_special_tokens=False).to(device)
        new_token_ids = self.tokenizer.encode(new_word, add_special_tokens=False).to(device)
        
        modified_cache: DynamicCache = DynamicCache()
        # past_key_values.key_cache # [32, 1, 8, length, 128]
        
        # Generate new cache for the replacement word
        with torch.no_grad():
            new_outputs = self.model(
                input_ids=new_token_ids,
                past_key_values=DynamicCache(),
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False
            )
        new_cache: DynamicCache = new_outputs.past_key_values
        
        # Modify each layer's cache
        
        # for i in range(len(past_key_values.key_cache)):
        #     kv.key_cache[i] = kv.key_cache[i][:, :, :origin_len, :]
        #     kv.value_cache[i] = kv.value_cache[i][:, :, :origin_len, :]
        
        for i in range(len(past_key_values.key_cache)):
            for layer_idx in range(len(past_key_values)):
                past_key_values.key_cache[i][layer_idx] = past_key_values.key_cache[i][layer_idx][:, :, :origin_len, :]

        
        for layer_idx in range(len(past_key_values)):
            
            # Copy original KV pairs
            layer_k = past_key_values.key_cache[layer_idx].clone()
            layer_v = past_key_values.value_cache[layer_idx].clone()
            
            # Find position of old token sequence
            seq_len = layer_k.size(-2)
            print(seq_len)
            found = False
            for pos in range(seq_len - len(old_token_ids) + 1):
                if torch.equal(
                    layer_k[..., pos:pos+len(old_token_ids), :].cpu(),
                    new_cache.key_cache[layer_idx][..., :len(old_token_ids), :].cpu()
                ):
                    found = True
                    # Replace the KV pairs
                    layer_k[..., pos:pos+len(old_token_ids), :] = new_cache.key_cache[layer_idx][..., :len(old_token_ids), :].to(device)
                    layer_v[..., pos:pos+len(old_token_ids), :] = new_cache[layer_idx][1][..., :len(old_token_ids), :].to(device)
                    break
            
            if not found:
                print(f"Warning: Could not find exact match for replacement in layer {layer_idx}")


            print("layer_idx: ", layer_idx)
            
            # print(past_key_values[layer_idx])

            # Store modified KV pairs in new cache
            modified_cache.key_cache[layer_idx] = layer_k
            modified_cache.value_cache[layer_idx][0] = layer_v
            
        return modified_cache
     
if __name__ == "__main__":
    
    modifier = KVCacheModifier(model, tokenizer)
    # modifier.comparing_test1()
    # modifier.comparing_test2()
    modifier.comparing_test3()
    
    # Note: The following are the notes for the test cases
    # Test 1: Compare the KV cache between prompt with and without past KV cache
    # Test 2: Compare the KV cache between full sentence and partial sentence
    # : full sentence: "Jack has a dog named Max, and he loves to play with him."
    # : partial sentence: "Jack has a dog"
    # Test 3: Modify the KV cache by replacing old word with new word