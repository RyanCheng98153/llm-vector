import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache
import numpy as np
from transformers.modeling_outputs import CausalLMOutputWithPast

import copy

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
    ):
        # print the size of the kv_cache
        print("Cache 1 size", cache1.key_cache[0].size())
        print("Cache 2 size", cache2.key_cache[0].size())
        
        # print the length of the kv_cache
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
    
    def comparing_test1(self,
    ):
        
        knowledge: str = "Jack has a dog named Max, and he loves to play with him."
        prompt: str = "What type of pet does Jack have?"
        # Get KV cache for the prompt
        past_key_values: DynamicCache = self.get_kv_cache(knowledge, DynamicCache())
        origin_len = past_key_values.key_cache[0].shape[-2]
        
        kv_with_pastkv: DynamicCache = self.get_kv_cache(prompt, past_key_values)
        kv_no_pastkv: DynamicCache = self.get_kv_cache(prompt, DynamicCache())
        
        for i in range(len(past_key_values.key_cache)):
            kv_with_pastkv.key_cache[i] = kv_with_pastkv.key_cache[i][:, :, origin_len:, :]
            kv_with_pastkv.value_cache[i] = kv_with_pastkv.value_cache[i][:, :, origin_len:, :]
        
        print("kv_with_pastkv.key_cache[0][:, :, 0, :]")
        print(kv_with_pastkv.key_cache[0][:, :, 0, :])
        print("kv_no_pastkv.key_cache[0][:, :, 0, :]")
        print(kv_no_pastkv.key_cache[0][:, :, 0, :])
    
    def comparing_test2(self):
        
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
        
        print("past_prompt_kv.key_cache[0][:, :, 0:, :]")
        print(past_prompt_kv.key_cache[0][:, :, 0:, :])
        print("prompt_kv.key_cache[0][:, :, 0:, :]")
        print(prompt_kv.key_cache[0][:, :, 0:, :])
        
    def comparing_test3(self,
    ):
        """Modify KV cache by replacing old word with new word"""
        # Get token IDs for both words
        full_dog_prompt: str = "Jack has a dog named Max, and he loves to play with him."
        full_dog_kv: DynamicCache = self.get_kv_cache(full_dog_prompt)
        full_dog_kvlen = full_dog_kv.key_cache[0].shape[-2]
        # print("full_dog_kvlen: ", full_dog_kvlen)
        
        full_cat_prompt: str = "Jack has a cat named Max, and he loves to play with him."
        full_cat_kv: DynamicCache = self.get_kv_cache(full_cat_prompt)
        full_cat_kvlen = full_cat_kv.key_cache[0].shape[-2]
        # print("full_dog_kvlen: ", full_dog_kvlen)
        
        past_prompt: str = "Jack has a"
        past_prompt_kv = self.get_kv_cache(past_prompt)
        past_kvlen = past_prompt_kv.key_cache[0].shape[-2]
        # print("past_kvlen: ", past_kvlen)
        
        dog_prompt: str = "Jack has a dog"
        dog_prompt_kv = self.get_kv_cache(dog_prompt)
        dog_kvlen = dog_prompt_kv.key_cache[0].shape[-2]
        # print("dog_kvlen: ", dog_kvlen)
        
        cat_prompt: str = "Jack has a cat"
        cat_prompt_kv = self.get_kv_cache(cat_prompt)
        cat_kvlen = cat_prompt_kv.key_cache[0].shape[-2]
        # print("cat_kvlen: ", cat_kvlen)
        
        # Calculate the delta between dog and cat KV cache of the index of last token
        delta_key = cat_prompt_kv.key_cache[0][:, :, -1, :] - dog_prompt_kv.key_cache[0][:, :, -1, :]
        delta_value = cat_prompt_kv.value_cache[0][:, :, -1, :] - dog_prompt_kv.value_cache[0][:, :, -1, :]
        
        # Calculate the delta between full dog and full cat KV cache of the index of last token
        full_delta_key = full_cat_kv.key_cache[0][:, :, cat_kvlen-1, :] - full_dog_kv.key_cache[0][:, :, cat_kvlen-1, :]
        full_delta_value = full_cat_kv.value_cache[0][:, :, cat_kvlen-1, :] - full_dog_kv.value_cache[0][:, :, cat_kvlen-1, :]
        
        print("delta key_cache: dog - cat")
        print(delta_key)
        
        print("full sentence delta key_cache: dog - cat")
        print(full_delta_key)
        
        print("delta value_cache: dog - cat")
        print(delta_value)
        
        print("full sentence delta value_cache: dog - cat")
        print(full_delta_value)
        
        # Compare the key and value cache between delta and full delta
        if not torch.equal(delta_key, full_delta_key):
            print("delta_key and full_delta_key are not equal")
        else :
            print("delta_key and full_delta_key are equal")
    
    def comparing_test4(self,
    ):
        """Modify KV cache by replacing old word with new word"""
        # Get token IDs for both words
        full_dog_prompt: str = "Jack has a dog named Max, and he loves to play with him."
        full_dog_kv: DynamicCache = self.get_kv_cache(full_dog_prompt)
        full_dog_kvlen = full_dog_kv.key_cache[0].shape[-2]
        # print("full_dog_kvlen: ", full_dog_kvlen)
        
        full_cat_prompt: str = "Jack has a cat named Max, and he loves to play with him."
        full_cat_kv: DynamicCache = self.get_kv_cache(full_cat_prompt)
        full_cat_kvlen = full_cat_kv.key_cache[0].shape[-2]
        # print("full_dog_kvlen: ", full_dog_kvlen)
        
        past_prompt: str = "Jack has a"
        past_prompt_kv = self.get_kv_cache(past_prompt)
        past_kvlen = past_prompt_kv.key_cache[0].shape[-2]
        # print("past_kvlen: ", past_kvlen)
        
        dog_prompt: str = "Jack has a dog"
        dog_prompt_kv = self.get_kv_cache(dog_prompt)
        dog_kvlen = dog_prompt_kv.key_cache[0].shape[-2]
        # print("dog_kvlen: ", dog_kvlen)
        
        cat_prompt: str = "Jack has a cat"
        cat_prompt_kv = self.get_kv_cache(cat_prompt)
        cat_kvlen = cat_prompt_kv.key_cache[0].shape[-2]
        # print("cat_kvlen: ", cat_kvlen)
        
        # Calculate the delta between dog and cat KV cache of the index of last token
        delta_key = cat_prompt_kv.key_cache[0][:, :, -1, :] - dog_prompt_kv.key_cache[0][:, :, -1, :]
        delta_value = cat_prompt_kv.value_cache[0][:, :, -1, :] - dog_prompt_kv.value_cache[0][:, :, -1, :]
        
        # Calculate the delta between full dog and full cat KV cache of the index of last token
        # Copy the full dog KV cache to test_cat_kv deep copy
        test_cat_kv = copy.deepcopy(full_dog_kv)
        
        # print size of delta_key
        print("delta_key size: ", delta_key.size())
        # print size of full_dog_kv.key_cache[0][:, :, cat_kvlen-1:, :]
        print("full_dog_kv.key_cache[0][:, :, cat_kvlen-1:, :] size: ", full_dog_kv.key_cache[0][:, :, cat_kvlen-1:, :].size())
        # delta_key size:  torch.Size([1, 8, 128])
        # full_dog_kv.key_cache[0][:, :, cat_kvlen-1:, :] size:  torch.Size([1, 8, 12, 128])
        
        # Modify the token of cat_kvlen-1 with the delta_key and delta_value
        for i in range(len(full_dog_kv.key_cache)):
            delta_key_expanded = delta_key.unsqueeze(2).expand(-1, -1, full_dog_kv.key_cache[i].size(2), -1)
            delta_value_expanded = delta_value.unsqueeze(2).expand(-1, -1, full_dog_kv.value_cache[i].size(2), -1)
            test_cat_kv.key_cache[i][:, :, cat_kvlen-1, :] = full_dog_kv.key_cache[i][:, :, cat_kvlen-1, :] + delta_key_expanded[:, :, cat_kvlen-1, :]
            test_cat_kv.value_cache[i][:, :, cat_kvlen-1, :] = full_dog_kv.value_cache[i][:, :, cat_kvlen-1, :] + delta_value_expanded[:, :, cat_kvlen-1, :]
            
        print("\nCompare kv between full_cat v.s. full_dog + delta\n")
        print("full_cat_kv.key_cache[0][:, :, cat_kvlen-1, :]")
        print(full_cat_kv.key_cache[0][:, :, cat_kvlen-1, :])
        print("test_cat_kv.key_cache[0][:, :, cat_kvlen-1, :]")
        print(test_cat_kv.key_cache[0][:, :, cat_kvlen-1, :])
        print("full_dog_kv.key_cache[0][:, :, cat_kvlen-1, :]")
        print(full_dog_kv.key_cache[0][:, :, cat_kvlen-1, :])
        
        # Modify the token of cat_kvlen-1 with the delta_key and delta_value
        print(" ======= ")
        print("\nCompare the follow-up tokens kv between full_cat v.s. full_dog + delta\n")
        print("full_cat_kv.key_cache[0][:, :, cat_kvlen, :]")
        print(full_cat_kv.key_cache[0][:, :, cat_kvlen, :])
        print("test_cat_kv.key_cache[0][:, :, cat_kvlen, :]")
        print(test_cat_kv.key_cache[0][:, :, cat_kvlen, :])
        print("full_dog_kv.key_cache[0][:, :, cat_kvlen, :]")
        print(full_dog_kv.key_cache[0][:, :, cat_kvlen, :])
        print(" ======= ")
        
        for i in range(len(full_dog_kv.key_cache)):
            delta_key_expanded = delta_key.unsqueeze(2).expand_as(full_dog_kv.key_cache[i][:, :, cat_kvlen-1:, :])
            delta_value_expanded = delta_value.unsqueeze(2).expand_as(full_dog_kv.value_cache[i][:, :, cat_kvlen-1:, :])
            test_cat_kv.key_cache[i][:, :, cat_kvlen-1:, :] = full_dog_kv.key_cache[i][:, :, cat_kvlen-1:, :] + delta_key_expanded
            test_cat_kv.value_cache[i][:, :, cat_kvlen-1:, :] = full_dog_kv.value_cache[i][:, :, cat_kvlen-1:, :] + delta_value_expanded
        
        print("\nCompare the follow-up tokens kv between full_cat v.s. full_dog + delta\n")
        print("full_cat_kv.key_cache[0][:, :, cat_kvlen, :]")
        print(full_cat_kv.key_cache[0][:, :, cat_kvlen, :])
        print("test_cat_kv.key_cache[0][:, :, cat_kvlen, :]")
        print(test_cat_kv.key_cache[0][:, :, cat_kvlen, :])
        print("full_dog_kv.key_cache[0][:, :, cat_kvlen, :]")
        print(full_dog_kv.key_cache[0][:, :, cat_kvlen, :])
        
        # Compare the key and value cache between delta and full delta
        if not torch.equal(full_cat_kv.key_cache[0][:, :, cat_kvlen-1:, :], test_cat_kv.key_cache[0][:, :, cat_kvlen-1:, :]):
            print("full_cat_kv.key_cache and test_cat_kv.key_cache are not equal")
        
        # print("delta key_cache: dog - cat")
        # print(delta_key)
        
        # print("delta value_cache: dog - cat")
        # print(delta_value)
    
    def comparing_test5(self):
        pass
        
     
if __name__ == "__main__":
    modifier = KVCacheModifier(model, tokenizer)
    # print("[ Test 1 ]: Compare the first same token's KV cache between prompt with and without past KV cache\n")
    # modifier.comparing_test1()
    
    # print("[ Test 2 ]: Compare the first same token's KV cache between full sentence and partial sentence\n")
    # print(" Info: full sentence: 'Jack has a dog named Max, and he loves to play with him.'")
    # print(" Info: partial sentence: 'Jack has a dog'\n")
    # print(" Result: should be equal\n")
    # modifier.comparing_test2()
    
    # print("[ Test 3 ]\n")
    # print(" Info: Check the kv_cache delta of token \"dog\" or \"cat\" between full_dog and full_cat v.s. partial_dog and partial_cat\n")
    # print(" Instruction: full_dog - full_cat <= v.s. => partial_dog - partial_cat\n")
    # print(" Info: full dog sentence: 'Jack has a dog named Max, and he loves to play with him.")
    # print(" Info: full cat sentence: 'Jack has a cat named Max, and he loves to play with him.")
    # print(" Info: partial dog sentence: 'Jack has a dog")
    # print(" Info: partial cat sentence: 'Jack has a cat'\n")
    # modifier.comparing_test3()
    
    print("[ Test 4 ]\n")
    print(" Info: Modify KV cache by replacing old word with new word\n")
    print(" Instruction: full_dog - part_dog + part_cat -> full_cat \n")
    modifier.comparing_test4()
    
    # print("[ Test 5 ]\n")
    # print(" Info: See every follow-up token after \"cat\" in full_cat sentence when delta (dog - cat) is added to full_dog sentence\n")
    # print(" Instruction: full_dog - part_dog + part_cat -> full_cat \n")
    # modifier.comparing_test5()