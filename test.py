import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache
import torch
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
    past_key_values: DynamicCache,
    max_new_tokens: int = 20
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

def get_kv_cache(
    model,
    tokenizer,
    prompt: str,
) -> DynamicCache:
    
    # Get embedding layer device
    embed_device = model.model.embed_tokens.weight.device
    
    # Encode and move input to embedding device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(embed_device)
    
    # Initialize dynamic cache
    past_key_values = DynamicCache()
    
    # Generate KV cache with proper device placement
    with torch.no_grad():
        outputs:CausalLMOutputWithPast = model(
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

if __name__ == "__main__":
    
    knowledge = "Hello, how are you? -> Bonjour, comment ça va?"
    kv_cache = get_kv_cache(model, tokenizer, knowledge)
    
    prompt = "Translate the following English text to French: 'Hello, how are you?'"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output_ids = generate_text(model, input_ids, kv_cache)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True, temperature=None)
    
    print(generated_text)