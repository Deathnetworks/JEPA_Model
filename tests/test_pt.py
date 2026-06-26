import torch
from transformers import AutoTokenizer

def check_for_truncation(distilled_file_path, model_id="Jackrong/Qwopus3.6-27B-v2"):
    print("Loading tokenizer and data...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Load the distilled matrix data
    data = torch.load(distilled_file_path, map_location="cpu")
    
    input_tokens = data["input_tokens"]
    qwen_tokens = data["qwen_tokens"]
    
    num_samples = min(3, input_tokens.shape[0])
    print(f"\n--- Inspecting {num_samples} Samples for Truncation ---\n")
    
    for idx in range(num_samples):
        print(f"=== SAMPLE {idx} ===")
        
        # Decode the prompt sent to the teacher
        # Filter out padding tokens so it's clean
        prompt_ids = [t for t in input_tokens[idx].tolist() if t != tokenizer.pad_token_id]
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        print(f"👉 PROMPT:\n{prompt_text[:300]}...\n")
        
        # Decode what the Qwopus teacher generated
        gen_ids = qwen_tokens[idx].tolist()
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        
        print(f"🤖 TEACHER RESPONSE:")
        print(generated_text)
        print("-" * 50)
        
        # Diagnostics
        total_generated_tokens = len(gen_ids)
        print(f"Token Count: {total_generated_tokens}")
        
        # Check if it stopped abruptly
        if total_generated_tokens >= 250: # Close to a 256 threshold
            print("🚨 STATUS: LIKELY TRUNCATED! The token count is maxed out.")
        else:
            print("✅ STATUS: COMPLETED! The model stopped on its own (EOS).")
        print("=" * 60 + "\n")

if __name__ == "__main__":
    # Point to your local file path
    check_for_truncation(r"F:\JEPA_Model\distilled_data\distilled_agentic_set_0.pt")