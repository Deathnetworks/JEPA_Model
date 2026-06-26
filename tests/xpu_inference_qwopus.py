import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers import BitsAndBytesConfig
from threading import Thread

# -------------------------------------------------------------------------
# Advanced Inductor & SYCL Compilation Tuning (Native Upstream)
# -------------------------------------------------------------------------
# We aggressively tune torch._inductor to generate optimal SYCL kernels.
# NOTE: IPEX is deprecated. These flags target the native XPU stack.
torch._inductor.config.freezing = True                 
torch._inductor.config.max_autotune = True             
torch._inductor.config.coordinate_descent_tuning = True 
torch._inductor.config.triton.cudagraphs = False # Disable CUDA graphs for XPU

def setup_model_and_tokenizer(model_id="Jackrong/Qwopus3.6-27B-v2"):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("[SYSTEM] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print("[SYSTEM] Loading model onto native XPU with SDPA...")
    # -------------------------------------------------------------------------
    # Native XPU Generation Config
    # -------------------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="xpu",                        # Strict native targeting
        quantization_config=bnb_config,          
        torch_dtype=torch.bfloat16,              
        attn_implementation="sdpa",              # Force PyTorch SDPA kernels
        kv_quantization_config={"backend": "quanto", "format": "int8"}
    )

    # Enforce static cache to prevent VRAM fragmentation over 64k tokens
    model.generation_config.cache_implementation = "static"

    print("[SYSTEM] Compiling model via torch.compile (SYCL Fusion)...")
    model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    return model, tokenizer

def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 4096):
    inputs = tokenizer(prompt, return_tensors="pt").to("xpu")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    print("\n[QWOPUS 27B] Generating...\n")
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        print(new_text, end="", flush=True)
        generated_text += new_text

    thread.join()
    
    if hasattr(torch.xpu, 'empty_cache'):
        torch.xpu.empty_cache()

    return generated_text

if __name__ == "__main__":
    try:
        model, tokenizer = setup_model_and_tokenizer()
        test_prompt = "Write a low-level C++ function to allocate aligned memory on an Intel XPU."
        generate_text(model, tokenizer, test_prompt, max_new_tokens=512)
    except Exception as e:
        print(f"Error during execution: {e}")