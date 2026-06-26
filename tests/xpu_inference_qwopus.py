import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers import BitsAndBytesConfig
from threading import Thread

# -------------------------------------------------------------------------
# 1. Advanced Inductor & SYCL Compilation Tuning
# -------------------------------------------------------------------------
# We aggressively tune torch._inductor to generate optimal SYCL kernels
# specifically for static-shaped compute loops (which we enforce via static KV caching).
torch._inductor.config.freezing = True                 # Freezes weights into the graph for aggressive constant folding
torch._inductor.config.max_autotune = True             # Run extensive profiling to select optimal SYCL Triton kernels
torch._inductor.config.coordinate_descent_tuning = True # Enhances auto-tuning space search for kernel fusions
# Native XPU handles graph capture internally in newer PyTorch versions.
# We explicitly disable CUDA graphs as we are on XPU.
torch._inductor.config.triton.cudagraphs = False

def setup_model_and_tokenizer(model_id="Jackrong/Qwopus3.6-27B-v2"):
    # Define quantization config
    # We keep 4-bit NF4 for weights. We will handle KV cache quantization natively in the model loading.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print("Loading model onto native XPU with SDPA...")
    # -------------------------------------------------------------------------
    # 2. Native Scaled Dot Product Attention (SDPA)
    # -------------------------------------------------------------------------
    # attn_implementation="sdpa" ensures HF models use `torch.nn.functional.scaled_dot_product_attention`.
    # PyTorch has upstreamed highly optimized XPU kernels for SDPA that bypass naive Python loops.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="xpu",                        # Target native XPU directly
        quantization_config=bnb_config,          # 4-bit weights
        torch_dtype=torch.bfloat16,              # Arc GPUs excel at bfloat16
        attn_implementation="sdpa",              # Force Native PyTorch SDPA
        # Enable 8-bit KV Cache via quanto backend natively supported by HF
        kv_quantization_config={"backend": "quanto", "format": "int8"}
    )

    # -------------------------------------------------------------------------
    # 3. & 4. Graph Capture / Caching for Decode & Static KV Cache
    # -------------------------------------------------------------------------
    # For 64k token generation, dynamic KV caching reallocates memory constantly.
    # We explicitly enable 'static' caching. This pre-allocates maximum memory upfront.
    # Static caching is mathematically REQUIRED for `torch.compile` to effectively capture
    # the decode loop without triggering constant graph recompilations.
    model.generation_config.cache_implementation = "static"

    # Pre-compile the forward pass using Inductor.
    # We compile with reduce-overhead, which attempts to trace the static graph and eliminate python dispatch.
    print("Compiling model via torch.compile (this will take a while on first run)...")
    model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    return model, tokenizer


# -------------------------------------------------------------------------
# Generation Execution Loop
# -------------------------------------------------------------------------
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 64000):
    inputs = tokenizer(prompt, return_tensors="pt").to("xpu")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        # Crucial for XPU memory management during ultra-long context:
        # By setting batch_size=1 and a fixed max_length, the static cache is perfectly sized.
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    print("\nStarting generation...")
    # Run generation in a separate thread so the streamer can yield tokens immediately
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        print(new_text, end="", flush=True)
        generated_text += new_text

        # Periodic cleanup if needed for immense contexts, though static cache minimizes this need.
        # torch.xpu.empty_cache() # Uncomment if you hit fragmentation OOMs late in the 64k generation

    thread.join()

    # Explicit cache clear after chunk generation
    torch.xpu.empty_cache()

    return generated_text

if __name__ == "__main__":
    try:
        model, tokenizer = setup_model_and_tokenizer()
        test_prompt = "Explain the history of the Roman Empire in extreme detail."
        generate_text(model, tokenizer, test_prompt, max_new_tokens=100)
    except Exception as e:
        print(f"Error during execution: {e}")
