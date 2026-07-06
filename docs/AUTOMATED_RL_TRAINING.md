# Automated Reinforcement Learning & GRPO Training

This document details the concepts, mathematical foundations, and programmatic practices required to fully automate Group Relative Policy Optimization (GRPO) and Reinforcement Learning (RL) training loops within the Mamba-JEPA ecosystem.

It is designed to be highly detailed so that both human engineers and AI agents can utilize it as a direct blueprint to implement hands-free optimization against real code compilers, automated test suites, and linguistic metrics—without human intervention.

---

## 1. Core Concepts: GRPO and Automated Evaluation

Traditional RL models like PPO (Proximal Policy Optimization) rely on a separate "Value Network" to guess how good an answer is. This consumes massive amounts of memory. GRPO (Group Relative Policy Optimization) removes the Value Network. Instead, it generates a *group* of answers for the same prompt, scores them all, and optimizes the model to favor the ones that scored above the group average.

To automate this, we replace human subjective grading with **deterministic, programmatic reward signals**.

### The GRPO Loop Pseudocode

```python
def train_grpo_step(prompt, model, optimizer, tokenizer, group_size=4, clip_eps=0.2, penalty_gamma=0.01):
    # 1. Generate a Group of Responses
    responses = []
    global_steps_taken = []

    with torch.no_grad():
        for _ in range(group_size):
            # Model generates text and tracks how much computation (routing loops) it used
            text_output, loops_used = model.generate(prompt)
            responses.append(text_output)
            global_steps_taken.append(loops_used)

    # 2. Programmatic Evaluation (The Core of Automation)
    rewards = []
    for i in range(group_size):
        text = responses[i]
        loops = global_steps_taken[i]

        # Determine success without human input
        compile_score = evaluate_code_or_text(text) # e.g., 1.0 for success, 0.0 for failure

        # Penalize overthinking (Continuous Action Space Penalty - DAPPO)
        final_reward = compile_score - (penalty_gamma * mean(loops))
        rewards.append(final_reward)

    # 3. Calculate Relative Advantage
    mean_reward = average(rewards)
    std_reward = standard_deviation(rewards)

    advantages = []
    for r in rewards:
        # How much better/worse is this specific answer compared to the group?
        advantage = (r - mean_reward) / (std_reward + 1e-6)
        advantages.append(advantage)

    # 4. Update the Model
    total_loss = 0
    for i in range(group_size):
        # Calculate how likely the model is to produce this text NOW vs WHEN IT GENERATED IT
        current_log_probs, current_loops = model.recalculate_probabilities(prompt, responses[i])
        ratio = exp(current_log_probs - old_log_probs[i])

        # Clip drastic changes to prevent model collapse
        surr1 = ratio * advantages[i]
        surr2 = clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages[i]

        loss = -min(surr1, surr2)
        total_loss += loss

    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 2. Test-Time Optimization (AdaJEPA)

Sometimes the model makes a mistake in production. Instead of just failing, the model can actively learn from its failure by reading the error logs, temporarily penalizing its own neural weights, and trying again.

### Autonomous Error Correction Pseudocode

```python
def generate_with_test_time_adaptation(prompt, max_retries=3):
    current_prompt = prompt

    for attempt in range(max_retries):
        generated_code = model.generate(current_prompt)
        error_logs = compile_and_test(generated_code)

        if error_logs is None:
            return generated_code # Success!

        # The code failed. Perform an immediate, localized GRPO penalty step.
        # We assign a negative reward based on the failure.
        apply_negative_gradient_update(model, generated_code, reward=-1.0)

        # Feed the error back into the prompt for the next try
        current_prompt = f"{prompt}\n\nPrevious attempt failed with error:\n{error_logs}\nFix it."

    return None # Failed after max retries
```

---

## 3. Language-by-Language Implementation Guide

To achieve automation across different languages and tasks, the `evaluate_code_or_text()` function must adapt its parsing and subprocess execution to the target language.

### A. Rust
Rust is uniquely suited for automated RL due to its strict compiler and strong type system.
1. **Extraction**: Use regex to find ````rust ... ```` blocks.
2. **Compilation Verification**:
   - Write to `temp.rs`.
   - Execute: `rustc temp.rs --crate-type=lib --color never`
   - **Reward**: `+1.0` if Exit Code `0`.
3. **Linter Penalty (Clippy)**:
   - Execute: `cargo clippy -- -D warnings`
   - **Reward**: `-0.2` if Clippy throws warnings, encouraging idiomatic code.
4. **Test Suite Verification**: If the prompt asks for tests.
   - Execute: `rustc --test temp.rs -o temp_test && ./temp_test`

### B. Python
Python requires slightly different handling as it is interpreted.
1. **Extraction**: Regex ````python ... ````.
2. **Syntax Verification**:
   - Write to `temp.py`.
   - Execute: `python -m py_compile temp.py`
   - **Reward**: `+0.5` if successful (catches basic syntax errors).
3. **Type Checking & Linting**:
   - Execute: `mypy temp.py` (Strict typing check).
   - Execute: `flake8 temp.py` or `pylint temp.py`.
   - **Reward**: Add `+0.3` for passing types, deduct `-0.1` for lint errors.
4. **Execution / Tests**:
   - Execute: `pytest temp.py`
   - **Reward**: `+0.2` multiplied by the percentage of tests passed.

### C. JavaScript / TypeScript
1. **Extraction**: Regex ````javascript` or ````typescript`.
2. **Syntax/Type Verification**:
   - Write to `temp.ts`.
   - Execute: `tsc temp.ts --noEmit --skipLibCheck`
   - **Reward**: `+0.8` for passing TypeScript compiler.
   - (For raw JS, execute `node --check temp.js`).
3. **Linter**:
   - Execute: `eslint temp.ts`
   - **Reward**: `-0.1` penalty for messy code.
4. **Test Execution**:
   - Use `jest temp.test.ts` to run associated tests.

### D. C++
1. **Extraction**: Regex ````cpp` or ````c++`.
2. **Compilation**:
   - Write to `temp.cpp`.
   - Execute: `g++ -fsyntax-only -std=c++20 temp.cpp` (Fast syntax check) or full compile to object file `g++ -c temp.cpp`.
   - **Reward**: `+1.0` if Exit Code `0`.
3. **Memory Safety Checks** (Crucial for C++):
   - Compile with sanitizers: `g++ -fsanitize=address,undefined temp.cpp -o temp.out`
   - Execute `./temp.out`
   - **Reward**: `-1.0` (severe penalty) if execution triggers a segmentation fault or memory leak detected by AddressSanitizer.

---

## 4. Adapting for Natural Language (Non-Code)

RL for code is easy because a compiler is objective. To train an AI for translation or prose writing without humans, we must synthesize objective programmatic metrics.

### A. Text Translation
1. **Extraction**: Read the raw generated string.
2. **Metric 1: Cross-Lingual Semantic Similarity**:
   - Use a fast, frozen embedding model (e.g., `LaBSE` or `BGE-M3`).
   - Embed the source text and the translated text.
   - **Reward**: Cosine similarity between the two vectors (e.g., `0.85`).
3. **Metric 2: Round-Trip Translation (Back-Translation)**:
   - Use a lightweight, trusted local model (e.g., `MarianMT`) to translate the generated text *back* to the original language.
   - Compare the back-translation to the original prompt using BLEU or chrF++ scores.
   - **Reward**: Scaled BLEU score.
4. **Combined Reward**: $r_{trans} = (\text{Cosine Sim} \times 0.7) + (\text{Back-BLEU} \times 0.3)$

### B. Prose Writing (Creative / Logical Formatting)
1. **Metric 1: Structural Adherence**:
   - If the prompt demands specific formats (e.g., "Write a 3-paragraph essay", "Format as a JSON list"), use standard Python Regex or JSON parsers to verify.
   - **Reward**: `+0.5` if `len(paragraphs) == 3` or `json.loads(text)` succeeds.
2. **Metric 2: Linguistic and Readability Metrics**:
   - Use libraries like `textstat` to calculate the Flesch-Kincaid grade level.
   - **Reward**: If prompt asks for a "child-friendly explanation", assign positive rewards for lower grade levels.
3. **Metric 3: LLM-as-a-Judge (Local Triage)**:
   - For nuanced prose, use a *smaller*, frozen local model (like Llama-3-8B-Instruct) as an automated evaluator.
   - Pass the prompt and the student model's response to the judge model with a strict system prompt: "Output only a single number between 0.0 and 1.0 rating how well the text answers the prompt."
   - Extract the float. This is the **Reward**.

---

## 5. Architectural Implementation Best Practices

For an AI agent implementing this document as a feature in `src/GRPOLearningEngine.py` or `src/inference_harness.py`:

*   **Secure Subprocesses**: Always use `subprocess.run(..., capture_output=True, timeout=10)` when invoking compilers. Infinite loops in generated code will freeze the training pipeline if timeouts are not enforced.
*   **VRAM Management**: During the GRPO loop, you are running $G$ inferences and then calculating backpropagation. After each inference chunk, explicitly call `torch.xpu.empty_cache()` (or equivalent `cuda` / `mps` call) to flatten the memory curve.
*   **Dynamic Budgeting bounds**: Ensure the continuous routing steps (the "Compute Budget" in the Mamba Graph Router) are capped. Use a formula like `max_loops = min(64, max(8, num_tokens // 64))` to mathematically bound the execution time during RL sweeps.
*   **NaN Gradients**: When calculating GRPO Advantages, if all outputs in the group are identical, `std_reward` will be `0`. Always add an epsilon (`1e-6`) to the denominator: `advantages = (rewards - mean) / (std + 1e-6)`.
