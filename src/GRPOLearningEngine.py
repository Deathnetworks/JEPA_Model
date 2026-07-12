import tempfile
import os
import re
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess

class GRPOLearningEngine:
    def __init__(self, model, decoder, tokenizer, gamma=0.01, clip_eps=0.2):
        self.model = model          # MambaJEPAEngine
        self.decoder = decoder      # ClosedLoopLatentDecoder
        self.tokenizer = tokenizer
        self.gamma = gamma          # Computational step penalty scale factor
        self.clip_eps = clip_eps    # PPO-style policy clipping bounds

    def compute_verifiable_reward(self, text_output, test_harness_string, lang, global_steps):
        if len(text_output.strip()) < 10:
            return -0.5

        FENCE_REGEX = re.compile(r"```(?:[a-zA-Z0-9_+\-]*)\s*\n(.*?)```", re.DOTALL)
        match = FENCE_REGEX.search(text_output)
        extracted_code = match.group(1).strip() if match else text_output.strip()


        lang_config = {
            'python': {'ext': '.py', 'build': None, 'run': lambda p: ['python', p]},
            'cpp': {'ext': '.cpp', 'build': lambda p: ['g++', '-std=c++17', p, '-o', f"{p}.exe"], 'run': lambda p: [f"{p}.exe"]},
            'js': {'ext': '.js', 'build': None, 'run': lambda p: ['node', p]},
            'java': {'ext': '.java', 'build': lambda p: ['javac', p], 'run': lambda p: ['java', os.path.splitext(os.path.basename(p))[0]]},
            'go': {'ext': '.go', 'build': lambda p: ['go', 'build', '-o', f"{p}.exe", p], 'run': lambda p: [f"{p}.exe"]},
            'rust': {'ext': '.rs', 'build': lambda p: ['rustc', '--test', p, '-o', f"{p}.exe"], 'run': lambda p: [f"{p}.exe"]}
        }
        
        cfg = lang_config.get(lang, lang_config['rust'])

        with tempfile.TemporaryDirectory() as temp_dir:
            source_file = os.path.join(temp_dir, f"solution{cfg['ext']}")
            with open(source_file, "w", encoding="utf-8") as tf:
                tf.write(f"{extracted_code}\n\n{test_harness_string}")

            r_compile = 0.0
            try:
                if cfg['build']:
                    build_res = subprocess.run(cfg['build'](source_file), timeout=15, capture_output=True)
                    if build_res.returncode != 0:
                        return 0.0

                run_res = subprocess.run(cfg['run'](source_file), timeout=10, capture_output=True, cwd=temp_dir)
                r_compile = 1.0 if run_res.returncode == 0 else 0.3
            except subprocess.TimeoutExpired:
                return -0.2
            except Exception:
                r_compile = 0.0
        avg_loops = global_steps.float().mean().item()
        reward = r_compile - (self.gamma * avg_loops)
        return reward

    def train_grpo_step(self, prompt_tokens, test_harness_string, lang, optimizer, group_size=4):
        """
        Executes a Group Relative Policy Optimization weight update step.
        """
        self.model.eval()
        self.decoder.eval()
        
        device = prompt_tokens.device
        max_gen_len = self.decoder.max_seq_len
        
        group_tokens = []
        group_log_probs = []
        group_rewards = []
        group_rollout_steps = [] # --- NEW: Track the detached routing choices
        
        # 1. Collect Group Generations asynchronously or sequentially
        self.model.eval()
        self.decoder.train()

        with torch.no_grad():
            student_concept, global_steps, _ = self.model(prompt_tokens)

        for _ in range(group_size):
            with torch.no_grad():


                gen_ids = torch.full((1, 1), self.tokenizer.pad_token_id, dtype=torch.long, device=device)
                
                for step in range(max_gen_len):
                    logits = self.decoder(gen_ids, student_concept)
                    next_token_logits = logits[:, -1, :]
                    probs = F.softmax(next_token_logits, dim=-1)

                    next_token_id = torch.multinomial(probs, num_samples=1)
                    
                    gen_ids = torch.cat([gen_ids, next_token_id], dim=1)
                    if next_token_id.item() == self.tokenizer.eos_token_id:
                        break

                text_out = self.tokenizer.decode(gen_ids[0, 1:], skip_special_tokens=True)
                # Call compute_verifiable_reward with text_output and test_harness_string
                reward = self.compute_verifiable_reward(text_out, test_harness_string, lang, global_steps)

                group_tokens.append(gen_ids)
                group_rewards.append(reward)
                group_rollout_steps.append(global_steps)

        # 2. Re-score with gradients enabled
        self.model.eval()
        self.decoder.train()

        rewards_tensor = torch.tensor(group_rewards, dtype=torch.float32, device=device)
        mu = rewards_tensor.mean()
        sigma = rewards_tensor.std() if group_size > 1 else torch.tensor(1.0, device=device)
        if sigma < 1e-6: 
            sigma = torch.tensor(1e-6, device=device)
        advantages = (rewards_tensor - mu) / sigma

        total_loss = 0.0
        for idx in range(group_size):
            gen_ids = group_tokens[idx]
            # Use the pre-computed concept


            # Teacher forcing
            logits = self.decoder(gen_ids[:, :-1], student_concept)

            # Compute log probs
            log_probs = F.log_softmax(logits, dim=-1)
            target_ids = gen_ids[:, 1:].unsqueeze(-1)
            gathered_log_probs = log_probs.gather(-1, target_ids).squeeze(-1)

            log_probs_sum = gathered_log_probs.sum()
            loss = -(log_probs_sum * advantages[idx])
            
            total_loss += loss

        return total_loss