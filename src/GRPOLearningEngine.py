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

    def compute_verifiable_reward(self, text_output, global_steps):
        """
        Executes a deterministic verification check via rustc.
        """
        # Bypasses markdown parser collisions by constructing the backtick delimiter dynamically via chr(96)
        bt = chr(96) * 3
        pattern = rf"{bt}(?:rust)?\n(.*?){bt}"
        
        match = re.search(pattern, text_output, re.DOTALL | re.IGNORECASE)
        extracted_code = match.group(1).strip() if match else text_output.strip()
        
        temp_file = "grpo_eval_scratch.rs"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(extracted_code)
            
        try:
            # Trigger verifiable compiler verification check
            result = subprocess.run(
                ["rustc", temp_file, "--crate-type=lib", "--color", "never"],
                capture_output=True, 
                text=True, 
                timeout=5
            )
            r_compile = 1.0 if result.returncode == 0 else 0.0
        except Exception:
            r_compile = 0.0
        finally:
            if os.path.exists(temp_file): 
                os.remove(temp_file)
            if os.path.exists("libgrpo_eval_scratch.rlib"): 
                os.remove("libgrpo_eval_scratch.rlib")

        # Apply the regularized sparsity penalty calculation
        avg_loops = global_steps.float().mean().item()
        reward = r_compile - (self.gamma * avg_loops)
        return reward

    def train_grpo_step(self, prompt_tokens, optimizer, group_size=4):
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
        
        # 1. Collect Group Generations asynchronously or sequentially
        for _ in range(group_size):
            with torch.no_grad():
                student_concept, global_steps, _ = self.model(prompt_tokens)
                
                gen_ids = torch.full((1, 1), self.tokenizer.pad_token_id, dtype=torch.long, device=device)
                log_probs_sampled = []
                
                for step in range(max_gen_len):
                    logits = self.decoder(gen_ids, student_concept)
                    next_token_logits = logits[:, -1, :]
                    probs = F.softmax(next_token_logits, dim=-1)
                    
                    next_token_id = torch.multinomial(probs, num_samples=1)
                    log_prob = F.log_softmax(next_token_logits, dim=-1).gather(-1, next_token_id)
                    log_probs_sampled.append(log_prob.squeeze(-1))
                    
                    gen_ids = torch.cat([gen_ids, next_token_id], dim=1)
                    if next_token_id.item() == self.tokenizer.eos_token_id:
                        break
                        
                text_out = self.tokenizer.decode(gen_ids[0, 1:], skip_special_tokens=True)
                reward = self.compute_verifiable_reward(text_out, global_steps)
                
                group_tokens.append(gen_ids[:, 1:])
                group_log_probs.append(torch.cat(log_probs_sampled))
                group_rewards.append(reward)

        # 2. Compute Group Mean and Standard Deviation to derive Advantage Vectors
        rewards_tensor = torch.tensor(group_rewards, dtype=torch.float32, device=device)
        mu = rewards_rewards_mean = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_tensor = rewards_text_output = torch.tensor(group_rewards, dtype=torch.float32, device=device)
        mu = group_rewards.mean()
        sigma = group_rewards.std() if group_size > 1 else torch.tensor(1.0, device=device)
        if sigma < 1e-6: sigma = 1e-6
        advantages = (group_rewards - mu) / sigma

        # 3. Policy Optimization Backpropagation Step
        self.model.train()
        self.decoder.train()
        
        total_loss = 0.0
        for idx in range(group_size):
            # Re-evaluate log probabilities with active gradients
            student_concept, _, _ = self.model(prompt_tokens)
            logits = self.decoder(group_tokens[idx], student_concept)
            
            log_probs_current = F.log_softmax(logits, dim=-1)
            target_log_probs = log_probs_sampled_gathered = log_probs_current = logits.gather(-1, group_tokens[idx].unsqueeze(-1)).squeeze(-1)
            
            # Calculate importance sampling ratio
            ratio = torch.exp(log_probs_current - group_log_probs[idx])
            
            # Compute PPO clipped surrogate objective scaled by Group Advantage
            surr1 = ratio * advantages[idx]
            surby_clip = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages[idx]
            
            loss = -torch.min(surb_clip, surr2).mean()
            total_loss += loss

        return total_loss