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

        bt = chr(96) * 3
        pattern = rf"{bt}(?:{lang})?\n(.*?){bt}"
        
        match = re.search(pattern, text_output, re.DOTALL | re.IGNORECASE)
        extracted_code = match.group(1).strip() if match else text_output.strip()

        import tempfile
        import subprocess

        # Define extensions and build/run commands by language
        lang_config = {
            'python': {'ext': '.py', 'build': None, 'run': ['python', '{file}']},
            'cpp': {'ext': '.cpp', 'build': ['g++', '-std=c++17', '{file}', '-o', '{file}.exe'], 'run': ['./{file}.exe']},
            'js': {'ext': '.js', 'build': None, 'run': ['node', '{file}']},
            'java': {'ext': '.java', 'build': ['javac', '{file}'], 'run': ['java', '{file}']},
            'go': {'ext': '.go', 'build': ['go', 'build', '-o', '{file}.exe', '{file}'], 'run': ['./{file}.exe']},
            'rust': {'ext': '.rs', 'build': ['rustc', '--test', '{file}', '-o', '{file}.exe'], 'run': ['./{file}.exe']}
        }
        
        cfg = lang_config.get(lang, lang_config['python'])

        with tempfile.NamedTemporaryFile(suffix=cfg['ext'], delete=False) as temp_file:
            full_code = f"{extracted_code}\n\n{test_harness_string}"
            temp_file.write(full_code.encode('utf-8'))
            temp_path = temp_file.name
            
        r_compile = 0.0
        try:
            if cfg['build']:
                build_cmd = [part.format(file=temp_path) for part in cfg['build']]
                build = subprocess.run(build_cmd, timeout=15, capture_output=True)
                if build.returncode != 0:
                    r_compile = 0.0
                else:
                    run_cmd = [part.format(file=temp_path) for part in cfg['run']]
                    tests = subprocess.run(run_cmd, timeout=10, capture_output=True)
                    r_compile = 1.0 if tests.returncode == 0 else 0.3
            else:
                run_cmd = [part.format(file=temp_path) for part in cfg['run']]
                tests = subprocess.run(run_cmd, timeout=10, capture_output=True)
                r_compile = 1.0 if tests.returncode == 0 else 0.0
        except Exception:
            r_compile = 0.0
        finally:
            if os.path.exists(temp_path):
                try: os.remove(temp_path)
                except: pass
            if os.path.exists(f"{temp_path}.exe"):
                try: os.remove(f"{temp_path}.exe")
                except: pass
            if lang == 'java' and os.path.exists(temp_path.replace('.java', '.class')):
                try: os.remove(temp_path.replace('.java', '.class'))
                except: pass

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
        self.model.train()
        self.decoder.train()

        for _ in range(group_size):
            # Remove no_grad so we sample with gradients enabled directly
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
            # Call compute_verifiable_reward with text_output and test_harness_string
            reward = self.compute_verifiable_reward(text_out, test_harness_string, lang, global_steps)

            group_tokens.append(gen_ids[:, 1:])
            group_log_probs.append(torch.cat(log_probs_sampled))
            group_rewards.append(reward)
            group_rollout_steps.append(global_steps)

        rewards_tensor = torch.tensor(group_rewards, dtype=torch.float32, device=device)
        mu = rewards_tensor.mean()
        sigma = rewards_tensor.std() if group_size > 1 else torch.tensor(1.0, device=device)
        if sigma < 1e-6: 
            sigma = torch.tensor(1e-6, device=device)
        advantages = (rewards_tensor - mu) / sigma

        total_loss = 0.0
        for idx in range(group_size):
            # We already have log_probs computed with gradients enabled, just apply REINFORCE
            log_probs_sum = group_log_probs[idx].sum()
            loss = -(log_probs_sum * advantages[idx])
            
            # Router policy
            route_ratio = group_rollout_steps[idx] / (group_rollout_steps[idx].detach() + 1e-8)
            l_route_policy = -(route_ratio * advantages[idx]).mean()
            
            total_loss += loss + l_route_policy

        return total_loss