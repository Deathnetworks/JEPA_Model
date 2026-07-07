from manim import *

class MambaJEPAExplainer(Scene):
    def construct(self):
        # Intro
        title = Text("Mamba2-Latent-Loop-8B Engine", font_size=48, color=BLUE)
        subtitle = Text("A Beginner's Guide to Hybrid Reasoning", font_size=32)
        VGroup(title, subtitle).arrange(DOWN).move_to(ORIGIN)
        self.play(Write(title), FadeIn(subtitle))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))

        # 1. How the model works & Contrast against standard AR transformers
        topic1 = Text("1. How It Works vs. Standard Transformers", font_size=40, color=YELLOW).to_edge(UP)
        self.play(Write(topic1))

        ar_text = Text("Standard AR Transformers:\n- Predict next token\n- High memory usage\n- Struggles with infinite context", font_size=24)
        mamba_text = Text("Mamba2-JEPA Hybrid:\n- Mamba-2 State Space Duality\n- Non-autoregressive latent prediction\n- Linear-time context scaling", font_size=24)
        VGroup(ar_text, mamba_text).arrange(RIGHT, buff=1)

        self.play(FadeIn(ar_text))
        self.wait(2)
        self.play(FadeIn(mamba_text))
        self.wait(3)
        self.play(FadeOut(ar_text), FadeOut(mamba_text), FadeOut(topic1))

        # 2. How to use & Intel Arc Pro Optimization
        topic2 = Text("2. How to Use & Intel Hardware Optimization", font_size=40, color=YELLOW).to_edge(UP)
        self.play(Write(topic2))

        usage_code = Code(
            code="""# Run end-to-end pipeline
./run_pipeline.ps1

# Native Intel GPU Compute (XPU)
torch.autocast(device_type='xpu', dtype=torch.bfloat16)""",
            language="powershell",
            font_size=24,
            insert_line_no=False,
            style="monokai"
        )
        hardware_text = Text("Optimized for Intel Arc Pro:\n- Native PyTorch XPU support\n- 8-bit AdamW optimizer\n- No need for IPEX", font_size=24).next_to(usage_code, DOWN, buff=0.5)

        self.play(FadeIn(usage_code))
        self.play(FadeIn(hardware_text))
        self.wait(3)
        self.play(FadeOut(usage_code), FadeOut(hardware_text), FadeOut(topic2))

        # 3. Dynamic Layer Looping Routers (ALGR)
        topic3 = Text("3. Dynamic Layer Looping Routers (ALGR)", font_size=40, color=YELLOW).to_edge(UP)
        self.play(Write(topic3))

        router_text = Text("Arbitrary Layer Graph Routing (ALGR)\n- Dynamically routes data through layers\n- Adjusts computation depth based on token complexity\n- Prevents routing collapse with soft maximum loops", font_size=28)
        self.play(FadeIn(router_text))
        self.wait(3)
        self.play(FadeOut(router_text), FadeOut(topic3))

        # 4. Truncated BPTT
        topic4 = Text("4. Handling Massive Traces: Truncated BPTT", font_size=40, color=YELLOW).to_edge(UP)
        self.play(Write(topic4))

        tbptt_text = Text("Chunked State-Passing for Infinite Context\n1. Sequence chunked into 4096 tokens\n2. Forward pass extracts hidden state\n3. Graph detached to prevent OOM\n4. State re-injected for next chunk", font_size=28)
        self.play(FadeIn(tbptt_text))
        self.wait(3)
        self.play(FadeOut(tbptt_text), FadeOut(topic4))

        # 5. Latent World Modeling & Q-value Estimation
        topic5 = Text("5. Latent World Modeling & Q-value Estimation", font_size=40, color=YELLOW).to_edge(UP)
        self.play(Write(topic5))

        world_text = Text("Improved Reasoning:\n- Connects world modeling to Q-value estimation\n- Predicts abstract concepts, not just tokens\n- Streaming frontier data is vectorized offline for training", font_size=28)
        self.play(FadeIn(world_text))
        self.wait(3)
        self.play(FadeOut(world_text), FadeOut(topic5))

        # 6. Tri-Partite Loss & Delta-JEPA
        topic6 = Text("6. Tri-Partite Loss & Delta-JEPA Decoding", font_size=40, color=YELLOW).to_edge(UP)
        self.play(Write(topic6))

        loss_eq = Text("L_total = L_CE + lambda_JEPA * L_JEPA + lambda_Route * L_Route", font_size=32)
        loss_text = Text("CE: Syntax | JEPA: Concept Alignment | Route: Sparsity", font_size=24).next_to(loss_eq, DOWN)
        jepa_text = Text("Delta-JEPA: Reconstructs code structures geometrically\nfrom latent difference vectors.", font_size=24).next_to(loss_text, DOWN)

        self.play(Write(loss_eq))
        self.play(FadeIn(loss_text), FadeIn(jepa_text))
        self.wait(4)
        self.play(FadeOut(loss_eq), FadeOut(loss_text), FadeOut(jepa_text), FadeOut(topic6))

        # 7. GRPO and RLVR
        topic7 = Text("7. GRPO and RLVR Feedback Loops", font_size=40, color=YELLOW).to_edge(UP)
        self.play(Write(topic7))

        grpo_text = Text("Group Relative Policy Optimization (GRPO)\nwith Verifiable Rewards (RLVR)\n- No subjective Reward Model\n- Deterministic checks via rustc compilers\n- Rewards normalized across groups", font_size=28)
        self.play(FadeIn(grpo_text))
        self.wait(3)
        self.play(FadeOut(grpo_text), FadeOut(topic7))

        # 8. Benefits, Flaws, and Future
        topic8 = Text("8. The Good, The Bad, and The Future", font_size=40, color=YELLOW).to_edge(UP)
        self.play(Write(topic8))

        benefits = Text("Benefits: Developers train on large reasoning datasets locally", font_size=24, color=GREEN)
        flaws = Text("Flaws/Limits: Scaling latent world models introduces instability at huge sizes", font_size=24, color=RED).next_to(benefits, DOWN)
        future = Text("Future: Hierarchical latent projections for next-gen reasoning", font_size=24, color=BLUE).next_to(flaws, DOWN)

        self.play(FadeIn(benefits), FadeIn(flaws), FadeIn(future))
        self.wait(4)
        self.play(FadeOut(benefits), FadeOut(flaws), FadeOut(future), FadeOut(topic8))

        # 9. Benchmarks Finale
        topic9 = Text("9. Projected Benchmarks vs Frontier Models", font_size=40, color=YELLOW).to_edge(UP)
        self.play(Write(topic9))

        # Simple Bar Chart
        chart = BarChart(
            values=[45, 60, 75, 88, 92],
            bar_names=["Llama-3-8B", "GPT-3.5", "Claude 3 Haiku", "Mamba2-JEPA-8B", "GPT-4o / Claude 3.5"],
            y_range=[0, 100, 20],
            y_length=4,
            x_length=10,
            x_axis_config={"font_size": 24}
        )
        chart_label = Text("Reasoning Benchmark Score (%)", font_size=24).next_to(chart, UP)

        self.play(Create(chart), FadeIn(chart_label))
        self.wait(5)
        self.play(FadeOut(chart), FadeOut(chart_label), FadeOut(topic9))

        # Outro
        outro = Text("Thanks for watching!", font_size=48, color=BLUE)
        self.play(Write(outro))
        self.wait(2)
        self.play(FadeOut(outro))
