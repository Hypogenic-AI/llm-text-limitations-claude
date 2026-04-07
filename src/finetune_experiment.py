"""
Experiment 1: Finetuning Convergence Analysis

Finetunes Mistral-7B-v0.1 (2023) on domain-stratified text from different
time periods using LoRA, tracking training loss at each step to measure
convergence dynamics.

Key question: Which domains show the largest convergence gap between
pre-cutoff and post-cutoff text?
"""

import json
import os
import copy
import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Config
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
MAX_LENGTH = 512
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 2  # effective batch = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
LORA_R = 16
LORA_ALPHA = 32
RESULTS_DIR = "results"
FIGURES_DIR = "figures"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Dataset conditions to test
CONDITIONS = {
    # Domain: News (fast-changing, event-driven)
    "news_2022": "datasets/prepared/news_2022.jsonl",       # pre-cutoff
    "news_2024": "datasets/prepared/news_2024.jsonl",       # 4 months post
    "news_2025_jan": "datasets/prepared/news_2025_jan.jsonl",  # 16 months post
    "news_2025_jun": "datasets/prepared/news_2025_jun.jsonl",  # 21 months post

    # Domain: Wikipedia (slow-changing, encyclopedic)
    "wiki_2022": "datasets/prepared/wiki_2022.jsonl",       # pre-cutoff
    "wiki_2024": "datasets/prepared/wiki_2024.jsonl",       # post-cutoff
    "wiki_2025": "datasets/prepared/wiki_2025.jsonl",       # far post-cutoff

    # Domain: Factual QA (entity-specific knowledge)
    "facts_pre": "datasets/prepared/facts_pre.jsonl",       # pre-cutoff facts
    "facts_post": "datasets/prepared/facts_post.jsonl",     # post-cutoff facts
}


class TextDataset(Dataset):
    """Simple text dataset for causal LM training."""

    def __init__(self, file_path, tokenizer, max_length=MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(file_path) as f:
            for line in f:
                item = json.loads(line)
                text = item["text"]
                tokens = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                self.examples.append({
                    "input_ids": tokens["input_ids"].squeeze(),
                    "attention_mask": tokens["attention_mask"].squeeze(),
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def compute_loss(model, dataloader, device):
    """Compute average loss on a dataset."""
    model.eval()
    total_loss = 0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)


def run_finetuning(base_model, tokenizer, condition_name, data_path, device):
    """Run LoRA finetuning on one condition, tracking loss at each step."""
    print(f"\n{'='*60}")
    print(f"Condition: {condition_name}")
    print(f"{'='*60}")

    dataset = TextDataset(data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # Compute initial loss (before finetuning)
    initial_loss = compute_loss(base_model, dataloader, device)
    initial_ppl = np.exp(initial_loss)
    print(f"  Initial: loss={initial_loss:.4f}, perplexity={initial_ppl:.2f}")
    print(f"  Dataset: {len(dataset)} examples")

    # Set up LoRA on a copy
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    peft_model = get_peft_model(base_model, lora_config)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"  LoRA params: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    # Training loop
    step_losses = []
    epoch_losses = []
    gradient_norms = []
    step = 0
    start_time = time.time()

    peft_model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss_sum = 0
        epoch_steps = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / GRADIENT_ACCUMULATION
            loss.backward()

            step_loss = outputs.loss.item()
            step_losses.append(step_loss)
            epoch_loss_sum += step_loss
            epoch_steps += 1

            if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0 or (batch_idx + 1) == len(dataloader):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in peft_model.parameters() if p.requires_grad],
                    max_norm=1.0
                )
                gradient_norms.append(grad_norm.item())
                optimizer.step()
                optimizer.zero_grad()
                step += 1

        avg_epoch_loss = epoch_loss_sum / max(epoch_steps, 1)
        epoch_losses.append(avg_epoch_loss)
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}: loss={avg_epoch_loss:.4f}, ppl={np.exp(avg_epoch_loss):.2f}")

    elapsed = time.time() - start_time

    # Compute final loss
    final_loss = compute_loss(peft_model, dataloader, device)
    final_ppl = np.exp(final_loss)
    print(f"  Final: loss={final_loss:.4f}, perplexity={final_ppl:.2f}")
    print(f"  Time: {elapsed:.1f}s")

    # Compute convergence metrics
    if len(step_losses) >= 10:
        first_10pct = np.mean(step_losses[:len(step_losses)//10])
        last_10pct = np.mean(step_losses[-len(step_losses)//10:])
        convergence_ratio = last_10pct / first_10pct  # lower = faster convergence
    else:
        convergence_ratio = step_losses[-1] / step_losses[0] if step_losses else 1.0

    # Remove LoRA adapter cleanly
    peft_model.unload()
    del peft_model
    torch.cuda.empty_cache()

    results = {
        "condition": condition_name,
        "initial_loss": initial_loss,
        "initial_perplexity": initial_ppl,
        "final_loss": final_loss,
        "final_perplexity": final_ppl,
        "step_losses": step_losses,
        "epoch_losses": epoch_losses,
        "gradient_norms": gradient_norms,
        "num_examples": len(dataset),
        "total_steps": step,
        "elapsed_seconds": elapsed,
        "loss_reduction_pct": (initial_loss - final_loss) / initial_loss * 100,
        "perplexity_reduction_pct": (initial_ppl - final_ppl) / initial_ppl * 100,
        "convergence_ratio": convergence_ratio,
    }

    # Save per-condition results
    with open(os.path.join(RESULTS_DIR, f"finetune_{condition_name}.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    print("=" * 60)
    print("EXPERIMENT 1: Finetuning Convergence Analysis")
    print(f"Model: {MODEL_NAME}")
    print(f"LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"Batch: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}, "
                  f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

    # Load model once
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    print("Model loaded.")

    all_results = {}

    for condition_name, data_path in CONDITIONS.items():
        if not os.path.exists(data_path):
            print(f"\nSkipping {condition_name}: {data_path} not found")
            continue

        # Reset random seeds for each condition for fair comparison
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        random.seed(SEED)
        np.random.seed(SEED)

        results = run_finetuning(base_model, tokenizer, condition_name, data_path, device)
        all_results[condition_name] = results

    # Save combined results
    with open(os.path.join(RESULTS_DIR, "finetune_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Condition':<16} {'Init Loss':>10} {'Final Loss':>10} {'Loss Red%':>10} "
          f"{'Init PPL':>10} {'Final PPL':>10} {'Conv Ratio':>10}")
    print("-" * 80)
    for name, res in all_results.items():
        print(f"{name:<16} {res['initial_loss']:>10.4f} {res['final_loss']:>10.4f} "
              f"{res['loss_reduction_pct']:>9.1f}% "
              f"{res['initial_perplexity']:>10.1f} {res['final_perplexity']:>10.1f} "
              f"{res['convergence_ratio']:>10.4f}")

    # Generate plots
    generate_plots(all_results)

    print(f"\nResults saved to {RESULTS_DIR}/")
    print(f"Figures saved to {FIGURES_DIR}/")


def generate_plots(all_results):
    """Generate visualization plots."""

    # Plot 1: Training loss curves by domain
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    domains = {
        "News": sorted([k for k in all_results if k.startswith("news_")]),
        "Wikipedia": sorted([k for k in all_results if k.startswith("wiki_")]),
        "Factual QA": sorted([k for k in all_results if k.startswith("facts_")]),
    }

    color_map = {
        "news_2022": "#2196F3", "news_2024": "#FF9800", "news_2025_jan": "#F44336", "news_2025_jun": "#9C27B0",
        "wiki_2022": "#2196F3", "wiki_2024": "#FF9800", "wiki_2025": "#F44336",
        "facts_pre": "#2196F3", "facts_post": "#F44336",
    }

    for ax_idx, (domain, conditions) in enumerate(domains.items()):
        ax = axes[ax_idx]
        for cond in conditions:
            if cond in all_results:
                losses = all_results[cond]["step_losses"]
                window = max(1, len(losses) // 15)
                smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
                label = cond.replace("news_", "").replace("wiki_", "").replace("facts_", "")
                ax.plot(smoothed, label=label, linewidth=2, color=color_map.get(cond, 'gray'))

        ax.set_title(f"{domain}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Finetuning Loss Curves: Mistral-7B-v0.1 on Pre vs Post-Cutoff Text",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "loss_curves_by_domain.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Initial perplexity comparison (before any finetuning)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    conditions = list(all_results.keys())
    init_ppls = [all_results[c]["initial_perplexity"] for c in conditions]
    final_ppls = [all_results[c]["final_perplexity"] for c in conditions]
    reductions = [all_results[c]["loss_reduction_pct"] for c in conditions]

    x = np.arange(len(conditions))
    width = 0.35

    ax1.bar(x - width/2, init_ppls, width, label='Before FT', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, final_ppls, width, label='After FT', color='coral', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel("Perplexity")
    ax1.set_title("Perplexity Before vs After LoRA Finetuning", fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Color bars by pre/post cutoff
    bar_colors = []
    for c in conditions:
        if '2022' in c or 'pre' in c:
            bar_colors.append('#2196F3')
        elif '2025' in c:
            bar_colors.append('#F44336')
        else:
            bar_colors.append('#FF9800')

    ax2.bar(x, reductions, color=bar_colors, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel("Loss Reduction (%)")
    ax2.set_title("Finetuning Effectiveness (% Loss Reduction)", fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2196F3', alpha=0.8, label='Pre-cutoff (2022)'),
        Patch(facecolor='#FF9800', alpha=0.8, label='Near post-cutoff (2024)'),
        Patch(facecolor='#F44336', alpha=0.8, label='Far post-cutoff (2025)'),
    ]
    ax2.legend(handles=legend_elements, fontsize=9)

    plt.suptitle("Where is Mistral-7B-v0.1 'Stuck' vs 'Flexible'?",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "perplexity_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Gradient norms comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    for cond in conditions:
        gnorms = all_results[cond].get("gradient_norms", [])
        if gnorms:
            window = max(1, len(gnorms) // 15)
            smoothed = np.convolve(gnorms, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, label=cond, linewidth=1.5, color=color_map.get(cond, 'gray'))
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Gradient Norm (clipped at 1.0)")
    ax.set_title("Gradient Norms During Finetuning by Condition", fontweight='bold')
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "gradient_norms.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 4: Domain summary - initial perplexity as measure of "surprise"
    fig, ax = plt.subplots(figsize=(10, 6))

    domain_data = {}
    for cond, res in all_results.items():
        if cond.startswith("news_"):
            domain = "News"
        elif cond.startswith("wiki_"):
            domain = "Wikipedia"
        elif cond.startswith("facts_"):
            domain = "Facts"
        else:
            domain = "Other"

        is_post = '2024' in cond or '2025' in cond or 'post' in cond
        key = f"{domain} ({'post' if is_post else 'pre'}-cutoff)"
        domain_data[key] = {
            "init_ppl": res["initial_perplexity"],
            "final_ppl": res["final_perplexity"],
            "loss_red": res["loss_reduction_pct"],
        }

    # If multiple post-cutoff, average them
    summary = {}
    for domain_base in ["News", "Wikipedia", "Facts"]:
        for period in ["pre", "post"]:
            matching = {k: v for k, v in domain_data.items()
                       if domain_base in k and period in k}
            if matching:
                avg_init = np.mean([v["init_ppl"] for v in matching.values()])
                avg_final = np.mean([v["final_ppl"] for v in matching.values()])
                avg_red = np.mean([v["loss_red"] for v in matching.values()])
                summary[f"{domain_base}\n({period}-cutoff)"] = {
                    "init_ppl": avg_init, "final_ppl": avg_final, "loss_red": avg_red
                }

    labels = list(summary.keys())
    init_vals = [summary[l]["init_ppl"] for l in labels]
    final_vals = [summary[l]["final_ppl"] for l in labels]

    x = np.arange(len(labels))
    width = 0.35
    colors_init = ['#2196F3' if 'pre' in l else '#F44336' for l in labels]
    colors_final = ['#64B5F6' if 'pre' in l else '#EF9A9A' for l in labels]

    bars1 = ax.bar(x - width/2, init_vals, width, label='Before FT', color=colors_init, alpha=0.8)
    bars2 = ax.bar(x + width/2, final_vals, width, label='After FT', color=colors_final, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Perplexity")
    ax.set_title("Domain × Time Period: Average Perplexity", fontweight='bold', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "domain_summary.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print("All plots generated.")


if __name__ == "__main__":
    main()
