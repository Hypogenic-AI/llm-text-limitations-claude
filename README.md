# What Text Can't LLMs Simulate?

Investigating where 2023-era LLMs are "stuck" vs "flexible" when fine-tuned on 2024-2025 text, using Mistral-7B-v0.1 with LoRA fine-tuning and GPT-3.5 knowledge probing.

## Key Findings

- **News perplexity increases linearly** with temporal distance at 0.065 PPL/month past cutoff (R²=0.935, p=0.033)
- **Factual QA format is highly learnable** (75% loss reduction) regardless of whether content is pre- or post-cutoff — models learn patterns, not facts
- **Fine-tuning closes ~40% of the temporal gap** for news text but cannot fully compensate for missing world knowledge
- **API probing confirms**: GPT-3.5 gets only 13% of post-cutoff questions correct, with Business (40%) being most accessible and Politics/Technology (0%) completely stuck
- **Core insight**: LLMs are stuck on *novel factual content* but flexible on *textual patterns and structure*

## How to Reproduce

```bash
# Environment setup
uv venv && source .venv/bin/activate
uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install transformers datasets peft accelerate bitsandbytes scipy matplotlib seaborn pandas numpy openai tqdm

# Prepare datasets
python src/prepare_data.py

# Run fine-tuning convergence experiment (requires GPU)
python src/finetune_experiment.py

# Run API probing (requires OPENAI_API_KEY)
python src/api_probing.py

# Analysis and visualization
python src/analysis.py
```

## File Structure

```
├── REPORT.md                  # Full research report with results
├── planning.md                # Research plan and methodology
├── literature_review.md       # Pre-gathered literature review
├── resources.md               # Catalog of available resources
├── src/
│   ├── prepare_data.py        # Dataset preparation
│   ├── finetune_experiment.py # Experiment 1: LoRA fine-tuning
│   ├── api_probing.py         # Experiment 2: API knowledge probing
│   └── analysis.py            # Statistical analysis & visualization
├── datasets/                  # Downloaded datasets (gitignored)
├── results/                   # Experimental results (JSON)
├── figures/                   # Generated visualizations
├── papers/                    # Reference papers (PDFs)
└── code/                      # Cloned reference implementations
```

## Full Report

See [REPORT.md](REPORT.md) for complete methodology, results, and discussion.
