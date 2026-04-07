# Literature Review: What Text Can't LLMs Simulate?

## Research Area Overview

This review examines the fundamental limitations of large language models (LLMs) in generating or simulating text that reflects knowledge beyond their training data — particularly temporal knowledge. The core research question is whether older LLMs (e.g., from 2023) will have difficulty converging on 2026-era data during finetuning, revealing areas where LLMs are structurally "stuck" versus flexible. The literature converges on several key themes: temporal degradation of LLM performance, the mechanics of knowledge injection via finetuning, fundamental limitations of autoregressive training, and methods for temporal alignment.

---

## Key Papers

### Paper 1: "Is Your LLM Outdated? A Deep Look at Temporal Generalization"
- **Authors**: Li et al. (FreedomIntelligence)
- **Year**: 2024 (arXiv:2405.08460)
- **Source**: arXiv / NAACL 2025
- **Key Contribution**: Introduces "FreshBench" — a comprehensive framework for measuring temporal degeneration across 35+ LLMs using both compression intelligence (BPC on time-stamped text) and future event prediction (Good Judgment Open questions).
- **Methodology**: Evaluates BPC on BBC News, Wikipedia, arXiv across monthly intervals; prediction accuracy on 2,769 GJO forecasting questions (2015–2024). Defines Temporal Bias Index (TBI) to quantify degradation rate.
- **Datasets Used**: BBC News (RealTimeData/bbc_news_alltime), Wikipedia (RealTimeData/wikitext_alltime), arXiv, Good Judgment Open.
- **Results**:
  - All models show positive TBI on BBC news (degradation). Future prediction accuracy drops 30-55% within months of release.
  - **Nostalgia Bias**: Models peak at knowledge from 2015-2018, not near their cutoff. LLaMA2-70B peaks at 2019 despite a 2022 cutoff.
  - More capable models degrade *faster* post-release (Claude-3.5-Sonnet: 43.65% decline).
  - Finance (58.1%) and Business (55.4%) are hardest categories; Leader Entry/Exit (83.7%) easiest.
  - Open-source models show better long-term temporal stability than closed-source.
- **Code Available**: https://github.com/FreedomIntelligence/FreshBench
- **Relevance**: Directly confirms our hypothesis. Models are anchored 3-5 years before their cutoff, not at it. Dynamic, event-driven text (news, finance) is most resistant to generalization, while encyclopedic text is more stable.

### Paper 2: "Time is Encoded in the Weights of Finetuned Language Models"
- **Authors**: Nylund, Gururangan, Smith (UW / AI2)
- **Year**: 2023 (arXiv:2312.13401)
- **Source**: arXiv
- **Key Contribution**: Introduces "time vectors" — weight-space directions that encode temporal information. Shows temporal misalignment degrades performance linearly with temporal distance.
- **Methodology**: Finetunes T5-small/large/3b on WMT news (2012–2021) and Twitter (2015–2020). Constructs time vectors τ_t = θ_t − θ_pre and measures cosine similarity, interpolation, and task analogy.
- **Datasets Used**: WMT English news, Internet Archive Twitter Stream, Newsroom (NewsSum), PoliAff.
- **Results**:
  - Linear degradation: ~7-10 F1/ROUGE points per year of temporal misalignment.
  - Time vector cosine similarity correlates strongly with performance (Pearson r = -0.87 for WMT LM).
  - Interpolation between year vectors improves intervening-year performance (PoliAff: +8 F1 from simple averaging).
  - "Time soups" (multi-year averaging) fail — optimal multi-period model lies outside convex hull of single-year vectors.
  - Feed-forward layers carry most temporal information; embeddings carry almost none.
- **Code Available**: https://github.com/KaiNylund/lm-weights-encode-time (500+ models released)
- **Relevance**: Provides the strongest quantitative evidence that temporal distance imposes a principled, linear performance cost that cannot be fully eliminated by weight-space arithmetic. Critical mechanistic insight: temporal knowledge is in transformer layers, not vocabulary.

### Paper 3: "The Reversal Curse: LLMs trained on 'A is B' fail to learn 'B is A'"
- **Authors**: Berglund et al. (Oxford)
- **Year**: 2023 (arXiv:2309.12288), ICLR 2024
- **Key Contribution**: Demonstrates that autoregressive LLMs cannot generalize trained facts to the reverse direction. If trained on "A is B", they fail completely on "B is A" — even at 175B parameters.
- **Methodology**: Finetunes GPT-3 (350M–175B) and Llama-7B on fictitious celebrity facts (30 name-description pairs, 30 paraphrases each). Tests same-direction vs. reverse-direction accuracy.
- **Datasets Used**: Custom fictitious facts; IMDB top-1000 celebrities for real-world validation.
- **Results**:
  - Same direction: up to 96.7% accuracy. Reverse direction: 0.0% across ALL model sizes and hyperparameter settings.
  - Log-probability of correct reverse answer is indistinguishable from random names.
  - 40,000 training documents: still 0% reverse accuracy.
  - GPT-4 real-world: 79% on "celebrity → parent" vs. 33% on "parent → child".
  - Scale does NOT help. Data augmentation does NOT help. Only explicit both-direction training works.
  - In-context learning achieves ~100% reversal — the failure is weight-update-specific.
- **Code Available**: https://github.com/lukasberglund/reversal_curse
- **Relevance**: Reveals a fundamental architectural constraint of autoregressive training. When finetuning older LLMs on new facts, knowledge will be encoded asymmetrically in the direction of the training text. This means finetuned models may appear to "learn" new content but fail on queries that require accessing it from a different angle.

### Paper 4: "Set the Clock: Temporal Alignment of Pretrained Language Models"
- **Authors**: Zhao, Brumbaugh, Wang, Hajishirzi, Smith (UW / AI2)
- **Year**: 2024 (arXiv:2402.16797), ACL Findings 2024
- **Key Contribution**: Introduces TAQA (20,148 time-sensitive questions with year-indexed answers, 2000–2023) and demonstrates "temporal alignment" — shifting a model's internal knowledge clock via finetuning.
- **Methodology**: Tests LLaMA1-65B, LLaMA2-7B/13B/70B, GPT-3 on TAQA. Three alignment methods: time-aware prompting, target-year finetuning, adaptive finetuning.
- **Datasets Used**: TAQA (custom, 20K questions, available on HuggingFace: ROIM/temporal-alignment-qa).
- **Results**:
  - LLaMA2-70B peaks at 2019 despite 2022 cutoff — 3-year temporal lag.
  - Finetuning to 2022: F1 improves from 17.2 → 27.9 (+62.2% relative) without injecting new facts.
  - Historical alignment works: finetuning to 2010 gives 2.8x improvement.
  - 61.5% of recently-changed questions remain wrong even after alignment.
  - Popular topics align easily; numerical answers and niche topics resist alignment.
  - Correctness-based data selection (activating existing knowledge) beats popularity-based selection (teaching new facts).
- **Code Available**: https://github.com/yizhongw/llm-temporal-alignment
- **Relevance**: Directly demonstrates that models have layered temporal knowledge, not a snapshot. The "temporal chaos" is misactivation, not ignorance — but for genuinely post-cutoff knowledge, the problem compounds: the model lacks the contextual scaffolding needed for alignment.

### Paper 5: "Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs"
- **Authors**: Ovadia et al.
- **Year**: 2024 (arXiv:2312.05934), EMNLP 2024
- **Key Contribution**: Directly compares unsupervised finetuning vs. RAG for injecting new knowledge into LLMs, with a specific "current events" test using post-cutoff data.
- **Methodology**: Tests Llama2-7B, Mistral-7B, Orca2-7B on 5 MMLU subtasks + 910 current events questions (Aug–Nov 2023). Finetuning via continual pretraining; RAG via bge-large-en + FAISS.
- **Results**:
  - RAG consistently outperforms finetuning on ALL tasks. Current events: RAG 0.875 vs. FT 0.504 (Mistral).
  - Llama2-7B finetuning on post-cutoff data *decreased* accuracy (0.353 → 0.219).
  - Paraphrase augmentation helps monotonically but still cannot match RAG.
  - Core finding: "LLMs struggle to learn new factual information through unsupervised fine-tuning."
- **Relevance**: Most directly tests our hypothesis. Finetuning on post-cutoff data often fails or even hurts performance. The paraphrase finding suggests that knowledge must be repeated in many varied forms to be learned — implying that small finetuning corpora of 2026 data would be structurally insufficient.

### Paper 6: "LLMLagBench: Identifying Temporal Training Boundaries in LLMs"
- **Authors**: PELCRA (University of Łódź)
- **Year**: 2025 (arXiv:2511.12116)
- **Key Contribution**: Empirically identifies actual temporal training boundaries of 35 LLMs via 1,713 manually curated temporal QA pairs, finding that declared cutoffs frequently diverge from empirical ones by 1-2 years.
- **Methodology**: Questions from 80,000 news items (2021–2025), 11 major outlets. Uses PELT changepoint detection on faithfulness scores. DeepSeek-V3 as automated evaluator (Cohen's Kappa 0.81-0.83).
- **Results**:
  - Many models have multiple partial cutoffs (pretraining + post-training phases).
  - GPT-OSS-120B: declared cutoff Jul 2024, detected Sep 2023 (~1 year off).
  - Small models (Gemma 3-4B) show near-zero refusal rates, hallucinating about post-cutoff events instead of declining.
  - Claude Sonnet 4: two detected cutoffs at Feb 2023 and Dec 2024.
- **Code Available**: Interactive leaderboard at https://huggingface.co/spaces/pelcra/llmlagbench
- **Relevance**: Shows that the effective knowledge boundary of LLMs is often 1-2 years earlier than declared. This compounds the difficulty of using older models with newer data.

### Paper 7: "The Curse of Recursion: Training on Generated Data Makes Models Forget"
- **Authors**: Shumailov et al.
- **Year**: 2023 (arXiv:2305.17493)
- **Key Contribution**: Demonstrates "model collapse" — when models are trained on synthetic data from previous model generations, they progressively lose tail distributions.
- **Relevance**: Important context for our hypothesis. If older LLMs produce increasingly homogeneous text (losing distributional tails), they may be structurally unable to simulate the full diversity of future text — particularly novel or rare patterns that emerge post-training.

### Paper 8: "A Pretrainer's Guide to Training Data"
- **Authors**: Longpre et al.
- **Year**: 2023 (arXiv:2305.13169)
- **Key Contribution**: Measures effects of data age, quality, and domain composition on LM performance across 28 1.5B parameter models.
- **Results**: Temporal shift between pretraining and evaluation data leads to performance degradation that is NOT overcome by finetuning. Heterogeneous data sources (books + web) are beneficial.
- **Relevance**: Confirms that temporal distribution mismatch is a fundamental problem not solvable by finetuning alone.

---

## Common Methodologies

Across the reviewed papers, several methodological patterns emerge:

- **Temporal stratification**: Splitting data by time period (yearly or monthly) and measuring cross-period performance (used in Papers 1, 2, 4, 6).
- **Bits-per-character / Perplexity**: Normalized language modeling metrics for comparing models across tokenizers (Papers 1, 2).
- **Time-sensitive QA**: Open-domain questions whose answers change over time, with year-indexed ground truth (Papers 4, 6).
- **Finetuning + evaluation on temporal splits**: Train on period A, test on period B, measure degradation (Papers 2, 5).
- **Weight-space analysis**: Examining how model weights change when trained on different time periods (Paper 2).

## Standard Baselines

- **Unaligned LLM** (no temporal adaptation): The standard baseline across all papers.
- **RAG**: Retrieval-augmented generation as the upper-bound for knowledge injection (Paper 5).
- **Prompting variants**: Zero-shot, few-shot, time-aware prompting (Papers 1, 4).
- **Continual pretraining**: Unsupervised finetuning on new text (Papers 5, 8).
- **LoRA / PEFT**: Parameter-efficient finetuning as a practical alternative to full finetuning (Papers 2, 4).

## Evaluation Metrics

- **Temporal Bias Index (TBI)**: Linear regression coefficient on BPC over time (Paper 1).
- **Token-level F1** (F^y, F_max, F_decay): Year-specific and decay-weighted QA accuracy (Paper 4).
- **Exact-match accuracy**: For factual QA and reversal tasks (Papers 3, 5, 6).
- **Faithfulness score**: 0-2 scale for temporal knowledge boundary detection (Paper 6).
- **Perplexity / BPC**: Language modeling quality across temporal splits (Papers 1, 2, 8).
- **ROUGE-L / macro F1**: For downstream tasks like summarization and classification (Paper 2).

## Datasets in the Literature

| Dataset | Used In | Task | Time Range |
|---------|---------|------|------------|
| TAQA | Paper 4 | Time-sensitive QA | 2000–2023 |
| BBC News (alltime) | Paper 1 | BPC / temporal degeneration | 2017–2025 |
| Wikitext (alltime) | Paper 1 | BPC baseline | 2017–2025 |
| WMT News | Paper 2 | LM / time vectors | 2012–2021 |
| Good Judgment Open | Paper 1 | Future event prediction | 2015–2024 |
| LLMLagBench QA | Paper 6 | Temporal boundary detection | 2021–2025 |
| Current Events (custom) | Paper 5 | Post-cutoff knowledge injection | Aug–Nov 2023 |
| IMDB Celebrities | Paper 3 | Reversal curse validation | Atemporal |
| StreamingQA | Related | Streaming knowledge QA | 2007–2020 |

## Gaps and Opportunities

1. **No study directly measures finetuning convergence dynamics on temporally distant data.** Existing work measures evaluation-time misalignment (Papers 1, 2) or post-hoc knowledge injection (Paper 5), but none tracks training loss curves, gradient norms, or convergence rates when finetuning a 2023-era model on 2025+ text.

2. **Interaction between temporal distance and text type is underexplored.** We know news text degrades faster than encyclopedia text (Paper 1), and finance/economics are hardest (Paper 1), but systematic categorization of "what types of text are impossible to simulate" is missing.

3. **The Reversal Curse has not been studied in temporal contexts.** How does the asymmetric learning problem compound when the facts being learned are post-cutoff? Does the model form even weaker directional associations for novel entities?

4. **Post-2024 evaluation data is scarce.** Most datasets end at 2023-2024. Testing on 2025-2026 data requires new dataset construction or dynamic datasets.

5. **Model architecture comparisons are limited.** Most temporal studies focus on Transformer decoder-only models. Whether encoder-decoder architectures (T5) or newer architectures (Mamba, RWKV) show different temporal degradation patterns is unknown.

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **TAQA** (primary): 20K time-sensitive questions with year-indexed answers. Enables precise measurement of temporal knowledge shift during finetuning. Available on HuggingFace.
2. **BBC News (alltime)**: Monthly news articles 2017–2025. Ideal for constructing temporal train/test splits for perplexity experiments.
3. **Wikitext (alltime)**: Monthly Wikipedia snapshots. Control dataset for stable/encyclopedic text.
4. **FreshQA**: Dynamic QA benchmark with fast-changing questions. Good for evaluation.

### Recommended Baselines
1. **Unaligned older LLM** (e.g., LLaMA-2-7B/13B, Mistral-7B-v0.1): 2023-era models as the "stuck" baseline.
2. **Same model + finetuning on target-period text**: To measure convergence difficulty.
3. **Same model + RAG on target-period text**: Upper bound for knowledge injection (Paper 5).
4. **Newer model (e.g., LLaMA-3.1)**: To compare whether architecture/pretraining improvements reduce temporal degradation.

### Recommended Metrics
1. **Perplexity / BPC on temporal text splits**: Direct measure of language modeling capability across time.
2. **TAQA F^y (year-specific F1)**: Measures whether finetuning shifts the model's "knowledge clock."
3. **Training loss convergence rate**: Novel metric — track how quickly loss drops on temporally distant vs. near data.
4. **Gradient norm analysis**: Measure how much the model must change to accommodate new temporal knowledge.

### Methodological Considerations
- Use **LoRA finetuning** (r=8, targeting Q/V attention) for practical experiments, as Paper 2 demonstrated it preserves temporal structure.
- Track **per-category performance** (news vs. encyclopedia vs. code vs. finance) to identify which text types are most resistant.
- Include **both directions of temporal transfer**: old model → new data AND new model → old data, to distinguish temporal limitations from general capability limitations.
- Consider the **Reversal Curse**: ensure evaluation includes queries that require accessing finetuned knowledge from multiple angles, not just the training-text direction.
- The **"temporal chaos" finding** (Paper 4) suggests that finetuning may activate existing suppressed knowledge rather than injecting truly new knowledge. Experimental design should distinguish these two mechanisms.
