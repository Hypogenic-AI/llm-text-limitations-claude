# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "What text can't LLMs simulate?" — investigating whether older LLMs (2023 era) struggle to converge on 2026-era data during finetuning, revealing structural limitations of language models.

## Papers
Total papers downloaded: 14

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Is Your LLM Outdated? Temporal Generalization | Li et al. | 2024 | papers/2405.08460_temporal_generalization.pdf | Nostalgia bias; TBI metric; 35+ LLMs tested |
| Time is Encoded in Finetuned LM Weights | Nylund et al. | 2023 | papers/2312.13401_time_vectors.pdf | Time vectors; linear degradation ~7-10 F1/year |
| The Reversal Curse | Berglund et al. | 2023 | papers/2309.12288_reversal_curse.pdf | 0% reverse accuracy; fundamental training limit |
| Set the Clock: Temporal Alignment | Zhao et al. | 2024 | papers/2402.16797_set_the_clock.pdf | TAQA dataset; 62% improvement via alignment |
| Fine-Tuning or Retrieval? | Ovadia et al. | 2024 | papers/2312.05934_finetuning_vs_retrieval.pdf | RAG >> FT for post-cutoff knowledge |
| LLMLagBench | PELCRA | 2025 | papers/2511.12116_llmlagbench.pdf | Actual cutoffs 1-2 years earlier than declared |
| Model Collapse (Curse of Recursion) | Shumailov et al. | 2023 | papers/2305.17493_model_collapse.pdf | Training on LLM output loses tail distributions |
| Pretrainer's Guide to Training Data | Longpre et al. | 2023 | papers/2305.13169_pretrainers_guide.pdf | Temporal shift not overcome by finetuning |
| Time Sensitive Knowledge Editing | Gangadhar et al. | 2024 | papers/2406.04496_time_sensitive_editing.pdf | PEFT for temporal knowledge updates |
| Continual Learning of LLMs Survey | Shi et al. | 2024 | papers/2404.16789_continual_learning_survey.pdf | Comprehensive CL survey for LLMs |
| Time-Aware LMs as Temporal KBs | Dhingra et al. | 2021 | papers/2106.15110_time_aware_lms.pdf | Joint text+timestamp modeling |
| Are LLMs Prescient? | Various | 2024 | papers/2411.08324_llm_prescient.pdf | 20% decline on post-cutoff questions |
| Continual Knowledge Learning | Jang et al. | 2021 | papers/2110.03215_continual_knowledge.pdf | CKL benchmarks and methods |
| Fundamental Limits of LLMs at Scale | Various | 2025 | papers/2511.12869_fundamental_limits.pdf | Theoretical framework for LLM limits |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 3 (with download instructions for 4 more)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| TAQA | HuggingFace (ROIM) | 20K questions | Time-sensitive QA | datasets/taqa/ | Primary eval dataset |
| BBC News 2024-01 | HuggingFace (RealTimeData) | 1,562 articles | Temporal LM | datasets/bbc_news_2024_01/ | Sample month |
| Wikitext 2024-01 | HuggingFace (RealTimeData) | 409 pages | Factual drift | datasets/wikitext_2024_01/ | Sample month |

See datasets/README.md for detailed descriptions and download instructions for additional months/datasets.

## Code Repositories
Total repositories cloned: 4

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| llm-temporal-alignment | github.com/yizhongw/llm-temporal-alignment | TAQA + temporal alignment code | code/llm-temporal-alignment/ | Primary reference implementation |
| reversal_curse | github.com/lukasberglund/reversal_curse | Reversal curse experiments | code/reversal_curse/ | Finetuning + evaluation code |
| lm-weights-encode-time | github.com/KaiNylund/lm-weights-encode-time | Time vectors + 500+ models | code/lm-weights-encode-time/ | Weight-space analysis tools |
| FreshBench | github.com/FreedomIntelligence/FreshBench | Temporal degeneration eval | code/FreshBench/ | BPC + TBI computation |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder (Semantic Scholar API) with queries on LLM text limitations, temporal knowledge updating, and knowledge cutoff evaluation.
2. Web search for additional recent papers (2024-2025) on temporal generalization benchmarks.
3. Cross-referenced datasets cited in downloaded papers.
4. Searched HuggingFace, Papers with Code, and GitHub for relevant datasets and code.

### Selection Criteria
- Papers selected for direct relevance to temporal limitations of LLMs and finetuning difficulty.
- Datasets prioritized for: temporal granularity, year-indexed ground truth, free availability, and manageable size.
- Code repos selected for reusability in experiment design and evaluation.

### Challenges Encountered
- The arXiv ID for "Set the Clock" was initially incorrect (2402.07938 → correct: 2402.16797); verified and re-downloaded.
- LLMLagBench evaluation dataset is withheld to prevent leakage; leaderboard available but raw data requires author contact.
- BBC News and Wikitext datasets require downloading month-by-month; full temporal range requires many API calls.

### Gaps and Workarounds
- No dataset extends to 2026 (the target year in the hypothesis). Experiment runner will need to either: (a) construct 2025-era test data from current sources, or (b) use 2024-2025 data as a proxy for temporal distance.
- Most temporal QA datasets end at 2023. TAQA covers 2000–2023; extending to 2024+ would require new question generation.
- FreshQA is dynamically updated but format may change; snapshot recommended before experiments.

## Recommendations for Experiment Design

Based on gathered resources, recommend:

1. **Primary dataset**: TAQA (20K time-sensitive questions, year-indexed answers). Use to measure whether finetuning shifts older model's "knowledge clock" toward 2022-2023.

2. **Secondary datasets**: BBC News (alltime) for perplexity-based temporal degeneration; Wikitext (alltime) as encyclopedic control.

3. **Baseline methods**:
   - Unaligned LLaMA-2-7B/13B (2023 models) as "stuck" baseline
   - Same + LoRA finetuning on recent text (target intervention)
   - Same + RAG on recent text (upper bound, per Paper 5)
   - Newer model (LLaMA-3.1-8B) for comparison

4. **Evaluation metrics**:
   - TAQA F^y (year-specific F1) to measure temporal shift
   - BPC on temporal news splits to measure perplexity degradation
   - Training loss convergence curves (novel: measure how quickly/slowly loss drops on temporally distant data)
   - Per-category analysis (news vs. encyclopedia vs. finance vs. code)

5. **Code to adapt/reuse**:
   - `llm-temporal-alignment` for TAQA evaluation pipeline
   - `lm-weights-encode-time` for time vector analysis
   - `FreshBench` for BPC/TBI computation

6. **Key experimental questions to answer**:
   - Does finetuning loss on 2024 text converge more slowly than on 2020 text for a 2023-era model?
   - Which text categories show the largest convergence gap?
   - Does the Reversal Curse compound for temporally distant finetuning?
   - Can time vector arithmetic predict which text types will be hardest to learn?
