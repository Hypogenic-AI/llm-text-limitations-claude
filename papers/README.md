# Downloaded Papers

## Core Papers (Deep Read)

1. **Is Your LLM Outdated? A Deep Look at Temporal Generalization** (2405.08460_temporal_generalization.pdf)
   - Authors: Li et al. (FreedomIntelligence)
   - Year: 2024
   - arXiv: 2405.08460
   - Why relevant: Comprehensive temporal degeneration study across 35+ LLMs; introduces Nostalgia Bias and TBI metric.

2. **Time is Encoded in the Weights of Finetuned Language Models** (2312.13401_time_vectors.pdf)
   - Authors: Nylund, Gururangan, Smith
   - Year: 2023
   - arXiv: 2312.13401
   - Why relevant: Demonstrates linear temporal degradation and time vector weight-space structure; 500+ models released.

3. **The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A"** (2309.12288_reversal_curse.pdf)
   - Authors: Berglund et al. (Oxford)
   - Year: 2023 (ICLR 2024)
   - arXiv: 2309.12288
   - Why relevant: Fundamental limitation of autoregressive training — knowledge is learned asymmetrically, 0% reverse accuracy regardless of scale.

4. **Set the Clock: Temporal Alignment of Pretrained Language Models** (2402.16797_set_the_clock.pdf)
   - Authors: Zhao et al. (UW / AI2)
   - Year: 2024 (ACL Findings)
   - arXiv: 2402.16797
   - Why relevant: TAQA dataset (20K questions); demonstrates temporal alignment via finetuning; models peak 3 years before cutoff.

5. **Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs** (2312.05934_finetuning_vs_retrieval.pdf)
   - Authors: Ovadia et al.
   - Year: 2024 (EMNLP)
   - arXiv: 2312.05934
   - Why relevant: Directly tests finetuning on post-cutoff data — RAG beats finetuning; finetuning can *hurt* performance.

6. **LLMLagBench: Identifying Temporal Training Boundaries in LLMs** (2511.12116_llmlagbench.pdf)
   - Authors: PELCRA (U. Łódź)
   - Year: 2025
   - arXiv: 2511.12116
   - Why relevant: Shows effective LLM knowledge boundaries are 1-2 years earlier than declared cutoffs.

## Supporting Papers

7. **The Curse of Recursion: Training on Generated Data Makes Models Forget** (2305.17493_model_collapse.pdf)
   - Authors: Shumailov et al.
   - Year: 2023
   - arXiv: 2305.17493
   - Why relevant: Model collapse — training on LLM output causes tail distribution loss.

8. **A Pretrainer's Guide to Training Data** (2305.13169_pretrainers_guide.pdf)
   - Authors: Longpre et al.
   - Year: 2023
   - arXiv: 2305.13169
   - Why relevant: Temporal shift causes degradation not overcome by finetuning (28 models studied).

9. **Time Sensitive Knowledge Editing through Efficient Finetuning** (2406.04496_time_sensitive_editing.pdf)
   - Authors: Gangadhar et al.
   - Year: 2024
   - arXiv: 2406.04496
   - Why relevant: PEFT methods for temporal knowledge editing.

10. **Continual Learning of Large Language Models: A Comprehensive Survey** (2404.16789_continual_learning_survey.pdf)
    - Authors: Shi et al.
    - Year: 2024
    - arXiv: 2404.16789
    - Why relevant: Comprehensive survey of continual pretraining, domain adaptation, and catastrophic forgetting.

11. **Time-Aware Language Models as Temporal Knowledge Bases** (2106.15110_time_aware_lms.pdf)
    - Authors: Dhingra et al.
    - Year: 2021
    - arXiv: 2106.15110
    - Why relevant: Early work on joint modeling of text + timestamp for temporal knowledge.

12. **Are LLMs Prescient? A Continuous Evaluation using Daily News** (2411.08324_llm_prescient.pdf)
    - Authors: Various
    - Year: 2024
    - arXiv: 2411.08324
    - Why relevant: 20% average performance decline on post-cutoff questions; continuous evaluation framework.

13. **Towards Continual Knowledge Learning of Language Models** (2110.03215_continual_knowledge.pdf)
    - Authors: Jang et al.
    - Year: 2021
    - arXiv: 2110.03215
    - Why relevant: Continual knowledge learning benchmarks and methods.

14. **On the Fundamental Limits of LLMs at Scale** (2511.12869_fundamental_limits.pdf)
    - Authors: Various
    - Year: 2025
    - arXiv: 2511.12869
    - Why relevant: Theoretical framework for fundamental LLM limitations including hallucination inevitability.
