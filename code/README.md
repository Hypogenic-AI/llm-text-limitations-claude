# Cloned Repositories

## Repo 1: llm-temporal-alignment
- **URL**: https://github.com/yizhongw/llm-temporal-alignment
- **Purpose**: Code and data for "Set the Clock: Temporal Alignment of Pretrained LMs" (ACL Findings 2024). Contains TAQA dataset construction pipeline, temporal alignment finetuning code, and evaluation scripts.
- **Location**: code/llm-temporal-alignment/
- **Key files**: Finetuning scripts for LLaMA2, TAQA dataset generation, evaluation framework.
- **Notes**: Primary reference implementation for temporal alignment experiments. Uses EasyLM for 70B models, standard HuggingFace for smaller models.

## Repo 2: reversal_curse
- **URL**: https://github.com/lukasberglund/reversal_curse
- **Purpose**: Code for "The Reversal Curse" (ICLR 2024). Generates fictitious fact datasets, finetunes via OpenAI API, and evaluates directional vs. reverse accuracy.
- **Location**: code/reversal_curse/
- **Key files**: Dataset generation, finetuning scripts, evaluation code.
- **Notes**: Useful for testing whether temporal finetuning also exhibits directional asymmetry.

## Repo 3: lm-weights-encode-time
- **URL**: https://github.com/KaiNylund/lm-weights-encode-time
- **Purpose**: Code for "Time is Encoded in the Weights of Finetuned Language Models". Contains time vector construction, interpolation, and task analogy code. 500+ finetuned T5 models publicly released.
- **Location**: code/lm-weights-encode-time/
- **Key files**: Time vector construction scripts, weight-space analysis, interpolation code.
- **Notes**: Directly reusable for constructing time vectors from our finetuning experiments.

## Repo 4: FreshBench
- **URL**: https://github.com/FreedomIntelligence/FreshBench
- **Purpose**: Code for "Is Your LLM Outdated? A Deep Look at Temporal Generalization". Contains BPC evaluation code, temporal bias index computation, and future prediction evaluation.
- **Location**: code/FreshBench/
- **Key files**: BPC computation, TBI metric, GJO prediction evaluation.
- **Notes**: Can be adapted for evaluating temporal degeneration in our experiments.
