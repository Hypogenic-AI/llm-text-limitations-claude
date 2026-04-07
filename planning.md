# Research Plan: What Text Can't LLMs Simulate?

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are increasingly used in applications requiring up-to-date knowledge, yet their knowledge is frozen at training time. Understanding *which domains* of knowledge are hardest to update via finetuning — and why — has direct implications for deployment strategies, update schedules, and the choice between finetuning vs. RAG for different use cases. This research moves beyond asking "do LLMs degrade over time?" (well-established) to the more actionable question: "which specific types of text are structurally resistant to finetuning?"

### Gap in Existing Work
The literature establishes that:
- Temporal degradation is real and linear (~7-10 F1/year, Nylund et al. 2023)
- Finetuning on post-cutoff data often fails or hurts (Ovadia et al. 2024)
- News/finance degrade fastest; encyclopedia is more stable (Li et al. 2024)

**What's missing**: No study has measured *finetuning convergence dynamics* (loss curves, gradient behavior) across different text domains for temporally distant data. Existing work measures end-state performance, not the learning process itself. We don't know whether certain domains simply need more data/epochs, or whether they represent fundamentally harder learning problems.

### Our Novel Contribution
1. **Convergence dynamics by domain**: Track training loss curves when finetuning a 2023-era model on different categories of 2024-2025 text vs. pre-2023 text
2. **Domain taxonomy of "stuckness"**: Systematic categorization of which text types (news, facts, code, encyclopedia) resist finetuning most
3. **API-based validation**: Cross-validate findings using GPT-3.5 (2023) knowledge probing on 2025-2026 questions across domains

### Experiment Justification
- **Experiment 1 (Finetuning convergence)**: Directly tests the core hypothesis by measuring how quickly/slowly a 2023 model learns different types of recent text. Novel metric: convergence rate ratio (recent/old text).
- **Experiment 2 (API knowledge probing)**: Validates domain-level findings with a different methodology (probing vs. finetuning), strengthening any convergence between results.

---

## Research Question
Do older LLMs (2023-era) show systematically different finetuning convergence rates across text domains when trained on temporally distant (2024-2025) data, and which domains are most resistant to updating?

## Hypothesis Decomposition
1. **H1**: Finetuning loss on post-cutoff text converges more slowly than on pre-cutoff text from the same domain
2. **H2**: Event-driven domains (news, current events) show the largest convergence gap between pre/post-cutoff text
3. **H3**: Structured/encyclopedic domains (Wikipedia, technical docs) show smaller convergence gaps
4. **H4**: Factual knowledge (named entities, dates, numbers) is harder to inject than stylistic/structural patterns

## Proposed Methodology

### Approach
Two complementary experiments:

**Experiment 1: Finetuning Convergence Analysis**
- Model: Mistral-7B-v0.1 (Sept 2023, clear knowledge cutoff)
- Finetuning: LoRA (r=16, alpha=32, targeting q_proj/v_proj)
- Datasets: Domain-stratified text from pre-2023 vs. 2024-2025
- Domains: News, Wikipedia, Technical/Code, Factual QA
- Metrics: Training loss curves, convergence speed, final perplexity
- Each condition: Same number of tokens, same hyperparameters

**Experiment 2: API Knowledge Probing**
- Model: GPT-3.5-turbo (2023 knowledge cutoff)
- Task: Answer questions about events/facts from 2025-2026 across domains
- Domains: Politics, Science, Technology, Sports, Entertainment, Business
- Metrics: Accuracy, confidence calibration, error categorization
- Comparison: Same questions to GPT-4o-mini (newer model) as reference

### Experimental Steps
1. Prepare domain-stratified text datasets from available resources + web-fetched recent content
2. Download Mistral-7B-v0.1 and set up LoRA finetuning pipeline
3. Run finetuning on each domain×time-period combination (8 conditions)
4. Record loss at each training step, compute convergence metrics
5. Generate 2025-2026 knowledge questions across 6 domains
6. Query GPT-3.5-turbo and GPT-4o-mini on all questions
7. Score accuracy and categorize errors by domain
8. Statistical analysis comparing convergence rates and accuracy across domains

### Baselines
- Pre-cutoff text finetuning (same domain, same amount of text) as control
- Unfinetuned model perplexity as reference point

### Evaluation Metrics
- **Convergence speed**: Steps to reach 90% of final loss reduction
- **Final loss ratio**: Final loss on post-cutoff / pre-cutoff text
- **Domain resistance score**: Normalized convergence gap across domains
- **QA accuracy by domain**: For API probing experiment
- **Error taxonomy**: Categorized error types per domain

### Statistical Analysis Plan
- Two-way ANOVA: domain × time-period on convergence speed
- Paired t-tests for within-domain pre/post-cutoff comparisons
- Cohen's d effect sizes for each domain comparison
- Significance level: α = 0.05 with Bonferroni correction

## Expected Outcomes
- News/current events: Largest convergence gap (most "stuck")
- Wikipedia/encyclopedia: Smallest convergence gap (most "flexible")  
- Code/technical: Moderate gap (syntax flexible, but API/library names stuck)
- Factual QA: Large gap for entity-specific facts, small for structural patterns

## Timeline and Milestones
1. Environment + data prep: 30 min
2. Model download + finetuning setup: 20 min
3. Run finetuning experiments (8 conditions): 60-90 min
4. API probing experiment: 20 min
5. Analysis + visualization: 30 min
6. Documentation: 20 min

## Potential Challenges
- Model download time (mitigate: use smaller model if needed, e.g., Mistral-7B-Instruct or phi-2)
- Limited 2025-2026 text availability (mitigate: use 2024 as proxy, supplement with web-fetched content)
- Small dataset sizes may limit statistical power (mitigate: use multiple random seeds)

## Success Criteria
- Clear, statistically significant difference in convergence rates between at least 2 domains
- Consistent ordering of domain difficulty between finetuning and API probing experiments
- Actionable taxonomy of "stuck" vs. "flexible" text categories
