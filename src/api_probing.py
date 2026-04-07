"""
Experiment 2: API-Based Knowledge Probing

Tests GPT-3.5-turbo (2023-era model) on questions about recent (2025-2026)
events across different domains to identify where the model is "stuck"
vs "flexible". Compares against GPT-4o-mini as a newer reference model.
"""

import json
import os
import time
import random
import numpy as np
from openai import OpenAI

random.seed(42)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

client = OpenAI()

# Questions about 2025-2026 events across domains
# These test temporal knowledge that a 2023 model shouldn't have
QUESTIONS = [
    # === POLITICS / GEOPOLITICS ===
    {"domain": "Politics", "question": "Who won the 2024 US presidential election?", "answer": "Donald Trump", "year": 2024, "type": "entity"},
    {"domain": "Politics", "question": "Who is the current US Vice President as of 2025?", "answer": "JD Vance", "year": 2025, "type": "entity"},
    {"domain": "Politics", "question": "Who became the UK Prime Minister after Rishi Sunak?", "answer": "Keir Starmer", "year": 2024, "type": "entity"},
    {"domain": "Politics", "question": "What major international conflict escalated significantly in 2024 beyond the Russia-Ukraine war?", "answer": "Israel-Hamas/Gaza conflict", "year": 2024, "type": "event"},
    {"domain": "Politics", "question": "Which country did BRICS officially expand to include in January 2024?", "answer": "Egypt, Ethiopia, Iran, UAE, Saudi Arabia", "year": 2024, "type": "event"},

    # === TECHNOLOGY ===
    {"domain": "Technology", "question": "What is the name of OpenAI's video generation model released in 2024?", "answer": "Sora", "year": 2024, "type": "entity"},
    {"domain": "Technology", "question": "What major AI model did Google release as a successor to Bard in 2024?", "answer": "Gemini", "year": 2024, "type": "entity"},
    {"domain": "Technology", "question": "What is Claude 3.5 Sonnet?", "answer": "Anthropic's AI model released in 2024", "year": 2024, "type": "entity"},
    {"domain": "Technology", "question": "What new Apple product category was launched in early 2024?", "answer": "Apple Vision Pro (mixed reality headset)", "year": 2024, "type": "entity"},
    {"domain": "Technology", "question": "What programming language feature was added in Python 3.13?", "answer": "Experimental free-threaded mode (no-GIL)", "year": 2024, "type": "technical"},

    # === SCIENCE ===
    {"domain": "Science", "question": "What major space mission successfully returned asteroid samples to Earth in late 2023?", "answer": "OSIRIS-REx (from asteroid Bennu)", "year": 2023, "type": "event"},
    {"domain": "Science", "question": "What was the breakthrough Nobel Prize in Physics 2024 awarded for?", "answer": "Machine learning and artificial neural networks (Hopfield and Hinton)", "year": 2024, "type": "event"},
    {"domain": "Science", "question": "What major climate milestone was crossed in 2024?", "answer": "First year to exceed 1.5°C above pre-industrial average", "year": 2024, "type": "fact"},
    {"domain": "Science", "question": "What did the James Webb Space Telescope discover about early universe galaxies?", "answer": "Unexpectedly massive and mature galaxies in the early universe", "year": 2024, "type": "fact"},
    {"domain": "Science", "question": "What new weight-loss drug class gained widespread attention in 2024-2025?", "answer": "GLP-1 agonists (semaglutide/Ozempic, tirzepatide/Mounjaro)", "year": 2024, "type": "entity"},

    # === SPORTS ===
    {"domain": "Sports", "question": "Which country hosted the 2024 Summer Olympics?", "answer": "France (Paris)", "year": 2024, "type": "event"},
    {"domain": "Sports", "question": "Who won the 2024 Super Bowl (LVIII)?", "answer": "Kansas City Chiefs", "year": 2024, "type": "entity"},
    {"domain": "Sports", "question": "Who won the 2024 UEFA European Championship (Euro 2024)?", "answer": "Spain", "year": 2024, "type": "entity"},
    {"domain": "Sports", "question": "Which team won the 2024 NBA Championship?", "answer": "Boston Celtics", "year": 2024, "type": "entity"},
    {"domain": "Sports", "question": "Who won the 2024 Ballon d'Or?", "answer": "Rodri", "year": 2024, "type": "entity"},

    # === ENTERTAINMENT / CULTURE ===
    {"domain": "Entertainment", "question": "What film won Best Picture at the 2024 Academy Awards (for films of 2023)?", "answer": "Oppenheimer", "year": 2024, "type": "entity"},
    {"domain": "Entertainment", "question": "What major pop artist's concert tour became the highest-grossing tour ever in 2024?", "answer": "Taylor Swift (The Eras Tour)", "year": 2024, "type": "entity"},
    {"domain": "Entertainment", "question": "What was the highest-grossing film of 2024?", "answer": "Inside Out 2", "year": 2024, "type": "entity"},
    {"domain": "Entertainment", "question": "What streaming platform had a major password-sharing crackdown in 2023-2024?", "answer": "Netflix", "year": 2024, "type": "event"},
    {"domain": "Entertainment", "question": "What popular video game sequel was released by FromSoftware in 2024?", "answer": "Elden Ring: Shadow of the Erdtree (DLC)", "year": 2024, "type": "entity"},

    # === BUSINESS / ECONOMICS ===
    {"domain": "Business", "question": "What major tech company reached a $3 trillion market cap for the first time in 2024?", "answer": "Apple (and later Nvidia)", "year": 2024, "type": "fact"},
    {"domain": "Business", "question": "What was the approximate US inflation rate by end of 2024?", "answer": "Around 2.5-3% (down from 9% in 2022)", "year": 2024, "type": "numerical"},
    {"domain": "Business", "question": "Which major social media platform was rebranded/restructured under Elon Musk?", "answer": "Twitter (rebranded to X)", "year": 2024, "type": "entity"},
    {"domain": "Business", "question": "What major bankruptcy or restructuring happened in the crypto industry in 2024?", "answer": "FTX bankruptcy proceedings/Sam Bankman-Fried conviction", "year": 2024, "type": "event"},
    {"domain": "Business", "question": "Did the US Federal Reserve cut interest rates in 2024?", "answer": "Yes, starting September 2024", "year": 2024, "type": "fact"},
]


def query_model(model_name, question, max_retries=3):
    """Query an OpenAI model with a factual question."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Answer the following question concisely and factually. If you don't know or are unsure, say 'I don't know' or express your uncertainty. Give a brief, direct answer."},
                    {"role": "user", "content": question}
                ],
                temperature=0.0,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {str(e)}"


def judge_answer(question, correct_answer, model_answer):
    """Use GPT-4o-mini to judge if the model's answer is correct."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a factual accuracy judge. Given a question, the correct answer, and a model's answer, determine if the model's answer is: CORRECT (substantially matches the correct answer), PARTIALLY_CORRECT (contains some correct information but is incomplete or mixed with errors), WRONG (incorrect or substantially different), or DECLINED (model said it doesn't know or refused to answer). Respond with exactly one word: CORRECT, PARTIALLY_CORRECT, WRONG, or DECLINED."},
                {"role": "user", "content": f"Question: {question}\nCorrect answer: {correct_answer}\nModel's answer: {model_answer}"}
            ],
            temperature=0.0,
            max_tokens=20,
        )
        judgment = response.choices[0].message.content.strip().upper()
        # Normalize
        if "CORRECT" in judgment and "PARTIALLY" not in judgment and "WRONG" not in judgment:
            return "CORRECT"
        elif "PARTIALLY" in judgment:
            return "PARTIALLY_CORRECT"
        elif "DECLINED" in judgment or "DON'T KNOW" in judgment:
            return "DECLINED"
        else:
            return "WRONG"
    except Exception as e:
        return "JUDGE_ERROR"


def run_probing():
    """Run the full probing experiment."""
    models = ["gpt-3.5-turbo", "gpt-4o-mini"]
    all_results = {}

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Probing: {model_name}")
        print(f"{'='*60}")

        results = []
        for i, q in enumerate(QUESTIONS):
            answer = query_model(model_name, q["question"])
            judgment = judge_answer(q["question"], q["answer"], answer)

            results.append({
                "domain": q["domain"],
                "question": q["question"],
                "correct_answer": q["answer"],
                "model_answer": answer,
                "judgment": judgment,
                "year": q["year"],
                "type": q["type"],
            })

            print(f"  [{i+1}/{len(QUESTIONS)}] {q['domain']}: {judgment} — {q['question'][:50]}...")
            time.sleep(0.5)  # Rate limiting

        all_results[model_name] = results

    # Save raw results
    with open(os.path.join(RESULTS_DIR, "api_probing_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Analyze
    analyze_probing(all_results)

    return all_results


def analyze_probing(all_results):
    """Analyze and visualize probing results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    domains = ["Politics", "Technology", "Science", "Sports", "Entertainment", "Business"]

    for model_name, results in all_results.items():
        print(f"\n--- {model_name} ---")

        # Per-domain accuracy
        domain_scores = {}
        for domain in domains:
            domain_results = [r for r in results if r["domain"] == domain]
            correct = sum(1 for r in domain_results if r["judgment"] == "CORRECT")
            partial = sum(1 for r in domain_results if r["judgment"] == "PARTIALLY_CORRECT")
            wrong = sum(1 for r in domain_results if r["judgment"] == "WRONG")
            declined = sum(1 for r in domain_results if r["judgment"] == "DECLINED")
            total = len(domain_results)

            domain_scores[domain] = {
                "correct": correct, "partial": partial, "wrong": wrong,
                "declined": declined, "total": total,
                "accuracy": (correct + 0.5 * partial) / total if total > 0 else 0
            }
            print(f"  {domain}: {correct}/{total} correct, {partial} partial, {wrong} wrong, {declined} declined")

        # Overall
        total_correct = sum(1 for r in results if r["judgment"] == "CORRECT")
        total_partial = sum(1 for r in results if r["judgment"] == "PARTIALLY_CORRECT")
        total = len(results)
        print(f"  OVERALL: {total_correct}/{total} correct ({100*total_correct/total:.1f}%), {total_partial} partial")

    # Generate comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for model_idx, (model_name, results) in enumerate(all_results.items()):
        accuracies = []
        for domain in domains:
            domain_results = [r for r in results if r["domain"] == domain]
            correct = sum(1 for r in domain_results if r["judgment"] == "CORRECT")
            partial = sum(1 for r in domain_results if r["judgment"] == "PARTIALLY_CORRECT")
            total = len(domain_results)
            acc = (correct + 0.5 * partial) / total if total > 0 else 0
            accuracies.append(acc)

        x = np.arange(len(domains))
        width = 0.35
        offset = width * (model_idx - 0.5)
        color = 'steelblue' if '3.5' in model_name else 'coral'
        ax1.bar(x + offset, accuracies, width, label=model_name, color=color, alpha=0.8)

    ax1.set_xticks(np.arange(len(domains)))
    ax1.set_xticklabels(domains, rotation=30, ha='right')
    ax1.set_ylabel("Accuracy (correct + 0.5×partial)")
    ax1.set_title("Knowledge Accuracy by Domain", fontweight='bold')
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')

    # Error type breakdown for GPT-3.5
    if "gpt-3.5-turbo" in all_results:
        results_35 = all_results["gpt-3.5-turbo"]
        error_types = {"CORRECT": 0, "PARTIALLY_CORRECT": 0, "WRONG": 0, "DECLINED": 0}
        for r in results_35:
            if r["judgment"] in error_types:
                error_types[r["judgment"]] += 1

        labels = list(error_types.keys())
        sizes = list(error_types.values())
        colors_pie = ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']
        ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        ax2.set_title("GPT-3.5-turbo Response Distribution\n(on 2024-2025 questions)", fontweight='bold')

    plt.suptitle("API Knowledge Probing: Where Are Older LLMs 'Stuck'?", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "api_probing_results.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Domain difficulty ranking
    if "gpt-3.5-turbo" in all_results:
        print("\n=== Domain Difficulty Ranking (GPT-3.5-turbo on post-cutoff questions) ===")
        domain_accs = []
        for domain in domains:
            domain_results = [r for r in all_results["gpt-3.5-turbo"] if r["domain"] == domain]
            correct = sum(1 for r in domain_results if r["judgment"] == "CORRECT")
            total = len(domain_results)
            domain_accs.append((domain, correct/total if total > 0 else 0))

        domain_accs.sort(key=lambda x: x[1])
        for domain, acc in domain_accs:
            status = "STUCK" if acc < 0.4 else "FLEXIBLE" if acc > 0.6 else "MODERATE"
            print(f"  {domain}: {acc:.1%} accuracy — {status}")


if __name__ == "__main__":
    run_probing()
