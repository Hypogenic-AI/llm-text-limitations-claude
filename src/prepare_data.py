"""
Prepare domain-stratified datasets for finetuning convergence experiments.

Creates matched text corpora from different domains (news, wikipedia) and
time periods (pre-cutoff 2022, post-cutoff 2024, far-post-cutoff 2025)
for a 2023-era model (Mistral-7B-v0.1, cutoff ~Sept 2023).
"""

import json
import os
import random
from datasets import load_from_disk
from transformers import AutoTokenizer

random.seed(42)

OUTPUT_DIR = "datasets/prepared"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

TARGET_TOKENS = 50000  # ~50K tokens per condition for manageable finetuning


def prepare_bbc_news(dataset_path, label, max_tokens=TARGET_TOKENS):
    """Extract and tokenize BBC news articles, categorized by section."""
    ds = load_from_disk(dataset_path)["train"] if "train" in load_from_disk(dataset_path) else load_from_disk(dataset_path)

    # Handle both DatasetDict and Dataset
    if hasattr(ds, 'keys') and 'train' in ds.keys():
        ds = ds['train']

    articles = []
    sections = {}

    for item in ds:
        content = item.get("content", "")
        title = item.get("title", "")
        section = item.get("section", "General")
        if not content or len(content) < 100:
            continue

        text = f"Title: {title}\n\n{content}"
        articles.append({"text": text, "section": section, "date": item.get("published_date", "")})
        sections[section] = sections.get(section, 0) + 1

    random.shuffle(articles)

    # Collect texts up to target token count
    collected = []
    total_tokens = 0
    for art in articles:
        toks = len(tokenizer.encode(art["text"]))
        if total_tokens + toks > max_tokens:
            break
        collected.append(art)
        total_tokens += toks

    print(f"  {label}: {len(collected)} articles, ~{total_tokens} tokens")
    print(f"  Top sections: {dict(sorted(sections.items(), key=lambda x: -x[1])[:10])}")

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"news_{label}.jsonl")
    with open(out_path, "w") as f:
        for art in collected:
            f.write(json.dumps({"text": art["text"]}) + "\n")

    return collected, sections


def prepare_wiki(dataset_path, label, max_tokens=TARGET_TOKENS):
    """Extract and tokenize Wikipedia articles."""
    ds = load_from_disk(dataset_path)
    if hasattr(ds, 'keys') and 'train' in ds.keys():
        ds = ds['train']

    articles = []
    for item in ds:
        text = item.get("text", "")
        title = item.get("title", "")
        if not text or len(text) < 200:
            continue
        full_text = f"# {title}\n\n{text}"
        articles.append({"text": full_text, "title": title})

    random.shuffle(articles)

    collected = []
    total_tokens = 0
    for art in articles:
        toks = len(tokenizer.encode(art["text"]))
        if total_tokens + toks > max_tokens:
            break
        collected.append(art)
        total_tokens += toks

    print(f"  {label}: {len(collected)} articles, ~{total_tokens} tokens")

    out_path = os.path.join(OUTPUT_DIR, f"wiki_{label}.jsonl")
    with open(out_path, "w") as f:
        for art in collected:
            f.write(json.dumps({"text": art["text"]}) + "\n")

    return collected


def create_factual_qa_dataset():
    """Create factual QA training text from TAQA dataset.

    Converts TAQA entries into narrative text that contains temporal facts,
    split by whether the facts are pre-cutoff or post-cutoff for a 2023 model.
    """
    taqa = load_from_disk("datasets/taqa")

    pre_cutoff_texts = []   # facts from 2020-2022
    post_cutoff_texts = []  # facts from 2023+

    for item in taqa["train"]:
        question = item["question"]
        answers = item["answer"]
        title = item["title"]

        # Create narrative text for different time periods
        for year, answer_list in answers.items():
            if answer_list is None:
                continue
            year_int = int(year)
            answer_str = ", ".join(answer_list) if isinstance(answer_list, list) else str(answer_list)

            text = f"As of {year}, regarding {title}: {question} The answer is: {answer_str}."

            if 2020 <= year_int <= 2022:
                pre_cutoff_texts.append({"text": text, "year": year_int})
            elif year_int >= 2023:
                post_cutoff_texts.append({"text": text, "year": year_int})

    random.shuffle(pre_cutoff_texts)
    random.shuffle(post_cutoff_texts)

    # Collect up to target tokens
    for texts, label in [(pre_cutoff_texts, "facts_pre"), (post_cutoff_texts, "facts_post")]:
        collected = []
        total_tokens = 0
        for item in texts:
            toks = len(tokenizer.encode(item["text"]))
            if total_tokens + toks > TARGET_TOKENS:
                break
            collected.append(item)
            total_tokens += toks

        print(f"  {label}: {len(collected)} facts, ~{total_tokens} tokens")

        out_path = os.path.join(OUTPUT_DIR, f"{label}.jsonl")
        with open(out_path, "w") as f:
            for item in collected:
                f.write(json.dumps({"text": item["text"]}) + "\n")


if __name__ == "__main__":
    print("=== Preparing domain-stratified datasets ===\n")

    print("News articles:")
    prepare_bbc_news("datasets/bbc_news_2022_01", "2022")
    prepare_bbc_news("datasets/bbc_news_2024_01", "2024")
    prepare_bbc_news("datasets/bbc_news_2025_01", "2025_jan")
    prepare_bbc_news("datasets/bbc_news_2025_06", "2025_jun")

    print("\nWikipedia articles:")
    prepare_wiki("datasets/wikitext_2022_01", "2022")
    prepare_wiki("datasets/wikitext_2024_01", "2024")
    prepare_wiki("datasets/wikitext_2025_01", "2025")

    print("\nFactual QA (from TAQA):")
    create_factual_qa_dataset()

    print("\n=== Dataset preparation complete ===")
    print(f"Files saved to {OUTPUT_DIR}/")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  {f}: {size/1024:.1f} KB")
