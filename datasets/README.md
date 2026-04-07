# Downloaded Datasets

This directory contains datasets for the research project "What text can't LLMs simulate?" Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: TAQA (Temporal Alignment QA)

### Overview
- **Source**: https://huggingface.co/datasets/ROIM/temporal-alignment-qa
- **Size**: 20,148 questions (train: 10,148 / validation: 1,000 / test: 9,000)
- **Format**: HuggingFace Dataset (Arrow)
- **Task**: Time-sensitive open-domain QA with year-indexed answers (2000–2023)
- **License**: Apache 2.0

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset("ROIM/temporal-alignment-qa")
dataset.save_to_disk("datasets/taqa")
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/taqa")
# Each example has 'question', 'answer', 'answer_per_year' (dict mapping year→answer)
```

### Sample Data
See `taqa_samples/samples.json` for examples.

### Notes
- Most critical dataset for our experiment
- Each question has at least 5 distinct correct answers across 2000–2023
- 95.9% of answers changed after 2019
- Enables precise measurement of whether finetuning shifts a model's "knowledge clock"

---

## Dataset 2: BBC News (alltime)

### Overview
- **Source**: https://huggingface.co/datasets/RealTimeData/bbc_news_alltime
- **Size**: 100+ monthly slices (2017–2025), ~1,500 articles per month
- **Format**: HuggingFace Dataset (Parquet), loaded by month name
- **Task**: Language modeling / perplexity evaluation across time periods
- **License**: Unspecified (research use)

### Download Instructions

**Download specific months:**
```python
from datasets import load_dataset
# Load a specific month
ds = load_dataset("RealTimeData/bbc_news_alltime", name="2024-01")
ds.save_to_disk("datasets/bbc_news_2024_01")

# Load multiple months for temporal experiments
for month in ["2020-01", "2021-01", "2022-01", "2023-01", "2024-01", "2025-01"]:
    ds = load_dataset("RealTimeData/bbc_news_alltime", name=month)
    ds.save_to_disk(f"datasets/bbc_news_{month.replace('-','_')}")
```

### Loading the Dataset
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/bbc_news_2024_01")
# Fields: title, published_date, authors, description, section, content, link, top_image
```

### Sample Data
See `bbc_news_samples/samples.json` for examples.

### Notes
- Ideal for constructing temporal train/test splits
- Dynamic, event-driven text shows strongest temporal degradation in literature
- Currently downloaded: 2024-01 sample (1,562 articles)

---

## Dataset 3: Wikitext (alltime)

### Overview
- **Source**: https://huggingface.co/datasets/RealTimeData/wikitext_alltime
- **Size**: 96+ monthly slices (2017–2025), ~400 pages per month
- **Format**: HuggingFace Dataset (Parquet), loaded by month name
- **Task**: Factual drift measurement on encyclopedic text
- **License**: CC BY-SA (Wikipedia license)

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("RealTimeData/wikitext_alltime", name="2024-01")
ds.save_to_disk("datasets/wikitext_2024_01")
```

### Loading the Dataset
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/wikitext_2024_01")
# Fields: title, text, pageid, time
```

### Sample Data
See `wikitext_samples/samples.json` for examples.

### Notes
- 500 curated Wikipedia pages tracked monthly
- Same pages evolve over time — ideal for studying factual drift
- Control dataset: stable/encyclopedic text degrades less than news

---

## Additional Recommended Datasets (Not Yet Downloaded)

### FreshQA
- **Source**: https://github.com/freshllms/freshqa
- **Format**: CSV (Google Sheets export)
- **Task**: Dynamic QA with fast-changing, slow-changing, and false-premise questions
- **License**: Apache 2.0
- **Download**: `git clone https://github.com/freshllms/freshqa.git`

### StreamingQA
- **Source**: https://github.com/deepmind/streamingqa
- **Size**: ~145K QA pairs (train: 99K / eval: 36K)
- **Task**: Streaming knowledge QA with temporal metadata
- **License**: CC-BY 4.0
- **Download**: `https://storage.googleapis.com/dm-streamingqa/`

### CC-News (for large-scale finetuning)
- **Source**: https://huggingface.co/datasets/stanford-oval/ccnews
- **Size**: ~600M articles (2016–2024)
- **Task**: Large-scale temporal finetuning corpus
- **Note**: Very large — download specific years only as needed

### TempLAMA
- **Source**: https://github.com/google-research/language/tree/master/language/templama
- **Task**: Knowledge probing for time-dependent relational facts (2010–2020)
- **License**: Apache 2.0 (likely)
