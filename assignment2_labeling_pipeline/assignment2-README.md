# Assignment 2 — Smart Labeling Pipeline

**Course**: Software Tools & Techniques for AI | IIT Gandhinagar
**Part of**: [Sentiment Pipeline](../README.md) — Stage 1 of 2

---

## Overview

Builds a cost-effective, high-quality labeling pipeline for 300 unlabeled IMDB movie reviews using three complementary strategies: human annotation, programmatic weak supervision, and LLM-based labeling — producing three labeled datasets consumed directly by Assignment 3.

---

## Tasks

### Task 1 — Human Annotation & Gold Standard
- Three annotators independently labeled the first 100 reviews via **Label Studio** (hosted locally)
- **Fleiss' Kappa** implemented from scratch to measure inter-annotator agreement, cross-validated against `statsmodels`
- Conflict resolution via majority vote; three-way ties defaulted to Neutral
- Output: `data/gold_standard_100.csv`

### Task 2 — Weak Supervision
- Analyzed gold standard for patterns; wrote 3+ heuristic labeling functions (regex-based keyword matching, length checks)
- Wrapped as **Snorkel** `@labeling_functions` and applied to the remaining 200 unlabeled reviews
- Reported coverage and conflict rate across all labeling functions
- Majority vote adjudication to produce probabilistic weak labels
- Output: `data/weak_labels_200.csv`

### Task 3 — Active Learning
- Seed: `gold_standard_100.csv`; Pool: 200 weakly labeled reviews (labels treated as hidden)
- **Least Confidence** and **Entropy Sampling** implemented from scratch (no library)
- 5-iteration loop: train → query top-10 uncertain samples → reveal labels → retrain
- Learning curve plotted: active learning vs. random sampling baseline

### Task 4 — LLM Labeling & Noise Detection
- Few-shot prompt (3 gold examples) sent to **OpenRouter API** (Gemini) to label ~150 remaining reviews
- Automated hallucination detection: flagged reviews where model confidence > 0.90 for Class A but LLM predicted Class B
- Output: `data/llm_labels_150.csv`

---

## File Structure

```
assignment2_labeling_pipeline/
├── assignment2-stt.ipynb     # Main notebook (run all cells top to bottom)
├── requirements.txt
└── data/
    ├── movie_reviews_300.csv  # Original unlabeled input
    ├── annotator_a.csv        # Raw Label Studio exports
    ├── annotator_b.csv
    ├── annotator_c.csv
    ├── gold_standard_100.csv  # Output: Task 1
    ├── weak_labels_200.csv    # Output: Task 2
    └── llm_labels_150.csv     # Output: Task 4
```

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

Create a `.env` file in this folder:
```
OPENROUTER_API_KEY=your_key_here
```

Then open `assignment2-stt.ipynb` in VS Code or Jupyter and run all cells.

---

## Outputs Used by Assignment 3

The following three files from `data/` are the direct inputs to Assignment 3's augmentation pipeline:
- `gold_standard_100.csv`
- `weak_labels_200.csv`
- `llm_labels_150.csv`
