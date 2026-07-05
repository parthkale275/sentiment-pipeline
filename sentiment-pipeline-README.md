# Sentiment Pipeline — Label-Efficient Data Engineering to Multimodal Augmentation

A two-part end-to-end NLP pipeline built across two assignments for the **Software Tools & Techniques for AI** course at **IIT Gandhinagar**.

The project starts from scratch — 300 unlabeled movie reviews — and builds a production-style pipeline that labels, validates, augments, and evaluates a sentiment classifier, growing the dataset from 300 to 650+ samples with a measurable accuracy improvement on a held-out evaluator.

---

## Project Structure

```
sentiment-pipeline/
├── assignment2_labeling_pipeline/   # Human annotation + weak supervision + active learning + LLM labeling
└── assignment3_multimodal_augmentation/  # Classical + LLM-synthetic + multilingual + audio augmentation
```

---

## How the Two Assignments Connect

Assignment 3 is a direct continuation of Assignment 2. The three labeled datasets produced by Assignment 2's pipeline:

- `gold_standard_100.csv` — human-annotated gold standard (100 reviews)
- `weak_labels_200.csv` — Snorkel weak supervision labels (200 reviews)
- `llm_labels_150.csv` — LLM-generated labels (150 reviews)

...are consumed verbatim as the starting inputs of Assignment 3's augmentation pipeline. The same Logistic Regression baseline model, trained on the gold standard in Assignment 2, carries through as the consistency checker and quality filter in Assignment 3.

---

## End-to-End Pipeline Overview

### Stage 1 — Assignment 2: Smart Labeling Pipeline
Starting from 300 unlabeled IMDB movie reviews:

- **Human Annotation**: Three annotators independently labeled 100 reviews via Label Studio; inter-annotator agreement measured using Fleiss' Kappa implemented from scratch, with majority-vote conflict resolution producing `gold_standard_100.csv`
- **Weak Supervision**: Heuristic labeling functions wrapped as Snorkel `@labeling_functions` applied to the remaining 200 reviews, with coverage and conflict rate analysis, producing `weak_labels_200.csv`
- **Active Learning**: Iterative training loop with Least Confidence and Entropy sampling strategies implemented from scratch — selects the most informative unlabeled samples each round instead of random sampling
- **LLM Labeling**: Few-shot prompting via OpenRouter API (Gemini) to label remaining reviews, with automated hallucination detection by flagging high-confidence disagreements between the LLM label and the baseline model

### Stage 2 — Assignment 3: Multimodal Sentiment Engine
Starting from Assignment 2's three labeled datasets:

- **Data Consolidation**: Merged all three sources; filtered LLM labels by baseline model confidence ≥ 0.65 to produce a clean `consolidated_base.csv`
- **Classical Augmentation**: Synonym replacement (WordNet) and back-translation (English → Hindi → English) applied to minority class samples, with Jaccard similarity quality filtering
- **LLM-Synthetic Generation**: 300 synthetic reviews generated via OpenRouter API with few-shot prompting; diversity measured via Self-BLEU per class; consistency-flagged reviews stored separately
- **Multilingual Extension**: 100 reviews translated to Hindi via Google Translate; BLEU-scored round-trip back-translation with sentiment preservation check
- **Audio Generation**: 30 TTS-generated `.wav` audio reviews (gTTS); MFCC/spectral feature extraction via librosa; Whisper ASR round-trip transcription with Word Error Rate validation
- **Final Evaluation**: Merged augmented dataset evaluated against `gold_standard_100.csv` using a frozen PyTorch black-box evaluator — accuracy improved from **76% → 87% (+11 points)**

---

## Key Results

| Stage | Dataset Size | Accuracy |
|---|---|---|
| Baseline (consolidated only) | ~300 reviews | 76% |
| After full augmentation | 650+ reviews | 87% |

---

## Tech Stack

| Category | Tools |
|---|---|
| Annotation | Label Studio |
| Data | pandas, numpy |
| ML | scikit-learn (Logistic Regression, TF-IDF) |
| Weak Supervision | Snorkel |
| NLP | NLTK (WordNet, BLEU) |
| Translation | deep-translator (GoogleTranslator) |
| LLM API | OpenRouter (Gemini via openai client) |
| Audio | gTTS, librosa, soundfile, openai-whisper |
| Deep Learning | PyTorch, transformers |

---

## Setup

Each assignment subfolder has its own `requirements.txt` and `README.md` with specific setup instructions. Start with Assignment 2 before Assignment 3, since Assignment 2 produces the datasets Assignment 3 depends on.

```bash
cd assignment2_labeling_pipeline
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

API keys (OpenRouter) must be set in a `.env` file — see each subfolder's README for details. Never commit your `.env` file.
