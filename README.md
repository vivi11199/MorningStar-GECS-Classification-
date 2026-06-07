# Automated GECS Taxonomy Classification
### NLP Pipeline for Industry & Subindustry Code Assignment
**DePaul University — Kellstadt Graduate School of Business**
MGT 599 Capstone Project | Morningstar Reference Entity Data (RED) Team | May 2026

---

## Project Overview
This project develops an end-to-end NLP machine learning pipeline to automate the assignment of Morningstar's Global Equity Classification System (GECS) taxonomy codes to company filings — a process previously performed manually by analysts on the RED team.

We tackle two classification tasks:
- **Task 1:** Assign one of **145 industry codes** to a company based on its primary business description (23,207 instances)
- **Task 2:** Assign one of **428 subindustry activity codes** to individual business segments (27,537 instances)

---

## Best Results

| Task | Best Model | Test Macro-F1 | Target |
|------|-----------|--------------|--------|
| Task 1 (145 classes) | RoBERTa-base fine-tuned | **0.6591** | 0.75 |
| Task 2 (428 classes) | LR Probability Ensemble | **0.4577** | 0.75 |

---

## Pipeline Stages
1. **Data Quality & Feature Engineering** — largest-revenue segment filtering, text concatenation
2. **TF-IDF Baseline Models** — Logistic Regression, LinearSVC, Complement Naive Bayes
3. **Sentence Embedding Augmentation** — MiniLM (384-dim) and mpnet (768-dim) + GECS knowledge augmentation
4. **RoBERTa Fine-tuning** — HuggingFace Transformers, weighted cross-entropy, AdamW optimizer

---

## Repository Structure
