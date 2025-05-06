# Teaching BERT New Facts – CS 5322 Final Project

This project investigates whether a pretrained BERT model (`bert-base-uncased`) can be incrementally updated with new factual knowledge through lightweight fine-tuning — without catastrophic forgetting or hallucination.

## 🧠 Project Goal

- **Problem:** Pretrained LLMs like BERT are static and can't easily learn new facts after training.
- **Objective:** Inject new facts (e.g., “The Eagles won the Super Bowl in 2025”) into BERT and evaluate:
  - If it learns the fact (top-1 / top-5 prediction)
  - If it forgets previous knowledge
  - If it hallucinates unrelated facts

## 📚 Facts and Evaluation

We tested 3 factual updates:
- `Chiefs 2024 win`
- `Eagles 2025 win`
- `Jay Hartzell as SMU President`

For each, we created:
- 50+ templated factual **training sentences**
- Multiple **masked evaluation prompts**
- Known facts and random trivia for control

## 🧪 Methodology

- Fine-tuned BERT using HuggingFace `Trainer` on each fact group
- Varied training size (`5`, `10`, `50` sentences) and epochs (`1`, `3`, `5`)
- Evaluated model accuracy before and after injection
- Tracked:
  - Top-1 / Top-5 Accuracy
  - Confidence
  - Forgetting (known facts)
  - Hallucination (random prompts)

## 🗂️ File Structure

```
.
├── bert_fact_eval_results.csv
├── top5.csv
├── Project Topic 3.ipynb
├── NLP Project 3 Report.qmd
├── presentation/
│   └── bert_fact_presentation_notes.md
├── docs/
├── README.md
```

## 🧰 Requirements

Install dependencies with:
```bash
pip install transformers pandas matplotlib
```
This project was run in **Google Colab** with GPU enabled.

## 📈 How to Reproduce

1. Open `Project Topic 3.ipynb` in Colab
2. Mount your Google Drive if needed
3. Run the cells to train and evaluate the models
4. Load the CSV to analyze results or use built-in plots

## 🔭 Future Work

Based on our findings, future improvements may include:
- Span masking or T5 for multi-token answers
- Adapter or LoRA-based fine-tuning
- TruthfulQA-style evaluations
- Retention testing across sequential fact updates

## 📎 References

- [From Static to Dynamic – Dai et al.](https://arxiv.org/abs/2204.11136)
- [Knowledge Editing in Language Models – De Cao et al.](https://arxiv.org/abs/2104.07143)
- [Retrieval-Augmented Generation Survey – Mao et al.](https://arxiv.org/abs/2301.00375)
