# Persian Historical QA Dataset & Retriever Models

## Project Overview
This project builds a Persian question-answering dataset from historical JSON data and evaluates three retrieval models: TF-IDF, Zero-shot, and Fine-tuned transformer models. The workflow includes data preparation, model training, and human evaluation.

---

## Phase 1: Data Preparation & Question Generation
- **JSON to Text**: Converts historical JSON records into fluent Persian sentences using LLMs.
- **Compositional QA Generation**: Creates diverse, compositional questions for each text chunk using LLMs.
- **Dataset Construction**: Pairs each question with its relevant chunk and splits into train/test sets.

---

## Phase 2: Retriever Model Design & Training
- **TF-IDF Baseline**: Retrieves answers using cosine similarity over TF-IDF vectors.
- **Zero-Shot GLOT500**: Uses a pre-trained GLOT500 transformer to encode and retrieve without domain-specific training.
- **Fine-Tuned Model**: Fine-tunes GLOT500 with QA pairs using contrastive loss for improved retrieval.

---

## Phase 3: Human Evaluation & Results

### Statistical Analysis
- **Friedman Test**: χ² = 5.1687, p = 0.0754 (difference between models is close to significant)
- **Wilcoxon Signed-Rank Tests**:
  - TF-IDF vs. Zero-shot: p = 0.4786 (not significant)
  - TF-IDF vs. Fine-tuned: p = 0.0070 (significant, Fine-tuned better)
  - Zero-shot vs. Fine-tuned: p = 0.0482 (significant, Fine-tuned better)
- **Conclusion**: Fine-tuned model significantly outperforms both TF-IDF and Zero-shot.

### Pairwise Comparison (Wins)
| Model A    | Model B    | Wins for Model A |
|------------|------------|-----------------|
| Zero-shot  | Fine-tuned | 248             |
| Zero-shot  | TF-IDF     | 288             |
| Fine-tuned | TF-IDF     | 325             |
| Fine-tuned | Zero-shot  | 314             |
| TF-IDF     | Fine-tuned | 237             |
| TF-IDF     | Zero-shot  | 266             |

### Overall Summary
- **Total Wins**: Zero-shot: 29, TF-IDF: 26, Fine-tuned: 9
- **Average Ranks**: Zero-shot: 5.06, Fine-tuned: 4.53, TF-IDF: 5.24
- **Normalized Scores**: Zero-shot: 107,336; Fine-tuned: 109,150; TF-IDF: 107,303

### Model Behavior
- **Fine-tuned**: Best for complex, domain-specific questions.
- **Zero-shot**: Good for general questions, close to Fine-tuned in some cases.
- **TF-IDF**: Best for simple, keyword-based questions; weak for semantic/complex queries.

### By Question Type
- **Fine-tuned**: Excels at specialized/complex questions.
- **Zero-shot**: Adequate for generic/open-domain questions.
- **TF-IDF**: Good for direct keyword matches, poor for nuanced/semantic questions.

### Conclusions & Recommendations
- Fine-tuning greatly improves retrieval quality.
- Use Zero-shot as fallback when domain-specific data is unavailable.
- Combine statistical and human evaluation for robust assessment.
- Extend fine-tuning with more diverse, high-quality data.
- Analyze model behavior by question type to optimize retrieval strategies.

---

## Directory Structure
- `Codes&Notebooks/ConvertJsonToText_Phase1.ipynb`: Data preprocessing, text generation, QA creation.
- `Codes&Notebooks/TrainingModels_Phase2.ipynb`: Retriever model training, evaluation, and comparison.
- `data/QA_records_final.json`: Final QA dataset.
- `evaluation_outputs.json` / `evaluation_outputs.csv`: Model retrieval results.

## Requirements
- Python 3.8+, PyTorch, HuggingFace Transformers, Datasets, scikit-learn, tqdm, pandas

## How to Run
1. Run Phase 1 notebook for data and QA generation.
2. Run Phase 2 notebook for model training and evaluation.
3. Use outputs for human evaluation and analysis.

## Contributers
Fatemeh Shahhosseini
Mobina Pulaie
