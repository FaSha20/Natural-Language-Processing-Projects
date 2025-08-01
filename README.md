## 📘 NLP Course Project

### 🧪 Phase 1: Persian Scientist & Philosopher Web Crawling & Data Extractor 

This project is the first phase of a multi-stage NLP pipeline focused on building a structured dataset about Persian scientists and philosophers. In this phase, we develop a web crawling system using Selenium to collect biographical texts from Persian-language web sources, primarily Wikipedia. The goal is to convert unstructured textual data into a clean, structured format aligned with a predefined JSON schema.

Given the cost and context limitations of large language models (LLMs), a rule-based keyword filtering method is used to select the most relevant paragraphs from raw text before passing them to an LLM. A reasoning-capable model (`deepseek-reasoner`) extracts key attributes—such as birth/death information, occupations, works, and historical events—into structured JSON. Additionally, a local lightweight model (Unsloth LLaMA-3) is used to normalize location names, and OpenStreetMap’s Nominatim API is queried to enrich data with geocoordinates.

In future phases of this course project, the extracted dataset will serve as training data for Retrieval with Finetuning a language model. This will enable downstream applications such as question answering, biographical summarization, and structured knowledge generation in Persian, all tailored to the domain of historical scientific figures.


### 🧪 Phase 2: Persian Historical QA Dataset & Retriever Models
This project builds a Persian question-answering dataset from historical JSON data and evaluates three retrieval models: TF-IDF, Zero-shot, and Fine-tuned transformer models. The workflow includes data preparation, model training, and human evaluation.

#### step 1: Data Preparation & Question Generation
JSON to Text: Converts historical JSON records into fluent Persian sentences using LLMs.
Compositional QA Generation: Creates diverse, compositional questions for each text chunk using LLMs.
Dataset Construction: Pairs each question with its relevant chunk and splits into train/test sets.
#### step 2: Retriever Model Design & Training
TF-IDF Baseline: Retrieves answers using cosine similarity over TF-IDF vectors.
Zero-Shot GLOT500: Uses a pre-trained GLOT500 transformer to encode and retrieve without domain-specific training.
Fine-Tuned Model: Fine-tunes GLOT500 with QA pairs using contrastive loss for improved retrieval.
#### step 3: Human Evaluation & Results
