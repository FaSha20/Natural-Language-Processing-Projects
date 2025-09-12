# MultimodalRAG Pipeline

This repository contains a full pipeline for building, retrieving, and evaluating a multimodal (text + image) question answering system focused on Persian scientists and philosophers. The workflow is split into three main notebooks:

---

## 1. Preprocessing ([1-Preprocessing.ipynb](MultimodalRAG/1-Preprocessing.ipynb))

**Purpose:**  
Prepare and enrich the raw scientist data for downstream retrieval and QA tasks.

**Main Steps:**
- **Load Data:** Reads structured JSON records containing biographical and image metadata.
- **Image Downloading:** Downloads images from URLs (including Wikipedia) and saves them locally.
- **Image & Text Embedding:** Uses CLIP-based models (`SajjadAyoubi/clip-fa-vision` and `clip-fa-text`) to encode both images and text into vector representations.
- **Similarity Calculation:** Computes cosine similarity between image embeddings for inspection.
- **Vector Database Storage:** Stores image embeddings in a persistent ChromaDB collection for fast retrieval.
- **Image Retrieval Demo:** Shows how to retrieve and visualize similar images using embeddings.
- **Text Generation:** Converts JSON records into fluent Persian text using an LLM (Gemini API).
- **Data Saving:** Saves enriched records with generated text for later use.

---

## 2. Implementation ([2- Implementation.ipynb](MultimodalRAG/2-%20Implementation.ipynb))

**Purpose:**  
Build the multimodal retrieval system and implement the RAG (Retrieval-Augmented Generation) QA pipeline.

**Main Steps:**
- **Environment Setup:** Installs required libraries (`stanza`, `chromadb`, `FlagEmbedding`).
- **Data Loading:** Loads preprocessed scientist records with text chunks and image URLs.
- **Embedding Models:** Loads CLIP models for text and image encoding, and BGE-M3 for text retrieval.
- **NER Extraction:** Uses a Persian NER model to extract person names for improved retrieval.
- **Vector Database Construction:** Stores text and image embeddings in ChromaDB collections.
- **Retrieval Functions:** Implements functions to search by text or image and return top-k results.
- **RAG QA Generation:** Combines retrieved text and image evidence to answer multiple-choice questions using a multimodal LLM (`google/gemma-3-4b-it`).
- **Pipeline Execution:** Runs the full QA pipeline for different question types (text-only, image-only, multimodal).
- **Base Model Evaluation:** Provides baseline QA using Gemini and Gemma models without retrieval.

---

## 3. Evaluation ([3- Evaluation.ipynb](MultimodalRAG/3-%20Evaluation.ipynb))

**Purpose:**  
Evaluate the performance of the retrieval and QA models using various metrics and visualizations.

**Main Steps:**
- **Utility Functions:** Includes chart plotting and NER extraction utilities.
- **Base Model Evaluation:** Loads and visualizes results for Gemini and Gemma models on test data (without retrieval).
- **Metric Calculation:** Implements precision, recall, and hit@k metrics for retrieval evaluation.
- **Evaluation Pipeline:** Runs automated evaluation over QA datasets, saving detailed and aggregate metrics.
- **Visualization:** Plots performance metrics (pie/bar charts, line plots) for different models and question types.
- **Image Retrieval Visualization:** Loads and displays retrieved images with captions for qualitative analysis.

---

## How to Run

1. **Preprocessing:**  
   Run [1-Preprocessing.ipynb](MultimodalRAG/1-Preprocessing.ipynb) to prepare and embed your scientist data, download images, and store embeddings in ChromaDB.

2. **Implementation:**  
   Run [2- Implementation.ipynb](MultimodalRAG/2-%20Implementation.ipynb) to build the retrieval system, run the RAG QA pipeline, and generate answers for your question set.

3. **Evaluation:**  
   Run [3- Evaluation.ipynb](MultimodalRAG/3-%20Evaluation.ipynb) to compute metrics, visualize results, and inspect retrieval quality.

---

## Requirements

- Python 3.8+
- PyTorch
- HuggingFace Transformers
- ChromaDB
- FlagEmbedding
- PIL, requests, tqdm, matplotlib, arabic_reshaper, bidi
- Gemini API access (for LLM text generation)

---

## Notes

- All code is designed for Persian-language data and supports both text and image modalities.
- The pipeline is modular: you can swap out models or data sources as needed.
- For large-scale runs, ensure you have sufficient disk space for image downloads and vector databases.

---

## Authors

- Mobina Poulaei
- Fatemeh Shahhosseini
