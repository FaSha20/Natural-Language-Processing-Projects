## 📘 NLP Course Project

###Phase 1: Web Crawling & Information Extraction

This project is the first phase of a multi-stage NLP pipeline focused on building a structured dataset about Persian scientists and philosophers. In this phase, we develop a web crawling system using Selenium to collect biographical texts from Persian-language web sources, primarily Wikipedia. The goal is to convert unstructured textual data into a clean, structured format aligned with a predefined JSON schema.

Given the cost and context limitations of large language models (LLMs), a rule-based keyword filtering method is used to select the most relevant paragraphs from raw text before passing them to an LLM. A reasoning-capable model (`deepseek-reasoner`) extracts key attributes—such as birth/death information, occupations, works, and historical events—into structured JSON. Additionally, a local lightweight model (Unsloth LLaMA-3) is used to normalize location names, and OpenStreetMap’s Nominatim API is queried to enrich data with geocoordinates.

In future phases of this course project, the extracted dataset will serve as training data for instruction-tuning a base language model. This will enable downstream applications such as question answering, biographical summarization, and structured knowledge generation in Persian, all tailored to the domain of historical scientific figures.
