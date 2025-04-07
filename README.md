# 🔍 Assessment Recommendation Engine

This project is a **Retrieval-Augmented Generation (RAG)** based intelligent recommendation engine designed to suggest the most relevant **SHL assessment tests** for a given job description or query. It supports both free-text queries and URLs pointing to job descriptions.

---

## 🧠 What It Does

- Accepts **text or job description URLs** as input
- Extracts dynamic webpage content using **Selenium**
- Uses **semantic search (FAISS + OpenAI Embeddings)** to find the top 30 most relevant assessments
- Sends these to an LLM (**ChatOpenAI**) for reasoning and selection of the best 10 recommendations
- Returns both **RAG-based** and **similarity-based** results in a user-friendly format (Streamlit)
- Supports programmatic access via a **FastAPI** `/recommendations` endpoint

---

## ✅ Key Features

- End-to-end **Retrieval-Augmented Generation pipeline**
- **Semantic vector search** powered by FAISS
- Context-aware reasoning with **LLM-based reranking**
- Option to extract skills, tools, platforms (e.g., Python, AWS) from query for better relevance
- Automatically fetches and parses job descriptions from URLs
- Dual output: RAG-based answer + top 5 similarity matches with scores
- Easy-to-use **Streamlit UI** and **REST API support**

---

## 🛠️ Tools & Libraries Used

| Phase | Description | Tools/Libraries |
|-------|-------------|-----------------|
| **1. Web Scraping & Data Preparation** | Scraped metadata of SHL assessments and stored in `.jsonl` format | `requests`, `BeautifulSoup`, `json`, `time` |
| **2. Document Preparation & Embedding** | Converted metadata into LangChain documents, embedded using OpenAI, and stored in FAISS | `LangChain`, `OpenAIEmbeddings`, `FAISS` |
| **3. RAG Pipeline & Recommendation** | Retrieved 30 most similar documents and used `ChatOpenAI` to generate final top 10 recommendations | `LangChain`, `ChatOpenAI`, `FAISS`, `cosine similarity` |
| **4. UI + API Integration** | Built an interactive frontend with Streamlit and a backend API using FastAPI | `Streamlit`, `FastAPI` |