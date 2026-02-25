# 🧠 Intelligent Retrieval Q&A Chatbot

A retrieval-based NLP chatbot built using TF-IDF vectorization and cosine similarity.  
The system supports dynamic JSON knowledge bases and is deployed with a Streamlit UI for interactive question answering.

---

## 🚀 Overview

This project implements a vector-space retrieval system that matches user queries against a structured Q&A knowledge base.

Instead of using generative AI, this chatbot:

- Converts text into TF-IDF vectors
- Computes similarity using cosine similarity
- Ranks responses based on relevance
- Returns top-K matches with confidence scores

The system demonstrates practical applications of Natural Language Processing (NLP) and information retrieval.

---

## 🏗️ Architecture

1. Load JSON Knowledge Base (Q → A format)
2. Combine question + answer for improved retrieval
3. Apply TF-IDF vectorization
4. Transform user query into vector space
5. Compute cosine similarity
6. Rank and return top results
7. Display responses via Streamlit UI

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Streamlit
- JSON-based knowledge storage
- Pandas

---

## 📊 Features

- 🔍 Ranked Top-K Retrieval
- 📈 Confidence Scoring
- 🧠 Search over Question + Answer
- ⚡ Efficient Vector Space Matching
- 🎛 Adjustable Similarity Threshold
- 💬 Interactive Streamlit UI

---

## 📂 Project Structure
├── app.py
├── qa_bank.json
├── requirements.txt
└── README.md

🧪 Example Queries

- "Tell me about SQL experience"
- "What is overfitting?"