import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



st.set_page_config(page_title="Q&A Chatbot", page_icon="💬", layout="wide")


def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@st.cache_data
def load_kb(json_path: str):
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p.resolve()}")
    data = json.loads(p.read_text(encoding="utf-8"))
    # Validate
    if not isinstance(data, list) or not data:
        raise ValueError("KB JSON must be a non-empty list of {q,a}.")
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "q" not in item or "a" not in item:
            raise ValueError(f"Bad KB item at index {i}. Must contain 'q' and 'a'.")
    return data


@st.cache_resource
def build_index(kb):
    # Search over both question + answer
    docs = [f"{x['q']} {x['a']}" for x in kb]
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        stop_words="english",
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(docs)
    return vectorizer, X


def retrieve(query: str, kb, vectorizer, X, top_k: int = 5, threshold: float = 0.10):
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, X).flatten()

    ranked_idx = scores.argsort()[::-1]
    best_idx = int(ranked_idx[0])
    best_score = float(scores[best_idx])

    results = []
    for i in ranked_idx[:top_k]:
        results.append({
            "Question": kb[i]["q"],
            "Answer": kb[i]["a"],
            "Score": float(scores[i])
        })

    if best_score < threshold:
        return None, best_score, results

    return kb[best_idx], best_score, results



st.title("💬 Q&A Chatbot (TF-IDF + Cosine Similarity)")
st.caption("Type a question. The bot retrieves the closest matching Q→A from your JSON knowledge base.")

with st.sidebar:
    st.header("Settings")
    json_path = st.text_input("Knowledge base JSON path", value="da_questions.json")
    top_k = st.slider("Top-K matches", 1, 10, 5)
    threshold = st.slider("Confidence threshold", 0.0, 0.5, 0.10, 0.01)
    show_debug = st.checkbox("Show debug (scores)", value=True)

# Load KB 
try:
    kb = load_kb(json_path)
    vectorizer, X = build_index(kb)
    st.success(f"Loaded {len(kb)} Q→A pairs.")
except Exception as e:
    st.error(str(e))
    st.stop()

query = st.text_input("Ask something", placeholder="e.g., Tell me about your experience with SQL")

if st.button("Ask") or query:
    if not query.strip():
        st.warning("Please type a question.")
    else:
        best, best_score, results = retrieve(query, kb, vectorizer, X, top_k=top_k, threshold=threshold)

        col1, col2 = st.columns([2, 1])

        if best is None:
             st.warning(f"Low confidence (best score = {best_score:.3f}). Try rephrasing.")
        else:
            st.subheader("✅ Best Answer")
            st.markdown(best["a"])
            st.caption(f"Matched question: **{best['q']}**  |  Confidence: **{best_score:.3f}**")
        
        st.markdown("---")
        st.subheader("📌 Top Matches")
        df = pd.DataFrame(results)
        if not show_debug:
            df = df[["Question", "Answer"]]
        st.dataframe(df, use_container_width=True)

        if best is None:
            st.subheader("Did you mean:")
            for r in results[:3]:
                st.write(f"• {r['Question']}")