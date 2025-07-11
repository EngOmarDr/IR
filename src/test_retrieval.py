# src/test_retrieval.py

import json
import joblib
from retrieval import retrieve_top_k_index
from preprocessing import clean_text

# تحميل البيانات
with open("data/quora/cleaned_corpus.jsonl", "r", encoding="utf-8") as f:
    corpus = {json.loads(line)["_id"]: json.loads(line) for line in f}

# تحميل النموذج والفهرس
vectorizer = joblib.load("vector_stores/quora_tfidf_vectorizer.joblib")
doc_ids, index = joblib.load("vector_stores/quora_tfidf_index.joblib")

# استعلام حقيقي
query = "What is the best way to learn AI?"

results = retrieve_top_k_index(query, vectorizer, index, doc_ids, corpus)

for r in results:
    print("📄", r["doc_id"])
    print("📝", r["text"])
    print("📊 Score:", round(r["score"], 3))
    print("-" * 50)
