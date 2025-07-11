# src/retrieve_inverted.py

import json
import joblib
from collections import Counter
from preprocessing import clean_text

def retrieve_using_inverted_index(query_text, inverted_index_path, corpus_path, top_k=10):
    # تحميل الفهرس المقلوب
    print("📥 تحميل الفهرس المقلوب ...")
    inverted_index = joblib.load(inverted_index_path)

    # تحميل الوثائق الأصلية
    print("📄 تحميل الوثائق ...")
    corpus = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc['_id']] = doc['text']

    # تنظيف الاستعلام
    query_cleaned = clean_text(query_text)
    query_words = query_cleaned.split()

    # العثور على الوثائق المطابقة
    doc_counter = Counter()
    for word in query_words:
        if word in inverted_index:
            for doc_id in inverted_index[word]:
                doc_counter[doc_id] += 1

    if not doc_counter:
        print("🚫 لم يتم العثور على نتائج.")
        return []

    # ترتيب النتائج حسب عدد الكلمات المتطابقة
    ranked_docs = doc_counter.most_common(top_k)

    # إرجاع النتائج
    results = [
        {
            "doc_id": doc_id,
            "text": corpus[doc_id],
            "score": score
        }
        for doc_id, score in ranked_docs
    ]

    return results

# ✅ تجربة تشغيل مباشرة
if __name__ == "__main__":
    dataset = "quora"
    results = retrieve_using_inverted_index(
        query_text="What is the best way to learn machine learning?",
        inverted_index_path=f"vector_stores/{dataset}_inverted_index.joblib",
        corpus_path=f"data/{dataset}/cleaned_corpus.jsonl",
        top_k=5
    )

    for res in results:
        print("📄 Doc ID:", res["doc_id"])
        print("📝 Text:", res["text"])
        print("📊 Score (matched words):", res["score"])
        print("-" * 50)
