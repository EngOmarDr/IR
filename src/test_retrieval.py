from src.indexing import build_index
from src.retrieval import retrieve_top_k_index
import joblib
import os

# تحميل الموارد
dataset = "antique"
vector_store = "vector_stores"
corpus_path = os.path.join("data", dataset, "cleaned_corpus.jsonl")
corpus = {}
with open(corpus_path, "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        corpus[doc["_id"]] = {"text": doc["text"]}

# تحميل الفهرس والـ vectorizer
doc_ids, index = joblib.load(os.path.join(vector_store, f"{dataset}_bert_index.docids"))  # لو استخدمت faiss
vectorizer = (tokenizer_bert, model_bert)  # حسب التعديل

query = "نص الاستعلام هنا"
results = retrieve_top_k_index(query, vectorizer, index, doc_ids, corpus, top_k=5)

for res in results:
    print(res["doc_id"], res["score"])
    print(res["text"][:300])
    print("----")
