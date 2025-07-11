import json
import os
import joblib
from tqdm import tqdm
from retrieval import retrieve_top_k_index
from evaluation_utils import (
    load_qrels,
    mean_average_precision,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from transformers import AutoTokenizer, AutoModel
import torch
from tabulate import tabulate

# 👇 عدل هذا السطر فقط لاختيار الداتا
DATASET = "quora"  # ← غيّره إلى "quora" أو "antique" حسب الحاجة
TOP_K = 10
REPRESENTATIONS = ["tfidf", "word2vec", "bert", "hybrid"]

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", DATASET)
VECTOR_STORE = os.path.join(BASE_DIR, "..", "vector_stores")
OUTPUT_DIR = os.path.join("outputs", DATASET)

QRELS_PATH = os.path.join(DATA_DIR, "qrels.jsonl")
QUERIES_PATH = os.path.join(DATA_DIR, "cleaned_queries.jsonl")
CORPUS_PATH = os.path.join(DATA_DIR, "cleaned_corpus.jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_corpus(path):
    corpus = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = {"text": doc["text"]}
    return corpus

def load_queries(path):
    queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            queries[data["_id"]] = data["text"]
    return queries

def load_resources(representation):
    index_path = os.path.join(VECTOR_STORE, f"{DATASET}_{representation}_index.joblib")
    doc_ids, index = joblib.load(index_path)

    if representation == "bert":
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        vectorizer = (tokenizer, model)
    elif representation == "hybrid":
        tfidf_vectorizer = joblib.load(os.path.join(VECTOR_STORE, f"{DATASET}_tfidf_vectorizer.joblib"))
        tokenizer, model = joblib.load(os.path.join(VECTOR_STORE, f"{DATASET}_bert_vectorizer.joblib"))
        vectorizer = (tfidf_vectorizer, tokenizer, model)
    else:
        vectorizer = joblib.load(os.path.join(VECTOR_STORE, f"{DATASET}_{representation}_vectorizer.joblib"))

    return vectorizer, index, doc_ids

def evaluate_representation(rep, queries, qrels, corpus):
    print(f"\n📊 Evaluating representation: {rep.upper()} for dataset: {DATASET.upper()}")
    vectorizer, index, doc_ids = load_resources(rep)
    predictions = {}

    for query_id, query_text in tqdm(queries.items(), desc=f"🔍 Running queries for {rep}"):
        results = retrieve_top_k_index(query_text, vectorizer, index, doc_ids, corpus, top_k=TOP_K)
        predictions[query_id] = [r["doc_id"] for r in results]

    map_score = mean_average_precision(predictions, qrels)
    mrr_score = mean_reciprocal_rank(predictions, qrels)
    p_at_10 = sum(precision_at_k(predictions[q], qrels.get(q, []), k=TOP_K) for q in queries) / len(queries)
    recall = sum(recall_at_k(predictions[q], qrels.get(q, []), k=TOP_K) for q in queries) / len(queries)

    print(f"✅ MAP:  {map_score:.4f}")
    print(f"✅ MRR:  {mrr_score:.4f}")
    print(f"✅ P@10: {p_at_10:.4f}")
    print(f"✅ Recall@10: {recall:.4f}")

    result_data = {
        "representation": rep,
        "dataset": DATASET,
        "MAP": round(map_score, 4),
        "MRR": round(mrr_score, 4),
        "P@10": round(p_at_10, 4),
        "Recall@10": round(recall, 4),
    }
    with open(os.path.join(OUTPUT_DIR, f"{rep}_evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4, ensure_ascii=False)


def evaluate_rag(queries, qrels, corpus, vectorizer, index, doc_ids):
    from app import generate_rag_answer
    predictions = {}

    for query_id, query_text in tqdm(queries.items(), desc="🔍 Evaluating with RAG"):
        results = retrieve_top_k_index(query_text, vectorizer, index, doc_ids, corpus)
        top_texts = [r["text"] for r in results[:5]]
        rag_answer = generate_rag_answer(query_text, top_texts)

        # نحسب التقييم على الوثائق المسترجعة نفسها
        predictions[query_id] = [r["doc_id"] for r in results]

    map_score = mean_average_precision(predictions, qrels)
    mrr_score = mean_reciprocal_rank(predictions, qrels)
    p_at_10 = sum(precision_at_k(predictions[q], qrels.get(q, []), k=10) for q in queries) / len(queries)
    recall = sum(recall_at_k(predictions[q], qrels.get(q, []), k=10) for q in queries) / len(queries)

    return {
        "representation": "rag",
        "dataset": DATASET,
        "MAP": round(map_score, 4),
        "MRR": round(mrr_score, 4),
        "P@10": round(p_at_10, 4),
        "Recall@10": round(recall, 4)
    }


if __name__ == "__main__":
    queries = load_queries(QUERIES_PATH)
    qrels = load_qrels(QRELS_PATH)
    corpus = load_corpus(CORPUS_PATH)

    all_results = []

    for rep in REPRESENTATIONS:
        output_file = os.path.join(OUTPUT_DIR, f"{rep}_evaluation.json")

        if os.path.exists(output_file):
            print(f"⏩ Skipping {rep.upper()} (already evaluated)")
        else:
            try:
                evaluate_representation(rep, queries, qrels, corpus)
            except Exception as e:
                print(f"❌ Error while evaluating {rep.upper()}: {e}")
                continue

        with open(output_file, encoding="utf-8") as f:
            result = json.load(f)
            all_results.append([
                result.get("dataset", DATASET).upper(),
                result["representation"].upper(),
                result["MAP"],
                result["MRR"],
                result["P@10"],
                result["Recall@10"]
            ])

    headers = ["Dataset", "Representation", "MAP", "MRR", "P@10", "Recall@10"]
    print("\n📊 Final Evaluation Results:")
    print(tabulate(all_results, headers=headers, tablefmt="fancy_grid"))
