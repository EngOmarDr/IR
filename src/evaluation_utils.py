import json
from collections import defaultdict
from typing import List, Dict

def load_qrels(path: str) -> Dict[str, List[str]]:
    qrels = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if int(item.get("relevance", 0)) > 0:
                qrels[item["query_id"]].append(item["doc_id"])
    return qrels

def average_precision(predicted: List[str], relevant: List[str]) -> float:
    if not relevant:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    for i, doc_id in enumerate(predicted):
        if doc_id in relevant:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / len(relevant)

def mean_average_precision(all_predictions: Dict[str, List[str]], qrels: Dict[str, List[str]]) -> float:
    scores = []
    for query_id, predicted_docs in all_predictions.items():
        ap = average_precision(predicted_docs, qrels.get(query_id, []))
        scores.append(ap)
    return sum(scores) / len(scores) if scores else 0.0

def mean_reciprocal_rank(all_predictions: Dict[str, List[str]], qrels: Dict[str, List[str]]) -> float:
    rr_total = 0
    for query_id, predicted_docs in all_predictions.items():
        relevant_docs = qrels.get(query_id, [])
        for rank, doc_id in enumerate(predicted_docs, start=1):
            if doc_id in relevant_docs:
                rr_total += 1 / rank
                break
    return rr_total / len(all_predictions) if all_predictions else 0.0

def precision_at_k(predicted: List[str], relevant: List[str], k=10) -> float:
    top_k = predicted[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k

def recall_at_k(predicted: List[str], relevant: List[str], k=10) -> float:
    top_k = predicted[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant) if relevant else 0.0
