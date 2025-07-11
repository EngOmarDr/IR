import json
import os
from tqdm import tqdm

DATASET_NAME = "antique"
DATA_DIR = f"data/{DATASET_NAME}"

def convert_collection_to_jsonl(collection_path, output_path):
    with open(collection_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in, desc="Converting collection.txt"):
            line = line.strip()
            if not line:
                continue
            # كل سطر: id \t text
            parts = line.split('\t', maxsplit=1)
            if len(parts) != 2:
                continue
            doc_id, text = parts
            json_line = {
                "_id": doc_id,
                "text": text
            }
            f_out.write(json.dumps(json_line, ensure_ascii=False) + "\n")

def convert_queries_to_jsonl(queries_path, output_path):
    with open(queries_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in, desc="Converting queries.txt"):
            line = line.strip()
            if not line:
                continue
            # كل سطر: id \t query_text
            parts = line.split('\t', maxsplit=1)
            if len(parts) != 2:
                continue
            query_id, query_text = parts
            json_line = {
                "_id": query_id,
                "text": query_text
            }
            f_out.write(json.dumps(json_line, ensure_ascii=False) + "\n")

def convert_qrels_to_jsonl(qrels_path, output_path):
    with open(qrels_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                print(f"⚠️ سطر {line_num} يتجاهل لأنه لا يحتوي 4 أجزاء: {line}")
                continue
            query_id, _, doc_id, relevance = parts
            try:
                relevance_int = int(relevance)
            except ValueError:
                print(f"⚠️ سطر {line_num} به قيمة relevance غير صحيحة: {relevance}")
                continue
            json_line = {
                "query_id": query_id,
                "doc_id": doc_id,
                "relevance": relevance_int
            }
            f_out.write(json.dumps(json_line, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    convert_collection_to_jsonl(
        collection_path=os.path.join(DATA_DIR, "collection.txt"),
        output_path=os.path.join(DATA_DIR, "corpus.jsonl")
    )

    convert_queries_to_jsonl(
        queries_path=os.path.join(DATA_DIR, "queries.txt"),
        output_path=os.path.join(DATA_DIR, "queries.jsonl")
    )

    convert_qrels_to_jsonl(
        qrels_path="data/antique/qrels.tsv",
        output_path="data/antique/qrels.jsonl"
    )

    print("✅ تم تحويل ملفات antique إلى jsonl بنجاح!")
