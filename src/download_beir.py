# src/download_beir.py

from beir import util
from beir.datasets.data_loader import GenericDataLoader # type: ignore

def download_and_save(dataset_name, save_folder):
    print(f"⬇️  Downloading {dataset_name}...")
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    out_dir = save_folder
    util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_folder=out_dir).load(split="test")

    # حفظ بصيغة JSONL و TSV
    import json, os

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "documents.jsonl"), "w", encoding="utf-8") as f:
        for doc_id, doc in corpus.items():
            f.write(json.dumps({"_id": doc_id, "text": doc["text"]}) + "\n")

    with open(os.path.join(out_dir, "test_queries.jsonl"), "w", encoding="utf-8") as f:
        for qid, q in queries.items():
            f.write(json.dumps({"_id": qid, "text": q}) + "\n")

    with open(os.path.join(out_dir, "qrels.txt"), "w", encoding="utf-8") as f:
        for qid, docs in qrels.items():
            for doc_id, rel in docs.items():
                f.write(f"{qid}\t{doc_id}\t{rel}\n")

    print(f"✅ Downloaded and saved dataset: {dataset_name} to {out_dir}")

if __name__ == "__main__":
    download_and_save("trec-covid", "data/trec-covid")
    download_and_save("nfcorpus", "data/nfcorpus")
