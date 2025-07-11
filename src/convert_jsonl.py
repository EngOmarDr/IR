import json, os

def convert(input_doc, input_query, input_qrel, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    with open(input_doc) as f, open(f"{out_folder}/documents.txt","w") as fo:
        for L in f:
            d = json.loads(L)
            fo.write(f"{d['_id']}\t{d.get('text','').strip()}\n")
    with open(input_query) as f, open(f"{out_folder}/test_queries.txt","w") as fo:
        for L in f:
            q = json.loads(L)
            fo.write(f"{q['_id']}\t{q['text'].strip()}\n")
    with open(input_qrel) as f, open(f"{out_folder}/qrels.txt","w") as fo:
        if ".tsv" in input_qrel or input_qrel.endswith(".txt"):
            next(f, None)
            for L in f:
                fo.write(L if "\t" in L else "\t".join(L.split()) + "\n")
    print(f"✅ Converted {out_folder}")

if __name__=="__main__":
    convert("data/trec-covid/corpus.jsonl","data/trec-covid/queries.jsonl","data/trec-covid/qrels.orig.txt","data/trec-covid")
    convert("data/nfcorpus/corpus.jsonl","data/nfcorpus/queries.jsonl","data/nfcorpus/qrels.txt","data/nfcorpus")
