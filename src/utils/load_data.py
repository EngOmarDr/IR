import json

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

def load_documents(path):
    docs = load_jsonl(path)
    # print([list(doc.keys()) for doc in docs[:3]])  # اطبع مفاتيح أول 3 مستندات
    return {doc['_id']: doc['text'] for doc in docs}

def load_queries(path):
    queries = load_jsonl(path)
    # print([list(query.keys()) for query in queries[:3]])
    return {query['_id']: query['text'] for query in queries}

def load_qrels(path):
    qrels = load_jsonl(path)
    rel_dict = {}
    for item in qrels:
        rel_dict.setdefault(item['query_id'], set()).add(item['doc_id'])
    return rel_dict
