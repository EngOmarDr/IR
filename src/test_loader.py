from utils.load_data import load_documents, load_queries, load_qrels

docs = load_documents("data/quora/corpus.jsonl")
queries = load_queries("data/quora/queries.jsonl")
qrels = load_qrels("data/quora/qrels.jsonl")

print("📄 Number of documents:", len(docs))
print("❓ Number of queries:", len(queries))
print("✅ First document:", list(docs.items())[0])
print("✅ First query:", list(queries.items())[0])
