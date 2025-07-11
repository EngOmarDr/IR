import numpy as np
import torch
from preprocessing import clean_text
import numpy as np

def retrieve_top_k_index(query_text, vectorizer, index, doc_ids, corpus, top_k=10):
    query_cleaned = clean_text(query_text)

    if hasattr(vectorizer, 'transform'):
        # ✅ TF-IDF
        query_vector = vectorizer.transform([query_cleaned])
        if hasattr(query_vector, "toarray"):
            query_vector = query_vector.toarray()

    elif hasattr(vectorizer, '__getitem__') and hasattr(vectorizer, 'vector_size'):
        # ✅ Word2Vec
        tokens = query_cleaned.split()
        vectors = [vectorizer[word] for word in tokens if word in vectorizer]
        if vectors:
            query_vector = np.mean(vectors, axis=0).reshape(1, -1)
        else:
            query_vector = np.zeros((1, vectorizer.vector_size))

    elif isinstance(vectorizer, tuple) and len(vectorizer) == 2:
        # ✅ BERT
        tokenizer, model = vectorizer
        model.eval()
        with torch.no_grad():
            inputs = tokenizer(query_cleaned, return_tensors='pt', truncation=True, padding=True)
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            query_vector = embedding.reshape(1, -1)

    elif isinstance(vectorizer, tuple) and len(vectorizer) == 3:
        # ✅ Hybrid = TF-IDF + BERT (sparse + dense)
        tfidf_vectorizer, tokenizer, model = vectorizer
        model.eval()

        # 🧠 Step 1: TF-IDF vector (sparse)
        tfidf_vec = tfidf_vectorizer.transform([query_cleaned])

        # 🧠 Step 2: BERT vector (dense to sparse)
        with torch.no_grad():
            inputs = tokenizer(query_cleaned, return_tensors='pt', truncation=True, padding=True)
            outputs = model(**inputs)
            bert_vec = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().reshape(1, -1)



        tfidf_dense = tfidf_vec.toarray()
        query_vector = np.hstack([tfidf_dense, bert_vec])


    else:
        raise ValueError("Unknown vectorizer type!")

    distances, indices = index.kneighbors(query_vector, n_neighbors=top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        doc_id = doc_ids[idx]
        results.append({
            "doc_id": doc_id,
            "text": corpus[doc_id]["text"],
            "score": 1 - dist
        })
    return results
