import numpy as np
import torch
import faiss  # ✅ دعم FAISS
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse
from preprocessing import clean_text

def retrieve_top_k_index(query_text, vectorizer, index, doc_ids, corpus, top_k=10):
    query_cleaned = clean_text(query_text)

    # ✅ 1. استخراج تمثيل الاستعلام
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
        # ✅ Hybrid = TF-IDF + BERT
        tfidf_vectorizer, tokenizer, model = vectorizer
        model.eval()

        tfidf_vec = tfidf_vectorizer.transform([query_cleaned])  # هذا يبقى sparse

        with torch.no_grad():
            inputs = tokenizer(query_cleaned, return_tensors='pt', truncation=True, padding=True)
            outputs = model(**inputs)
            bert_vec = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().reshape(1, -1)

        from scipy.sparse import hstack, csr_matrix
        bert_sparse = csr_matrix(bert_vec)
        query_vector = hstack([tfidf_vec, bert_sparse])  # ✅ الآن query_vector يبقى sparse


    else:
        raise ValueError("Unknown vectorizer type!")

    # ✅ 2. البحث باستخدام FAISS أو NearestNeighbors
    if isinstance(index, faiss.Index):
        query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.ndim == 1:  
            query_vector = query_vector.reshape(1, -1)
        distances, indices = index.search(query_vector, top_k)
    else:
        if not hasattr(index, 'kneighbors'):
            index_model = NearestNeighbors(n_neighbors=top_k, metric='cosine', algorithm='brute')
            index_model.fit(index)  # ⛳ مباشرة بدون تحويل إلى dense
            index = index_model

        distances, indices = index.kneighbors(query_vector, n_neighbors=top_k)

    # ✅ 3. تجهيز النتائج
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        doc_id = doc_ids[idx]
        results.append({
            "doc_id": doc_id,
            "text": corpus[doc_id]["text"],
            "score": 1 - dist  # لأن المسافة cosine
        })

    return results
