# src/indexing.py

import joblib
import os
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.sparse import issparse

def build_index(matrix_path, index_path, n_neighbors=10):
    print(f"📦 تحميل التمثيلات من {matrix_path} ...")
    ids, vectors = joblib.load(matrix_path)

    print(f"🔍 {'Sparse' if issparse(vectors) else 'Dense'} تمثيل، اختيار algorithm مناسب...")

    print(f"⚙️ بناء فهرس باستخدام NearestNeighbors (n={n_neighbors}) ...")

    if issparse(vectors):
        # استخدم brute لأنه يدعم sparse
        index = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
    else:
        # dense vector مثل Word2Vec أو BERT
        index = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')

    index.fit(vectors)

    print(f"💾 حفظ الفهرس إلى {index_path} ...")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    joblib.dump((ids, index), index_path)

    print("✅ تم بناء الفهرس بنجاح.")
#quora
if __name__ == "__main__":
    dataset = "quora"
    vector_store = "vector_stores"

    build_index(
        matrix_path=os.path.join(vector_store, f"{dataset}_tfidf_matrix.joblib"),
        index_path=os.path.join(vector_store, f"{dataset}_tfidf_index.joblib")
    )

    build_index(
        matrix_path=os.path.join(vector_store, f"{dataset}_word2vec_vectors.joblib"),
        index_path=os.path.join(vector_store, f"{dataset}_word2vec_index.joblib")
    )

    build_index(
        matrix_path=os.path.join(vector_store, f"{dataset}_bert_vectors.joblib"),
        index_path=os.path.join(vector_store, f"{dataset}_bert_index.joblib")
    )

    build_index(
        matrix_path=os.path.join(vector_store, f"{dataset}_hybrid_vectors.joblib"),
        index_path=os.path.join(vector_store, f"{dataset}_hybrid_index.joblib")
    )