# src/indexing.py

import joblib
import os
import numpy as np
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors
import faiss  # ✅ أضف هذا

def build_index(matrix_path, index_path, n_neighbors=10, use_faiss=False):
    print(f"📦 تحميل التمثيلات من {matrix_path} ...")
    ids, vectors = joblib.load(matrix_path)

    print(f"🔍 {'Sparse' if issparse(vectors) else 'Dense'} تمثيل، اختيار algorithm مناسب...")

    if use_faiss and not issparse(vectors):
        print(f"⚙️ بناء فهرس باستخدام FAISS (dense, fast)...")

        # ✅ تأكد أن vectors مصفوفة NumPy
        if isinstance(vectors, list):
            vectors = np.array(vectors)

        vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        print(f"💾 حفظ الفهرس إلى {index_path}.index و doc_ids إلى {index_path}.docids ...")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path + ".index")
        joblib.dump(ids, index_path + ".docids")

    else:
        print(f"⚙️ بناء فهرس باستخدام NearestNeighbors (n={n_neighbors}) ...")
        if issparse(vectors):
            index = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
        else:
            # ✅ إن كانت list حوّلها
            if isinstance(vectors, list):
                vectors = np.array(vectors)
            index = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        index.fit(vectors)

        print(f"💾 حفظ الفهرس إلى {index_path} ...")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        joblib.dump((ids, index), index_path)

    print("✅ تم بناء الفهرس بنجاح.")


# 👇 استخدم هنا antique وفعّل FAISS فقط لـ dense
if __name__ == "__main__":
    dataset = "quora"
    vector_store = "vector_stores"

    build_index(
        matrix_path=os.path.join(vector_store, f"{dataset}_tfidf_matrix.joblib"),
        index_path=os.path.join(vector_store, f"{dataset}_tfidf_index.joblib"),
        use_faiss=False
    )

    build_index(
        matrix_path=os.path.join(vector_store, f"{dataset}_word2vec_vectors.joblib"),
        index_path=os.path.join(vector_store, f"{dataset}_word2vec_index.joblib"),
        use_faiss=False
    )

    build_index(
        matrix_path=os.path.join(vector_store, f"{dataset}_bert_vectors.joblib"),
        index_path=os.path.join(vector_store, f"{dataset}_bert_index"),
        use_faiss=True
    )

    build_index(
        matrix_path=os.path.join(vector_store, f"{dataset}_hybrid_vectors.joblib"),
        index_path=os.path.join(vector_store, f"{dataset}_hybrid_index"),
        use_faiss=True  # ✅ نستخدم FAISS لأنه الآن dense
    )

