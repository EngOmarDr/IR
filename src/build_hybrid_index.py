# build_hybrid_index.py
import os
from indexing import build_index

dataset = "antique"
vector_store = "vector_stores"

build_index(
    matrix_path=os.path.join(vector_store, f"{dataset}_hybrid_vectors.joblib"),
    index_path=os.path.join(vector_store, f"{dataset}_hybrid_index"),
    use_faiss=True
)
