import json
import os
import joblib
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.sparse import hstack
from preprocessing import clean_text
from scipy.sparse import issparse


# ----------------------------- 🔹 Utility 🔹 -----------------------------
def load_clean_texts(jsonl_path, field='text'):
    texts = []
    ids = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            ids.append(data['_id'])
            texts.append(data[field])
    return ids, texts

def ensure_dir(path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)



# ✅ في أعلى الملف (خارج أي دالة)
def custom_tokenizer(text):
    return clean_text(text).split()

# -------------------------- 🔹 TF-IDF 🔹 --------------------------
def build_tfidf(corpus_path, vectorizer_path, matrix_path):
    print("🔍 [TF-IDF] تحميل الوثائق المنظفة...")
    ids, texts = load_clean_texts(corpus_path)

    print("⚙️ [TF-IDF] تدريب نموذج TF-IDF ...")
    vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, max_features=50000)
    tfidf_matrix = vectorizer.fit_transform(texts)

    print("💾 [TF-IDF] حفظ النموذج والـ matrix ...")
    ensure_dir(vectorizer_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump((ids, tfidf_matrix), matrix_path)

    print("✅ [TF-IDF] تم تمثيل الوثائق بنجاح.")

# -------------------------- 🔹 Word2Vec 🔹 --------------------------
def build_word2vec(corpus_path, model_path, matrix_path, vectorizer_path):
    print("🔤 [W2V] تحميل الوثائق المنظفة ...")
    ids, texts = load_clean_texts(corpus_path)
    tokenized_texts = [text.split() for text in texts]

    print("⚙️ [W2V] تدريب نموذج Word2Vec ...")
    model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=2, workers=4)

    print("📊 [W2V] إنشاء تمثيلات الوثائق ...")
    doc_embeddings = []
    for tokens in tqdm(tokenized_texts):
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        avg_vec = np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
        doc_embeddings.append(avg_vec)

    print("💾 [W2V] حفظ النموذج والتمثيلات والـ vectorizer...")
    ensure_dir(model_path)
    model.save(model_path)

    ensure_dir(matrix_path)        # <--- تأكد وجود مجلد حفظ ملف التمثيلات
    joblib.dump((ids, doc_embeddings), matrix_path)

    ensure_dir(vectorizer_path)
    joblib.dump(model.wv, vectorizer_path)

    print("✅ [W2V] تم تمثيل الوثائق بنجاح.")



    # -------------------------- 🔹 BERT 🔹 --------------------------
def build_bert(corpus_path, model_path, matrix_path, vectorizer_path, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    print("🤖 [BERT] تحميل الوثائق المنظفة ...")
    ids, texts = load_clean_texts(corpus_path)

    print(f"📦 [BERT] تحميل النموذج: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model.eval()

    print("📊 [BERT] إنشاء تمثيلات الوثائق ...")
    doc_embeddings = []

    with torch.no_grad():
        for text in tqdm(texts):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            doc_embeddings.append(embedding)

    print("💾 [BERT] حفظ التمثيلات ...")
    ensure_dir(matrix_path)
    joblib.dump((ids, doc_embeddings), matrix_path)

    print("💾 [BERT] حفظ vectorizer (tokenizer + model) ...")
    ensure_dir(vectorizer_path)
    joblib.dump((tokenizer, model), vectorizer_path)  # ✅ هنا الجديد

    print("✅ [BERT] تم تمثيل الوثائق بنجاح.")



# -------------------------- 🔹 Hybrid 🔹 --------------------------

def build_hybrid(tfidf_matrix_path, bert_vectors_path, output_path):
    print("🧬 [Hybrid] تحميل تمثيلات TF-IDF و BERT ...")
    tfidf_ids, tfidf_matrix = joblib.load(tfidf_matrix_path)
    bert_ids, bert_vectors = joblib.load(bert_vectors_path)

    assert tfidf_ids == bert_ids, "❌ IDs mismatch between TF-IDF and BERT!"

    print("🔗 [Hybrid] دمج التمثيلين ...")

    # ✅ تأكد أن BERT هو dense NumPy matrix
    bert_matrix = np.vstack(bert_vectors).astype(np.float32)

    # ✅ دمج sparse + dense باستخدام hstack
    from scipy.sparse import hstack, csr_matrix
    bert_sparse = csr_matrix(bert_matrix)  # تحويل bert إلى sparse لتوافق الدمج

    from scipy.sparse import hstack
    hybrid_sparse = hstack([tfidf_matrix, bert_sparse])
     # ⛳ حوله إلى dense مسبقًا


    print("💾 [Hybrid] حفظ التمثيلات المدمجة ...")
    ensure_dir(output_path)
    joblib.dump((tfidf_ids, hybrid_sparse), output_path)

    print("✅ [Hybrid] تم بناء التمثيل الهجين بنجاح.")





# -------------------------- 🔹 MAIN 🔹 --------------------------
if __name__ == "__main__":
    dataset = "quora"  # بدلًا من "quora"
    data_dir = f"data/{dataset}"
    vector_store = "vector_stores"

    corpus_path = os.path.join(data_dir, "cleaned_corpus.jsonl")

    # ✅ 1. تمثيل TF-IDF
    build_tfidf(
        corpus_path=corpus_path,
        vectorizer_path=os.path.join(vector_store, f"{dataset}_tfidf_vectorizer.joblib"),
        matrix_path=os.path.join(vector_store, f"{dataset}_tfidf_matrix.joblib")
    )

    # ✅ 2. تمثيل Word2Vec
    build_word2vec(
    corpus_path=corpus_path,
    model_path=os.path.join(vector_store, f"{dataset}_word2vec.model"),
    matrix_path=os.path.join(vector_store, f"{dataset}_word2vec_vectors.joblib"),
    vectorizer_path=os.path.join(vector_store, f"{dataset}_word2vec_vectorizer.joblib")  # ✅ الجديد
    )

        # ✅ 3. تمثيل BERT
    build_bert(
        corpus_path=corpus_path,
        model_path=None,  # لا حاجة لحفظ النموذج منفصلًا
        matrix_path=os.path.join(vector_store, f"{dataset}_bert_vectors.joblib"),
        vectorizer_path=os.path.join(vector_store, f"{dataset}_bert_vectorizer.joblib")  # ✅ الجديد
    )

    # ✅ 4. Hybrid = TF-IDF + BERT
    build_hybrid(
        tfidf_matrix_path=os.path.join(vector_store, f"{dataset}_tfidf_matrix.joblib"),
        bert_vectors_path=os.path.join(vector_store, f"{dataset}_bert_vectors.joblib"),
        output_path=os.path.join(vector_store, f"{dataset}_hybrid_vectors.joblib")
    )



