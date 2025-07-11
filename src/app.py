# app.py (إضافة دعم RAG و Query Suggestions)

from flask import Flask, request, render_template, jsonify
import joblib
import json
import os
import datetime
from preprocessing import clean_text
from retrieval import retrieve_top_k_index
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import torch
from tabulate import tabulate
from representation import custom_tokenizer

app = Flask(__name__)

DATASETS = ["quora", "antique"]
REPRESENTATIONS = ["tfidf", "word2vec", "bert", "hybrid"]

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
VECTOR_STORE = os.path.join(BASE_DIR, "..", "vector_stores")
CORPUS_DIR = os.path.join(BASE_DIR, "..", "data")
HISTORY_PATH = os.path.join(BASE_DIR, "..", "outputs", "history.jsonl")

# تحميل الموارد كما سابقاً
def load_resources(dataset, representation):
    index_path = os.path.join(VECTOR_STORE, f"{dataset}_{representation}_index.joblib")
    corpus_path = os.path.join(CORPUS_DIR, dataset, "cleaned_corpus.jsonl")
    doc_ids, index = joblib.load(index_path)
    
    if representation == "bert":
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        vectorizer = (tokenizer, model)
    elif representation == "hybrid":
        tfidf_vectorizer = joblib.load(os.path.join(VECTOR_STORE, f"{dataset}_tfidf_vectorizer.joblib"))
        tokenizer, model = joblib.load(os.path.join(VECTOR_STORE, f"{dataset}_bert_vectorizer.joblib"))
        vectorizer = (tfidf_vectorizer, tokenizer, model)
    else:
        vectorizer_path = os.path.join(VECTOR_STORE, f"{dataset}_{representation}_vectorizer.joblib")
        vectorizer = joblib.load(vectorizer_path)

    corpus = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = {"text": doc["text"]}
    return vectorizer, index, doc_ids, corpus


def load_rag_model():
    base_path = os.path.join(BASE_DIR, "..", "offline_models", "flan-t5-base")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(base_path, "tokenizer"))
    model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(base_path, "model"))
    model.eval()
    return tokenizer, model




rag_tokenizer, rag_model = load_rag_model()


# توليد إجابة معتمدة على نصوص مسترجعة (RAG)
def generate_rag_answer(query, retrieved_texts, max_length=200):
    # فلترة ذكية للنصوص:
    query_keywords = set(query.lower().split())
    unique_texts = []
    seen = set()

    for text in retrieved_texts:
        cleaned = text.strip().lower()
        if len(cleaned) < 25 or cleaned in seen:
            continue
        if any(kw in cleaned for kw in query_keywords):
            if not any(x in cleaned for x in ["newton", "tesla", "other person"]):  # خصائص قابلة للتوسعة
                unique_texts.append(text.strip())
                seen.add(cleaned)

    if not unique_texts:
        unique_texts = retrieved_texts  # fallback

    # استخدام جميع النصوص المسترجعة القابلة للعرض والتي تحتوي على جمل مفيدة
    top_texts = [txt for txt in retrieved_texts if len(txt.split()) > 5][:5]
    context = "\n".join(top_texts)

    prompt = f"""
    Answer the following question using only the information provided in the context. 
    If you are not sure or the information is not present, try to give the best approximate answer based on the context.


    Context:
    {context}

    Question: {query}
    Answer:"""


    inputs = rag_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    with torch.no_grad():
        outputs = rag_model.generate(
            input_ids=inputs["input_ids"],  # ✅ التصحيح هنا
            attention_mask=inputs.get("attention_mask"),  # اختياري لكن يفضل
            max_length=max_length,
            num_beams=5,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            early_stopping=True
        )


    answer = rag_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer




@app.route("/", methods=["GET", "POST"])
def home():
    query = ""
    selected_dataset = "quora"
    selected_representation = "tfidf"
    results = []
    rag_answer = None
    rag_enabled = False
    suggestions_enabled = False

    if request.method == "POST":
        query = request.form["query"]
        selected_dataset = request.form["dataset"]
        selected_representation = request.form["representation"]
        rag_enabled = request.form.get("enable_rag") == "on"
        suggestions_enabled = request.form.get("enable_suggestions") == "on"


        try:
            vectorizer, index, doc_ids, corpus = load_resources(selected_dataset, selected_representation)
            results = retrieve_top_k_index(query, vectorizer, index, doc_ids, corpus)

            # RAG: إذا اخترت BERT أو Hybrid، فعّل الإجابة التوليدية باستخدام RAG
            if rag_enabled and selected_representation in ["bert", "hybrid"]:
                top_texts = [r["text"] for r in results[:5]]
                rag_answer = generate_rag_answer(query, top_texts)
            else:
                rag_answer = None


            # حفظ تاريخ الاستعلام
            os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
            with open(HISTORY_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "query": query,
                    "dataset": selected_dataset,
                    "representation": selected_representation,
                    "timestamp": str(datetime.datetime.now())
                }) + "\n")

        except Exception as e:
            results = [{"doc_id": "Error", "text": str(e), "score": 0}]

    return render_template(
        "index.html",
        datasets=DATASETS,
        representations=REPRESENTATIONS,
        selected_dataset=selected_dataset,
        selected_representation=selected_representation,
        query=query,
        results=results,
        rag_answer=rag_answer,
        rag_enabled=rag_enabled,
        suggestions_enabled=suggestions_enabled
    )




# API للـ Query Suggestions (اقترح من history.jsonl الكلمات الأكثر تطابقًا)
@app.route("/suggest", methods=["GET"])
def suggest_queries():
    partial_query = request.args.get("q", "").lower()
    suggestions = set()

    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    q = item.get("query", "").lower()
                    if q.startswith(partial_query) and q != partial_query:
                        suggestions.add(q)
                        if len(suggestions) >= 10:
                            break
                except:
                    continue

    return jsonify(sorted(list(suggestions)))


@app.route("/evaluation")
def evaluation():
    results_dir = os.path.join(BASE_DIR, "..", "outputs")
    evaluation_files = [f for f in os.listdir(results_dir) if f.endswith("_evaluation.json")]

    evaluations = []
    for fname in evaluation_files:
        path = os.path.join(results_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            evaluations.append(data)

    # ترتيب النتائج حسب التمثيل
    evaluations.sort(key=lambda x: x["representation"])

    return render_template("evaluation.html", evaluations=evaluations)


# ✅ خدمة REST لاسترجاع النتائج فقط (API)
@app.route("/search", methods=["POST"])
def search_api():
    data = request.get_json()
    query = data.get("query")
    dataset = data.get("dataset", "quora")
    representation = data.get("representation", "tfidf")

    try:
        vectorizer, index, doc_ids, corpus = load_resources(dataset, representation)
        results = retrieve_top_k_index(query, vectorizer, index, doc_ids, corpus)

        return jsonify({
            "query": query,
            "results": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ خدمة توليد الإجابة الذكية باستخدام RAG فقط (API)
@app.route("/rag", methods=["POST"])
def rag_api():
    data = request.get_json()
    query = data.get("query")
    texts = data.get("contexts", [])  # قائمة نصوص مسترجعة

    try:
        answer = generate_rag_answer(query, texts)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/search/<rep_type>", methods=["POST"])
def search_specific_rep(rep_type):
    data = request.get_json()
    query = data.get("query")
    dataset = data.get("dataset", "quora")

    if rep_type not in REPRESENTATIONS:
        return jsonify({"error": "Unsupported representation"}), 400

    try:
        vectorizer, index, doc_ids, corpus = load_resources(dataset, rep_type)
        results = retrieve_top_k_index(query, vectorizer, index, doc_ids, corpus)
        return jsonify({
            "query": query,
            "representation": rep_type,
            "results": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/clean_text", methods=["POST"])
def clean_text_api():
    data = request.get_json()
    raw_text = data.get("text", "")
    if not raw_text:
        return jsonify({"error": "No text provided"}), 400
    try:
        cleaned = clean_text(raw_text)
        return jsonify({"cleaned_text": cleaned})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

@app.route("/topics", methods=["POST"])
def topic_modeling():
    from nltk.corpus import stopwords

    data = request.get_json()
    dataset = data.get("dataset", "quora")
    num_topics = int(data.get("n_topics", 5))

    corpus_path = os.path.join(CORPUS_DIR, dataset, "cleaned_corpus.jsonl")
    documents = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            documents.append(doc["text"])

    try:
        vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
        doc_term_matrix = vectorizer.fit_transform(documents)
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(doc_term_matrix)
        terms = vectorizer.get_feature_names_out()

        topic_words = {}
        for i, topic in enumerate(lda.components_):
            topic_words[f"Topic {i+1}"] = [terms[i] for i in topic.argsort()[:-11:-1]]

        return jsonify(topic_words)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)
