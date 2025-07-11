# 🔍 Intelligent Information Retrieval System (IR Project)

نظام استرجاع معلومات ذكي مبني باستخدام تمثيلات متعددة للنصوص (TF-IDF, Word2Vec, BERT, Hybrid) مع دعم استرجاع معرفي RAG.

---

## 📁 هيكل المشروع

```
project-root/
├── data/                  # بيانات الاستعلامات والوثائق (QUORA / ANTIQUE)
│   └── [dataset]/         # quora / antique
│       ├── corpus.jsonl
│       ├── queries.jsonl
│       └── qrels.jsonl
├── vector_stores/         # تخزين النماذج والتمثيلات والفهارس
├── outputs/               # ملفات التقييم وسجل الاستعلامات
├── notebooks/
│   └── IR_Demo_Notebook.ipynb  # Notebook تفاعلي يشرح مراحل النظام
├── src/
│   ├── preprocessing.py
│   ├── preprocess_dataset.py
│   ├── representation.py
│   ├── indexing.py
│   ├── retrieval.py
│   ├── evaluation_utils.py
│   ├── evaluation.py
│   └── app.py             # واجهة ويب Flask لعرض النظام
|   └── rag.py
```

---

## 🚀 طريقة التشغيل

### 1. تثبيت المتطلبات

```bash
pip install -r requirements.txt
```

> إذا لم يكن لديك ملف `requirements.txt`، أنشئه بالأوامر التالية:
```bash
pip install scikit-learn nltk gensim flask transformers torch tabulate tqdm joblib
```

---

### 2. المعالجة المسبقة للنصوص

```bash
cd src
python preprocess_dataset.py
```

---

### 3. بناء التمثيلات (TF-IDF, Word2Vec, BERT, Hybrid)

```bash
python representation.py
```

---

### 4. بناء الفهارس للاسترجاع

```bash
python indexing.py
```

---

### 5. تقييم جودة الاسترجاع

```bash
python evaluation.py
```

---

### 6. تشغيل واجهة المستخدم (Flask Web App)

```bash
python app.py
```

> ثم افتح المتصفح على:
```
http://127.0.0.1:5000
```

---

## 🎯 ميزات المشروع

- ✅ تمثيل متعدد للنصوص (TF-IDF, Word2Vec, BERT, Hybrid).
- ✅ نظام استرجاع باستخدام `k-NN`.
- ✅ دعم الاسترجاع التوليدي المعتمد على BERT و T5 (RAG).
- ✅ تحليل شامل لأداء النظام باستخدام MAP, MRR, P@10, Recall@10.
- ✅ واجهة تفاعلية للاستعلام والنتائج.
- ✅ Notebook احترافي لشرح كل خطوة بالتفصيل.

---

## 📊 نتائج التقييم (QUORA مثالًا)

| Representation | MAP   | MRR   | P@10  | Recall@10 |
|----------------|-------|-------|-------|------------|
| TF-IDF         | 0.4366| 0.4649| 0.0719| 0.5405     |
| Word2Vec       | 0.3001| 0.3278| 0.0501| 0.3835     |
| BERT           | 0.5048| 0.5289| 0.0826| 0.5993     |
| Hybrid         | 0.5057| 0.5297| 0.0827| 0.6001     |

---

## 🧠 تقنيات مستخدمة

- Python, Flask
- scikit-learn, gensim, nltk
- HuggingFace Transformers (BERT, T5)
- Word2Vec, TF-IDF, NearestNeighbors
- Jupyter Notebook

---

## 👨‍💻 إعداد: 
- اسم الطالب: **[ضع اسمك هنا]**
- الكلية / الجامعة: **[مثلاً: كلية تكنولوجيا المعلومات - جامعة كذا]**
- المشروع: **استرجاع المعلومات الذكي وتحليل التمثيلات النصية**

---

## 📘 لعرض المشروع بشكل مرئي:
افتح الملف:
```
notebooks/IR_Demo_Notebook.ipynb
```
