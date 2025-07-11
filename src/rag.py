from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from retrieval import retrieve_top_k_index

model_name = "google/flan-t5-base"  # أو حتى flan-t5-large إذا عندك GPU
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_answer(query, vectorizer, index, doc_ids, corpus, top_k=5):
    results = retrieve_top_k_index(query, vectorizer, index, doc_ids, corpus, top_k=top_k)
    context = " ".join([r["text"] for r in results])
    input_text = f"question: {query} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    outputs = model.generate(
        input_ids=inputs["input_ids"],                   # ✅ التعديل هنا
        attention_mask=inputs["attention_mask"],         # ✅ مهم جداً
        max_length=150
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

