import json
from collections import defaultdict
import os

# المسارات
original_queries_path = "data/quora/cleaned_queries.jsonl"
history_path = "outputs/history.jsonl"
output_path = "data/quora/cleaned_queries_with_suggestions.jsonl"

# حمّل الاستعلامات الأصلية
original_queries = []
with open(original_queries_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        original_queries.append(item)

# حمّل الاستعلامات من السجل (history)
history_queries = defaultdict(int)
with open(history_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            item = json.loads(line)
            query = item.get("query", "").strip().lower()
            dataset = item.get("dataset", "")
            if dataset == "quora" and query:
                history_queries[query] += 1
        except:
            continue

# استخراج أكثر الاستعلامات تكرارًا كمرشّحات
sorted_history = sorted(history_queries.items(), key=lambda x: x[1], reverse=True)
top_history_queries = [q for q, _ in sorted_history[:300]]  # يمكنك تغيير العدد حسب الحاجة

# أنشئ نسخة محسّنة من الاستعلامات الأصلية
enhanced_queries = []
for i, item in enumerate(original_queries):
    new_text = top_history_queries[i % len(top_history_queries)] if i < len(top_history_queries) else item["text"]
    enhanced_queries.append({
        "_id": item["_id"],
        "text": new_text,
        "metadata": item.get("metadata", {})
    })

# احفظ الملف الجديد
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    for q in enhanced_queries:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")

print(f"✅ تم إنشاء الملف الجديد: {output_path} ({len(enhanced_queries)} استعلام)")
