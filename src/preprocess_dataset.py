import json
from preprocessing import clean_text
import os
from tqdm import tqdm

# مسارات الملفات
# quora dataset 1
# antique dataset 2
DATASET_NAME = "antique"
DATA_DIR = f"data/{DATASET_NAME}"
CLEANED_CORPUS_PATH = os.path.join(DATA_DIR, "cleaned_corpus.jsonl")
CLEANED_QUERIES_PATH = os.path.join(DATA_DIR, "cleaned_queries.jsonl")

def preprocess_file(input_path, output_path, field):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line in tqdm(infile, desc=f"🔄 Cleaning {os.path.basename(input_path)}"):
            data = json.loads(line)
            if field in data:
                data[field] = clean_text(data[field])
            outfile.write(json.dumps(data) + '\n')

if __name__ == "__main__":
    print("🚀 بدء المعالجة المسبقة للبيانات...")

    preprocess_file(
        input_path=os.path.join(DATA_DIR, "corpus.jsonl"),
        output_path=CLEANED_CORPUS_PATH,
        field="text"
    )

    preprocess_file(
        input_path=os.path.join(DATA_DIR, "queries.jsonl"),
        output_path=CLEANED_QUERIES_PATH,
        field="text"
    )

    print("✅ تم حفظ الملفات المنظفة بنجاح:")
    print(f"📄 {CLEANED_CORPUS_PATH}")
    print(f"📄 {CLEANED_QUERIES_PATH}")
