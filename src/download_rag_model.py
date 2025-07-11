# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# model_name = "google/flan-t5-base"

# # تحميل النموذج من الإنترنت
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# # حفظه محليًا في مجلد يمكن نسخه لاحقًا
# tokenizer.save_pretrained("offline_models/flan-t5-base/tokenizer")
# model.save_pretrained("offline_models/flan-t5-base/model")

# print("✅ تم تحميل النموذج وحفظه في offline_models/flan-t5-base")


#_____________________________________
from transformers import AutoTokenizer, AutoModel
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
tokenizer.save_pretrained("./my_model/tokenizer")
model.save_pretrained("./my_model/model")
