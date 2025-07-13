import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer

# تنزيل الموارد المطلوبة
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')  # لـ lemmatizer

# مكونات التنظيف
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()

def clean_text(text):
    # إزالة الرموز والأرقام
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # تصغير الحروف
    text = text.lower()
    
    # ✅ قسيم الجملة إلى كلمات
    tokens = tokenizer.tokenize(text)

    # حذف الكلمات الشائعة
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)
