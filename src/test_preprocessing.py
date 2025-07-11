from preprocessing import clean_text

sample = "Here's an example: Does removing STOPWORDS and punctuation really help?"
cleaned = clean_text(sample)
print("✅ Original:", sample)
print("🧹 Cleaned :", cleaned)
