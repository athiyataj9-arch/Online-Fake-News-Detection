# Fake News Detection - Simplified Version

import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("Fake.csv")  # change path if required
print("Dataset Loaded Successfully!")
print(df.head())

# -----------------------------
# Data Cleaning
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)       # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)     # remove non-letters
    text = re.sub(r"\s+", " ", text).strip() # remove extra spaces
    return text

df["text"] = df["text"].apply(clean_text)

# -----------------------------
# Map labels from 'subject'
# -----------------------------
df["label"] = df["subject"].map({"FAKE": 0, "REAL": 1})

# -----------------------------
# Train-Test Split
# -----------------------------
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------------
# Model Training
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# -----------------------------
# Save Model and Vectorizer
# -----------------------------
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("✅ Model and vectorizer saved successfully!")

# -----------------------------
# Predictions & Evaluation
# -----------------------------
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
