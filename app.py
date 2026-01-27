import streamlit as st
import joblib
import re
import numpy as np

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="News Category Detection",
    page_icon="📰",
    layout="centered"
)

# ==============================
# Load Model, Vectorizer & Category Mapping
# ==============================
@st.cache_resource
def load_artifacts():
    model = joblib.load("fake_news_model.pkl")      # your multi-category Logistic Regression model
    vectorizer = joblib.load("tfidf_vectorizer.pkl")  # TF-IDF vectorizer
    num_to_category = joblib.load("num_to_category.pkl")  # mapping numbers to category names
    return model, vectorizer, num_to_category

model, vectorizer, num_to_category = load_artifacts()

# ==============================
# Text Cleaning Function
# ==============================
def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ==============================
# UI
# ==============================
st.title("📰 News Category Detection System")
st.markdown(
    """
    Enter a news article below and check its **Category**.  
    This system uses **Logistic Regression** with **TF-IDF features**.
    """
)

news_input = st.text_area(
    "📝 Paste News Text Here",
    height=220,
    placeholder="Enter the full news article text..."
)

# ==============================
# Prediction
# ==============================
if st.button("🔍 Predict Category"):
    if not news_input.strip():
        st.warning("⚠️ Please enter some news text!")
    else:
        cleaned_text = clean_text(news_input)
        vectorized_text = vectorizer.transform([cleaned_text])

        # Prediction
        pred_num = model.predict(vectorized_text)[0]
        pred_category = num_to_category[pred_num]

        # Probability / confidence scores
        prob_scores = model.predict_proba(vectorized_text)[0]
        # Map probabilities to categories
        prob_dict = {num_to_category[i]: prob_scores[i]*100 for i in range(len(prob_scores))}
        # Sort by highest probability
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))

        # Result
        st.subheader("📌 Predicted Category")
        st.success(f"**{pred_category}**")

        # Confidence Scores
        st.subheader("📊 Confidence Scores")
        for cat, score in sorted_probs.items():
            st.info(f"{cat}: **{score:.2f}%**")

# ==============================
# Footer
# ==============================
st.markdown("---")
st.caption("News Category Detection Project | Logistic Regression + TF-IDF")
