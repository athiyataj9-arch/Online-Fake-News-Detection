import streamlit as st
import joblib
import re
import numpy as np

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# ==============================
# Load Model & Vectorizer
# ==============================
@st.cache_resource
def load_artifacts():
    model = joblib.load("fake_news_model.pkl")      # Logistic Regression model from notebook
    vectorizer = joblib.load("tfidf_vectorizer.pkl")  # TF-IDF vectorizer from notebook
    return model, vectorizer

model, vectorizer = load_artifacts()

# ==============================
# Text Cleaning Function (matches notebook)
# ==============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)        # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)      # remove non-letters
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text

# ==============================
# Output Label
# ==============================
def output_label(label):
    return "Fake News 🔴" if label == 0 else "Real News 🟢"

# ==============================
# UI
# ==============================
st.title("📰 Fake News Detection System")
st.markdown(
    """
    Enter a news article below and check whether it is **Fake** or **Real**.  
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
if st.button("🔍 Predict"):
    if not news_input.strip():
        st.warning("⚠️ Please enter some news text!")
    else:
        cleaned_text = clean_text(news_input)
        vectorized_text = vectorizer.transform([cleaned_text])

        prediction = model.predict(vectorized_text)[0]
        probability = model.predict_proba(vectorized_text)[0]

        # Result
        st.subheader("📌 Prediction Result")
        st.success(output_label(prediction))

        # Confidence
        st.subheader("📊 Confidence Score")
        st.info(f"Fake News 🔴: **{probability[0]*100:.2f}%**")
        st.info(f"Real News 🟢: **{probability[1]*100:.2f}%**")

# ==============================
# Footer
# ==============================
st.markdown("---")
st.caption("Fake News Detection Project | Logistic Regression + TF-IDF")
