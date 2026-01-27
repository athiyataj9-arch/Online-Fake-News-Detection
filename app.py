import streamlit as st
import joblib
import re


# Page Config

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)


# Load Model & Vectorizer

@st.cache_resource
def load_artifacts():
    model = joblib.load("fake_news_model.pkl")      # Logistic Regression
    vectorizer = joblib.load("tfidf_vectorizer.pkl")  # TF-IDF
    return model, vectorizer

model, vectorizer = load_artifacts()


# Text Cleaning Function (SAME AS NOTEBOOK)

def wordopt(text):
    text = text.lower()
    text = re.sub(r'[.*?/\\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# Output Label

def output_label(n):
    return "Fake News 🔴" if n == 0 else "Not A Fake News 🟢"


# UI

st.title("📰 Fake News Detection System")
st.write("Enter a news article to check whether it is **Fake** or **Real**.")

news_input = st.text_area(
    "📝 Paste News Text Here",
    height=220,
    placeholder="Enter the full news article text..."
)


# Prediction

if st.button("🔍 Predict"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some news text!")
    else:
        cleaned_text = wordopt(news_input)
        vectorized_text = vectorizer.transform([cleaned_text])

        prediction = model.predict(vectorized_text)[0]
        probability = model.predict_proba(vectorized_text)[0]

        st.subheader("📌 Prediction Result")
        st.success(output_label(prediction))

        st.subheader("📊 Confidence Score")
        st.write(f"Fake News Confidence 🔴: **{probability[0]*100:.2f}%**")
        st.write(f"Real News Confidence 🟢: **{probability[1]*100:.2f}%**")


# Footer

st.markdown("---")
st.caption("Fake News Detection using Logistic Regression & TF-IDF")
