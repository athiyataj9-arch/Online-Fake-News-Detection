# 📰 Online Fake News Detection
📌 Introduction

The rapid growth of digital media has made it easier for misinformation and fake news to spread across online platforms. Fake news can mislead the public, influence opinions, and cause social harm. The Online Fake News Detection project aims to address this issue by using Machine Learning and Natural Language Processing (NLP) techniques to automatically classify news articles as Fake or Real based on their textual content.

This project demonstrates an end-to-end machine learning pipeline, including data preprocessing, feature extraction, model training, evaluation, and deployment readiness through a user-friendly web interface.

🎯 Project Objectives

The main objectives of this project are:

To build a system that can accurately identify fake news articles

To apply NLP techniques for cleaning and preparing textual data

To transform text into numerical features using vectorization techniques

To train and evaluate a machine learning classification model

To provide an easy-to-use interface for testing news articles

# 🛠️Technology Stack

Programming Language: Python

Machine Learning: Scikit-learn

Text Processing: Regular Expressions, TF-IDF Vectorizer

Data Handling: Pandas, NumPy

Model: Logistic Regression

Deployment: Streamlit

Development Environment: Jupyter Notebook / VS Code

📂 Project Structure
├── fake_news_file.py          # Model training script
├── streamlit_app.py           # Streamlit web application
├── Fake.csv                   # Fake news dataset
├── True.csv                   # Real news dataset
├── fake_news_model.pkl        # Trained Logistic Regression model
├── tfidf_vectorizer.pkl       # TF-IDF vectorizer
├── README.md

⚙️ Methodology

The project follows a structured machine learning workflow:

1. Data Collection

Two labeled datasets are used:

Fake.csv – contains fake news articles

True.csv – contains real news articles

Each article is labeled accordingly to create a binary classification problem.

2. Data Preprocessing

The text data is cleaned by:

Converting text to lowercase

Removing URLs, punctuation, and special characters

Removing extra whitespaces

This step ensures that only meaningful textual information is used for training.

3. Feature Extraction

The cleaned text is converted into numerical form using the TF-IDF (Term Frequency–Inverse Document Frequency) technique. This helps the model understand the importance of words relative to the entire dataset.

4. Model Training

A Logistic Regression classifier is trained on the TF-IDF features to distinguish between fake and real news articles.

5. Model Evaluation

The trained model is evaluated using:

Accuracy score

Precision, recall, and F1-score

Probability thresholds are used to handle uncertain predictions more responsibly.

📊 Results and Performance

The model successfully classifies news articles as Fake or Real

Performs reliably on unseen news text

Handles uncertain cases using confidence-based prediction logic

⚠️ Note: Predictions are based on linguistic patterns and do not guarantee factual correctness.

🚀 How to Run the Project
Step 1: Clone the Repository
git clone https://github.com/your-username/Online_Fake_News_Detection.git

Step 2: Navigate to the Project Directory
cd Online_Fake_News_Detection

Step 3: Install Required Dependencies
pip install -r requirements.txt

Step 4: Train the Model
python fake_news_file.py

Step 5: Run the Streamlit Application
streamlit run streamlit_app.py

🧪 How to Use the Application

Paste a news article into the text input field

Click the Predict button

The system will classify the news as:

Fake News 🔴

Real News 🟢

Uncertain ⚠️ (for low-confidence cases)

🔮 Future Enhancements

Improve accuracy using deep learning models such as LSTM or BERT

Add real-time news URL verification

Highlight important words influencing predictions (Explainable AI)

Deploy the system on cloud platforms for public access

📜 Disclaimer

This project is intended for educational purposes only.
It should not be used as a sole source for verifying news authenticity.

👤 Author

MD Ansar
Machine Learning & Data Science Enthusiast