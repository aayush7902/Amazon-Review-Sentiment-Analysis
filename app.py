import streamlit as st
import pandas as pd
import re
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline

# ---------------------------
# 🔹 Load Dataset (FAST)
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Reviews.csv", nrows=20000)
    df = df[['Text', 'Score']]
    df.dropna(inplace=True)

    def convert_sentiment(score):
        return "positive" if score >= 4 else "negative"

    df['Sentiment'] = df['Score'].apply(convert_sentiment)
    return df

df = load_data()

# ---------------------------
# 🔹 Text Cleaning
# ---------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

df['Cleaned_Text'] = df['Text'].apply(clean_text)

# ---------------------------
# 🔹 Train TF-IDF Model
# ---------------------------
@st.cache_resource
def train_model(df):
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df['Cleaned_Text'])
    y = df['Sentiment']

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    return model, vectorizer

model, vectorizer = train_model(df)

# ---------------------------
# 🔹 Load BERT (SAFE)
# ---------------------------
@st.cache_resource
def load_bert():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

bert_model = load_bert()

# ---------------------------
# 🎯 UI START
# ---------------------------
st.title("📦 Amazon Review Sentiment Analysis")

# ---------------------------
# 📊 Sentiment Distribution
# ---------------------------
st.subheader("📊 Sentiment Distribution")

fig = px.histogram(df, x="Sentiment", color="Sentiment")
st.plotly_chart(fig)

# ---------------------------
# 📈 Score Distribution
# ---------------------------
st.subheader("📈 Score Distribution")

fig2 = px.histogram(df, x="Score")
st.plotly_chart(fig2)

# ---------------------------
# 🔍 User Input Prediction
# ---------------------------
st.subheader("🔍 Check Your Review")

user_input = st.text_area("Enter your review:")

model_choice = st.selectbox(
    "Choose Model",
    ["TF-IDF (Fast)", "BERT (Accurate)"]
)

if st.button("Predict"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)

        if model_choice == "TF-IDF (Fast)":
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]

        else:
            result = bert_model(cleaned[:512])[0]
            pred = "positive" if result['label'] == "POSITIVE" else "negative"

        if pred == "positive":
            st.success("😊 Positive Review")
        else:
            st.error("😡 Negative Review")
    else:
        st.warning("Please enter a review")

# ---------------------------
# 🔥 Top Negative Words
# ---------------------------
st.subheader("🔥 Top Negative Words")

from collections import Counter

neg_words = " ".join(df[df['Sentiment']=="negative"]['Cleaned_Text']).split()
common_words = Counter(neg_words).most_common(10)

words_df = pd.DataFrame(common_words, columns=["Word", "Frequency"])

fig3 = px.bar(words_df, x="Word", y="Frequency")
st.plotly_chart(fig3)

# ---------------------------
# 📋 Show Data
# ---------------------------
st.subheader("📋 Sample Data")
st.dataframe(df.head(50))

import base64

def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)

# Call function
set_bg("AMAZON.jpg")