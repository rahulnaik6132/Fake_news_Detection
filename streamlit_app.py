import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

# Load the trained LSTM model
model = load_model("models/lstm_model.h5")

# Load dataset (to rebuild tokenizer)
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")
data = pd.concat([fake, true], axis=0)

# ---------- TEXT CLEANING (same as training) ----------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return "text " + " ".join(tokens)

# ---------- REBUILD TOKENIZER (same as training) ----------
tokenizer = Tokenizer(num_words=50000)
data["clean_text"] = data["text"].apply(clean_text)
tokenizer.fit_on_texts(data["clean_text"])

# ---------- STREAMLIT UI ----------
st.title("📰 Fake News Detector")

user_input = st.text_area("Enter news text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.write("⚠️ Please enter some text!")
    else:
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=300)

        pred = model.predict(padded)[0][0]
        result = "REAL NEWS ✅" if pred >= 0.5 else "FAKE NEWS ❌"

        st.subheader(result)
