# ---------------------------------------------------------
# Fake News Classification using LSTM 
# ---------------------------------------------------------

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

nltk.download("stopwords")
nltk.download("wordnet")

# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], axis=0)
data = data.sample(frac=1).reset_index(drop=True)

# ---------------------------------------------------------
# 2. CLEAN TEXT FUNCTION
# ---------------------------------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return "text " + " ".join(tokens)


data["clean_text"] = data["text"].apply(clean_text)

# ---------------------------------------------------------
# 3. TOKENIZATION & PADDING
# ---------------------------------------------------------

X = data["clean_text"].values
y = data["label"].values

tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(X)

sequences = tokenizer.texts_to_sequences(X)
padded = pad_sequences(sequences, maxlen=300)

X_train, X_test, y_train, y_test = train_test_split(
    padded, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 4. BUILD LSTM MODEL
# ---------------------------------------------------------

model = Sequential()
model.add(Embedding(50000, 128, input_length=300))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

# ---------------------------------------------------------
# 5. TRAIN MODEL
# ---------------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_split=0.2,
    callbacks=[early_stop],
    batch_size=64,
)

# ---------------------------------------------------------
# 6. EVALUATE MODEL
# ---------------------------------------------------------

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# ---------------------------------------------------------
# 7. SAVE MODEL
# ---------------------------------------------------------

model.save("models/lstm_model.h5")

print("Model saved successfully!")


# ---------------------------------------------------------
# 8. FUNCTION TO PREDICT NEW ARTICLES
# ---------------------------------------------------------

def predict_news(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded_seq = pad_sequences(seq, maxlen=300)
    prediction = model.predict(padded_seq)[0][0]
    
    if prediction >= 0.5:
        return "REAL NEWS"
    else:
        return "FAKE NEWS"

# ---------------------------------------------------------
# TEST PREDICTION
# ---------------------------------------------------------

sample_text = "Government announces new policy to boost economy."
print("Prediction:", predict_news(sample_text))
