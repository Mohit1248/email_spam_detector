import streamlit as st
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load the tokenizer
with open("tokenizer.json", "r", encoding='utf-8') as f:
    tokenizer_data = json.load(f)  # This is a dict
    tokenizer = tokenizer_from_json(json.dumps(tokenizer_data))  # Convert dict to JSON string

# Load model
model = tf.keras.models.load_model('text_model.h5')

# Constants
max_len = 100  # Must match training

# Preprocessing: remove punctuation and stopwords
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    words = [word.lower() for word in text.split() if word.lower() not in stop_words]
    return ' '.join(words)

def predict_spam(text):
    text = preprocess(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    pred = model.predict(padded)[0][0]
    return "Spam" if pred >= 0.5 else "Not Spam", pred

# Streamlit UI
st.set_page_config(page_title="Spam Email Classifier", layout="centered")
st.title("📩 Spam Email Detector")
st.markdown("Enter your email message below and let the model decide whether it's spam or not!")

user_input = st.text_area("✉ Email Text", height=200)

if st.button("Detect Spam"):
    if user_input.strip() == "":
        st.warning("Please enter an email message.")
    else:
        label, confidence = predict_spam(user_input)
        st.success(f"Prediction: *{label}*")
        st.write(f"Confidence: {confidence:.2f}")
