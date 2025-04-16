import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model and tokenizer
model = load_model('my_model.h5')

# Load the tokenizer (make sure you saved it during training)
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Set the maximum sequence length (same as used during training)
MAX_LEN = 200  # Change it to your model's max length

# Streamlit UI
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter your movie review to predict its sentiment.")

user_input = st.text_input("Enter your review:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a valid review.")
    else:
        # Tokenize and pad the input
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=MAX_LEN)

        # Predict sentiment
        print(model.predict(padded),"9999")
        prediction = model.predict(padded)[0][0]  # Assuming binary classification
        print(prediction)


        # Interpret the result
        sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"
        st.success(f"Sentiment: {sentiment} (Confidence: {prediction:.2f})")

