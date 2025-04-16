import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

model = load_model('my_model.h5')

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

MAX_LEN = 200

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter your movie review to predict its sentiment.")

user_input = st.text_input("Enter your review:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a valid review.")
    else:
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=MAX_LEN)

        print(model.predict(padded),"9999")
        prediction = model.predict(padded)[0][0]
        print(prediction)

        sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"
        st.success(f"Sentiment: {sentiment} (Confidence: {prediction:.2f})")

