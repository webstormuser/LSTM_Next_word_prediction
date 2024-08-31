import streamlit as st
import pandas as pd
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the LSTM model
model = load_model('hamlet_lstm_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def generate_text(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted = np.argmax(predicted, axis=-1)
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word
    return None

# Streamlit app
st.title('Next word prediction streamlit app using LSTM')
input_text = st.text_input('Enter the sequence of words', 'to be or not to be')
if st.button("Next word predict"):
    max_sequence_len = model.input_shape[1] + 1
    predict_next_word = generate_text(model, tokenizer, input_text, max_sequence_len)
    st.write(f"Next word predicted: {predict_next_word}")
