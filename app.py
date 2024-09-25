import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import pickle

max_sequence_length = 60
# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

from tensorflow.keras.models import model_from_json
from tensorflow.keras.initializers import Constant
import numpy as np

# Load model architecture
with open("model11.json", "r") as json_file:
    model_json = json_file.read()
loaded_model = model_from_json(model_json)

# Load the saved weights (excluding embedding)
loaded_model.load_weights("model11.weights.h5")

# Manually load the embedding matrix
embedding_matrix = np.load('embedding_matrix11.npy')  # Load the pre-saved embedding matrix

# Reassign the embedding weights to the model
loaded_model.layers[0].set_weights([embedding_matrix])

# prompt: use above model

def generate_summary(seed_text, max_length=50):
    for _ in range(max_length):
        # Tokenize the input sequence
        seed_sequence = tokenizer.texts_to_sequences([seed_text])[0]
        # Pad the input sequence
        padded_seed_sequence = pad_sequences([seed_sequence], maxlen=max_sequence_length-1, padding='pre')
        # Predict the next word
        predicted_index = np.argmax(loaded_model.predict(padded_seed_sequence), axis=-1)
        # Convert index to word
        predicted_word = tokenizer.index_word.get(predicted_index[0], '')
        # Update the seed text for the next iteration
        seed_text += ' ' + predicted_word
        if predicted_word == '.':
            break  # Break if a period is predicted, assuming the end of a sentence
    return seed_text.split('.')[-1].strip()

# Streamlit app layout
st.title("Text Summarization App")

input_text = st.text_area("Enter text for summarization:", height=200)

if st.button("Generate Summary"):
    if input_text:
        generated_summary = generate_summary(input_text)
        st.subheader("Model Suggestion:")
        st.write(generated_summary)
    else:
        st.warning("Please enter some text to summarize.")



