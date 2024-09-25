import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Embedding
import os
import pickle

max_sequence_length = 60
# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load model architecture
st.title("Model Loader Test")

try:
    with open("model11.json", "r") as json_file:
        model_json = json_file.read()
    loaded_model = model_from_json(model_json)
    loaded_model.load_weights("model11.weights.h5")

    # Load the embedding matrix
    embedding_matrix = np.load("embedding_matrix11.npy")
    if isinstance(loaded_model.layers[0], Embedding):
        loaded_model.layers[0].set_weights([embedding_matrix])
    
    st.success("Model loaded successfully!")
    st.text(loaded_model.summary())

except Exception as e:
    st.error(f"Error loading model: {str(e)}")

# Function to generate summary based on input
def generate_summary(seed_text, max_length=50):
    if loaded_model is None:
        st.error("Model is not loaded.")
        return ""

    for _ in range(max_length):
        seed_sequence = tokenizer.texts_to_sequences([seed_text])[0]
        padded_seed_sequence = pad_sequences([seed_sequence], maxlen=max_sequence_length-1, padding='pre')
        predicted_index = np.argmax(loaded_model.predict(padded_seed_sequence), axis=-1)
        predicted_word = tokenizer.index_word.get(predicted_index[0], '')
        seed_text += ' ' + predicted_word
        if predicted_word == '.':
            break  # End of sentence
    return seed_text.split('.')[-1].strip()

# Streamlit app layout
st.title("Intelligent Feedback System")
st.write("### DT Project Showcase Module Demonstration")

# Input fields for user
input_compliment = st.text_input("Enter Compliments:", placeholder="e.g., The project's innovative approach to problem-solving is highly commendable.")
input_questions = st.text_input("Enter Questions:", placeholder="e.g., How did you validate the performance of your ML model?")
input_feedback = st.text_input("Enter Feedback:", placeholder="e.g., Constructive feedback on potential improvements and future research directions.")

# Button to generate summary
if st.button("Generate Summary"):
    if input_compliment or input_questions or input_feedback:
        input_text = input_compliment + " " + input_questions + " " + input_feedback
        generated_summary = generate_summary(input_text)
        st.subheader("Summary and Takeaway:")
        st.write(generated_summary)
    else:
        st.warning("Please enter Compliments, Questions, and Feedback to summarize.")
