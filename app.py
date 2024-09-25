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

# # Load model architecture
# with open("model11.json", "r") as json_file:
#     model_json = json_file.read()
# loaded_model = model_from_json(model_json)

import streamlit as st
from tensorflow.keras.models import model_from_json



import streamlit as st
from tensorflow.keras.models import model_from_json
import os

st.title("Model Loader Test")

# Print the current directory for verification
st.text(f"Current directory: {os.getcwd()}")

try:
    # Ensure the model file exists
    if not os.path.exists("model11.json"):
        raise FileNotFoundError("model11.json file not found.")
    
    with open("model11.json", "r") as json_file:
        model_json = json_file.read()
        
    # Load the model architecture
    loaded_model = model_from_json(model_json)
    
    # Ensure the weights file exists
    if not os.path.exists("model11.weights.h5"):
        raise FileNotFoundError("model11.weights.h5 file not found.")
        
    loaded_model.load_weights("model11.weights.h5")
    
    st.success("Model loaded successfully!")
    st.text(loaded_model.summary())
    
except Exception as e:
    st.error(f"Error loading model: {str(e)}")



# Load the saved weights (excluding embedding)
#loaded_model.load_weights("model11.weights.h5")

# Manually load the embedding matrix
embedding_matrix = np.load('embedding_matrix11.npy')  # Load the pre-saved embedding matrix

# Reassign the embedding weights to the model
loaded_model.layers[0].set_weights([embedding_matrix])

# Function to generate summary based on input
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
st.title("Intelligent Feedback System ")
st.write("""
### DT Project Showcase Module Demonstration
""")

justified_text = """
<div style='text-align: justify;'>
This Module is a part of s a virtual project showcase platform designed to transform student presentations by adding a personal touch. Traditional showcases often lack engagement, so we introduced the SlamBook ML module, which generates personalized feedback based on Compliments, Questions, and Constructive Feedback. This AI-driven system celebrates achievements, inspires improvements, and provides students with a memorable digital souvenir. As an exclusive platform, It provides institutions a competitive edge by enhancing student engagement and recognition.
</div>
"""

# Display the justified text in Streamlit
st.markdown(justified_text, unsafe_allow_html=True)
st.write("""

""")
# Wider, one-liner input fields with placeholder hints
input_compliment = st.text_input(
    "Enter Compliments:", 
    placeholder="e.g., The project's innovative approach to problem-solving is highly commendable.",
)

input_questions = st.text_input(
    "Enter Questions:", 
    placeholder="e.g., How did you validate the performance of your ML model?",
)

input_feedback = st.text_input(
    "Enter Feedback:", 
    placeholder="e.g., Constructive feedback on potential improvements and future research directions.",
)

# If the user clicks the button, generate the summary
if st.button("Generate Summary"):
    if input_compliment or input_questions or input_feedback:
        input_text = input_compliment + " " + input_questions + " " + input_feedback
        generated_summary = generate_summary(input_text)
        st.subheader("Summary and Takeaway:")
        st.write(generated_summary)
    else:
        st.warning("Please enter Compliments, Questions, and Feedback to summarize.")
