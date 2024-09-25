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

import streamlit as st
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Embedding


try:
    # Load model architecture
    with open("model11.json", "r") as json_file:
        model_json = json_file.read()
        
    loaded_model = model_from_json(model_json)

    # Load the saved weights (excluding embedding)
    loaded_model.load_weights("model11.weights.h5")

    # Ensure embedding_matrix is defined and properly shaped
    # Replace this with your actual embedding matrix loading logic
    embedding_matrix = np.load("embedding_matrix11.npy")

    # Assuming the first layer is an Embedding layer
    if isinstance(loaded_model.layers[0], Embedding):
        loaded_model.layers[0].set_weights([embedding_matrix])
    
except Exception as e:
    st.error(f"Error loading model: {str(e)}")



# Load model architecture
# with open("model11.json", "r") as json_file:
#     model_json = json_file.read()
# loaded_model = model_from_json(model_json)

# Load the saved weights (excluding embedding)
#loaded_model.load_weights("model11.weights.h5")

# Manually load the embedding matrix
#embedding_matrix = np.load('embedding_matrix11.npy')  # Load the pre-saved embedding matrix

# Reassign the embedding weights to the model
#loaded_model.layers[0].set_weights([embedding_matrix])

# Function to generate summary based on input
def generate_summary(seed_text, max_length=50):
    output_text = seed_text  # Initialize output text with the seed text
    for _ in range(max_length):
        try:
            # Tokenize the input sequence
            seed_sequence = tokenizer.texts_to_sequences([output_text])[0]
            # Pad the input sequence
            padded_seed_sequence = pad_sequences([seed_sequence], maxlen=max_sequence_length-1, padding='pre')
            
            # Check input shape
            if padded_seed_sequence.shape != (1, max_sequence_length - 1):
                st.error("Input shape is not as expected.")
                return output_text
            
            # Predict the next word
            predictions = loaded_model.predict(padded_seed_sequence)
            if predictions is None or len(predictions) == 0:
                st.error("Model returned no predictions.")
                return output_text
            
            predicted_index = np.argmax(predictions, axis=-1)
            
            # Convert index to word
            predicted_word = tokenizer.index_word.get(predicted_index[0], '')
            
            # Update the output text for the next iteration
            output_text += ' ' + predicted_word
            
            # Break if a period is predicted, assuming the end of a sentence
            if predicted_word == '.':
                break
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            return output_text  # Return the current output text in case of error
            
    return output_text.split('.')[-1].strip()


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
