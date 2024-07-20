from src.utils.model_download import HuggingFaceModelHandler
import streamlit as st
import torch
# Initialize the model handler
model_handler = HuggingFaceModelHandler()

# Load the model from the saved directory
model_handler.load_from_directory()

# Streamlit app
st.title("Text Classification with HuggingFace Transformers")
st.write("Enter some text and get the predicted class label.")


# Text input
text = st.text_area("Enter text here:")

if st.button("Predict"):
    if text:
        probs, pred_label_idx, pred_label = model_handler.predict(text)

        # Define the prediction labels
        labels = [
            "Computation and Language (CL)",
            "Cryptography and Security (CR)",
            "Distributed and Cluster Computing (DC)",
            "Data Structures and Algorithms (DS)",
            "Logic in Computer Science (LO)",
            "Networking and Internet Architecture (NI)",
            "Software Engineering (SE)"
        ]

        # Print the label from the predefined list
        if pred_label_idx < len(labels):
            st.write(labels[pred_label_idx])
        else:
            st.write("Unknown label")

        # st.write(f"Predicted Label: {pred_label}")
        # st.write(f"Probabilities: {probs}")
    else:
        st.write("Please enter some text to classify.")




