# app.py

import streamlit as st

# Set the title of the app
st.title("Simple Streamlit App")

# Create a text input widget
user_input = st.text_input("Enter some text:")

# Display the entered text
if user_input:
    st.write("You entered:", user_input)