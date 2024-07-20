# Text Classification with HuggingFace Transformers

This project is a Streamlit application for text classification using a Hugging Face model trained on the SPDRA 2021 dataset. The application allows users to input text and receive a predicted class label.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Acknowledgements](#acknowledgements)

## Introduction
This Streamlit app leverages a BERT-based model to classify input text into various computer science categories. The model and tokenizer are pre-trained and saved using Hugging Face's Transformers library.

## Features
- Load a pre-trained BERT model and tokenizer.
- Classify input text into predefined categories.
- Display the predicted category.

## Installation
To run this application, you need to have Python installed. It is recommended to use a virtual environment to manage dependencies.

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2. **Enter text:**
    - Open your web browser and go to `http://localhost:8501`.
    - Enter the text you want to classify in the provided text area.
    - Click the "Predict" button to see the predicted category.

## Model Details
The model is based on the `sachinsen1295/Bert` pre-trained model and is fine-tuned on the SPDRA 2021 dataset. The categories it predicts include:
- Computation and Language (CL)
- Cryptography and Security (CR)
- Distributed and Cluster Computing (DC)
- Data Structures and Algorithms (DS)
- Logic in Computer Science (LO)
- Networking and Internet Architecture (NI)
- Software Engineering (SE)

## Acknowledgements
- [Hugging Face](https://huggingface.co/) for the Transformers library.
- SPDRA 2021 dataset providers.
- Streamlit for providing a simple way to create web applications.

## Code Overview
### `model_handler.py`
This file contains the `HuggingFaceModelHandler` class for managing the model and tokenizer.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class HuggingFaceModelHandler:
    def __init__(self, modelname='sachinsen1295/Bert', tokenizer_name='sachinsen1295/my-train-models', save_dir="./saved_models"):
        self.modelname = modelname
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        self.tokenizer_name = tokenizer_name
        self.model = None

    def download_and_save_model(self):
        # Download tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.modelname)

        # Save tokenizer and model to directory
        self.tokenizer.save_pretrained(self.save_dir)
        self.model.save_pretrained(self.save_dir)

        print(f"Model and tokenizer saved to {self.save_dir}")

    def load_from_directory(self):
        # Load tokenizer and model from directory
        self.tokenizer = AutoTokenizer.from_pretrained(self.save_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.save_dir)
        
        print(f"Model and tokenizer loaded from {self.save_dir}")

    def predict(self, text):
        # Tokenize the input text and move tensors to the GPU if available
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)

        # Get model output (logits)
        outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        pred_label_idx = torch.argmax(probs, dim=1).item()
        pred_label = self.model.config.id2label[pred_label_idx]

        return probs, pred_label_idx, pred_label
