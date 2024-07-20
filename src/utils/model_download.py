from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class HuggingFaceModelHandler:
    def __init__(self,modelname='sachinsen1295/Bert',tokenizer_name='sachinsen1295/my-train-models',save_dir="./saved_models"):
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
        """
        Predicts the class label for a given input text

        Args:
            text (str): The input text for which the class label needs to be predicted.

        Returns:
            probs (torch.Tensor): Class probabilities for the input text.
            pred_label_idx (int): The index of the predicted class label.
            pred_label (str): The predicted class label.
        """
        # Tokenize the input text and move tensors to the GPU if available
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)

        # Get model output (logits)
        outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        pred_label_idx = torch.argmax(probs, dim=1).item()
        pred_label = self.model.config.id2label[pred_label_idx]

        return probs, pred_label_idx, pred_label


    # def predict(self, text):
    #     # Example usage: tokenize and encode text
    #     inputs = self.tokenizer(text, return_tensors="pt")
    #     outputs = self.model(**inputs)
        
    #     return outputs.logits
    

    # Example usage
# if __name__ == "__main__":
#     model_handler = HuggingFaceModelHandler()
    
#     # Download and save the model
#     model_handler.download_and_save_model()
    
#     # Load the model from the saved directory
#     model_handler.load_from_directory()
    
#     # Example usage
#     text = "Hello, world!"
#     logits = model_handler.predict(text)
    
#     print("Example logits:", logits)

    