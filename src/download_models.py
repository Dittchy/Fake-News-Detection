from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

def download_model():
    # --- 1. SETUP ---
    # We are using a model called "RoBERTa". It's like BERT, but trained longer and smarter.
    # This specific version was already fine-tuned by someone else to understand Fake News.
    model_name = "hamzab/roberta-fake-news-classification"
    save_directory = "./bert_model_save/"
    
    print(f"Downloading pre-trained model: {model_name}...")
    
    # --- 2. DOWNLOAD ---
    # Tokenizer: Converts words into numbers the AI understands.
    # Model: The actual neural network weights.
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    print(f"Saving to {save_directory}...")
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)
    
    print("Model downloaded successfully.")

if __name__ == "__main__":
    download_model()
