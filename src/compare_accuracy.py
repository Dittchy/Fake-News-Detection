import pandas as pd
import numpy as np
import joblib
import torch
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocess import preprocess_text
from tqdm import tqdm
import os

def compare_models():
    print("Loading Data...")
    if not os.path.exists("processed_data.csv"):
        print("processed_data.csv not found. Run src/preprocess.py first.")
        return
    
    df = pd.read_csv("processed_data.csv")
    df = df.dropna(subset=['clean_text'])
    
    # Use a smaller sample for speed if CPU
    # df = df.sample(n=1000, random_state=42)
    # print("DEBUG: Using 1000 samples for comparison speed.")

    y_true = df['label'].values
    texts = df['clean_text'].values
    
    # --- 1. Evaluate TF-IDF Baseline ---
    print("\n--- Evaluating TF-IDF + SVM ---")
    if os.path.exists("baseline_model.pkl"):
        svm_pipeline = joblib.load("baseline_model.pkl")
        y_pred_svm = svm_pipeline.predict(texts)
        acc_svm = accuracy_score(y_true, y_pred_svm)
        print(f"TF-IDF Accuracy: {acc_svm:.4f}")
    else:
        print("baseline_model.pkl not found. Run src/train_baseline.py.")
        acc_svm = None

    # --- 2. Evaluate RoBERTa ---
    print("\n--- Evaluating RoBERTa ---")
    bert_path = "./bert_model_save/"
    if not os.path.exists(bert_path):
        bert_path = "hamzab/roberta-fake-news-classification"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(bert_path)
        model = AutoModelForSequenceClassification.from_pretrained(bert_path)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        y_pred_bert = []
        batch_size = 32
        
        print(f"Running inference on {len(texts)} samples (Device: {device})...")
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = list(texts[i:i+batch_size])
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred_bert.extend(preds)
        
        acc_bert = accuracy_score(y_true, y_pred_bert)
        print(f"RoBERTa Accuracy: {acc_bert:.4f}")
        
    except Exception as e:
        print(f"Error evaluating RoBERTa: {e}")
        acc_bert = None

    # --- Summary ---
    print("\n======== COMPARISON RESULTS ========")
    if acc_svm: print(f"TF-IDF Model: {acc_svm*100:.2f}%")
    if acc_bert: print(f"RoBERTa Model: {acc_bert*100:.2f}%")
    
    if acc_svm and acc_bert:
        diff = acc_bert - acc_svm
        if diff > 0:
            print(f"Result: RoBERTa is {diff*100:.2f}% more accurate.")
        else:
            print(f"Result: TF-IDF is {-diff*100:.2f}% more accurate.")

if __name__ == "__main__":
    compare_models()
