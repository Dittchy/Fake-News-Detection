import shap
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import matplotlib.pyplot as plt

def explain_prediction(text):
    print("Initializing SHAP Explainer...")
    
    bert_path = "./bert_model_save/"
    if not os.path.exists(bert_path):
        bert_path = "hamzab/roberta-fake-news-classification"

    # Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    model = AutoModelForSequenceClassification.from_pretrained(bert_path)
    model.eval()
    
    # Wrapper function for SHAP
    def predictor(texts):
        # Determine if texts is a list or single string
        # SHAP often passes numpy arrays of strings, we need to convert them.
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, np.ndarray):
            texts = texts.tolist()
            
        # Ensure it's a list of strings (remove potential None values)
        texts = [str(t) for t in texts]
            
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).numpy()
        return probs

    # --- 3. CREATE EXPLAINER ---
    # SHAP (SHapley Additive exPlanations) needs a "Masker" to hide parts of text 
    # and see how much the prediction changes.
    masker = shap.maskers.Text(tokenizer)
    
    # The Explainer orchestrates the logic: "What if I remove word X? Does the Fake Score drop?"
    # CRITICAL FIX: Model outputs [Fake, Real].
    # So index 0 is Fake, index 1 is Real.
    # We want to explain the "Fake" class (Index 0).
    explainer = shap.Explainer(predictor, masker, output_names=["Fake", "Real"])
    
    print(f"Explaining text: {text[:50]}...")
    
    # Calculate SHAP values (This might take a few seconds)
    shap_values = explainer([text])
    
    # --- 4. VISUALIZE ---
    # We save the colorful text plot as an HTML file.
    # Red words = Pushed model towards prediction (mostly FAKE if score > 0.5).
    # Blue words = Pushed model away.
    # We specifically want to visualize the attribution for the "Fake" class (Index 0).
    output_html = "shap_explanation.html"
    with open(output_html, 'w', encoding='utf-8') as f:
        # shap_values[..., 0] gives us values for the "Fake" class
        f.write(shap.plots.text(shap_values[..., 0], display=False))
        
    print(f"Explanation saved to {output_html}")
    
    # Also print top features
    # This is rough as shap_values structure is complex
    # print(shap_values)

if __name__ == "__main__":
    sample_text = "The government is hiding aliens in Area 51 and the president knows about it."
    explain_prediction(sample_text)
