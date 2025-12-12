from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def debug_roberta():
    # If using local path, ensure it validates or use the hub name directly if local fails to load
    # For now, let's use the hub name directly to be sure of the mapping
    import os
    if os.path.exists("./bert_model_save"):
        model_name = "./bert_model_save"
    else:
        model_name = "hamzab/roberta-fake-news-classification"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # 1. Known REAL text (from standard datasets)
    # "The U.S. Congress sits in Washington D.C."
    real_text = "WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress, who voted this month for a massive expansion of the national debt to pay for tax cuts, called himself a “fiscal conservative” on Sunday and urged budget restraint in 2018."
    
    # 2. Known FAKE text
    # "Aliens found in New York."
    fake_text = "Donald Trump sends army to the moon to fight aliens. The white house confirmed that martians have invaded texas."
    
    inputs_real = tokenizer(real_text, return_tensors="pt")
    inputs_fake = tokenizer(fake_text, return_tensors="pt")
    
    with torch.no_grad():
        logits_real = model(**inputs_real).logits
        logits_fake = model(**inputs_fake).logits
        
    probs_real = torch.softmax(logits_real, dim=1).tolist()[0]
    probs_fake = torch.softmax(logits_fake, dim=1).tolist()[0]
    
    print(f"Test REAL Text: {probs_real}")
    print(f"Test FAKE Text: {probs_fake}")
    
    print("\n--- Interpretation ---")
    print(f"Label 0 Prob (Real Text): {probs_real[0]:.4f}")
    print(f"Label 1 Prob (Real Text): {probs_real[1]:.4f}")
    
    print(f"Label 0 Prob (Fake Text): {probs_fake[0]:.4f}")
    print(f"Label 1 Prob (Fake Text): {probs_fake[1]:.4f}")

if __name__ == "__main__":
    debug_roberta()
