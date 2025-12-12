import torch
import joblib
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.preprocess import preprocess_text
from src.build_graph import TfidfVectorizer # We need the same vectorizer config
from src.train_gnn import GCN
import os

class FakeNewsEnsemble:
    def __init__(self):
        self.svm_pipeline = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.gnn_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.load_models()

    def load_models(self):
        print("Loading models...")
        
        # 1. Load SVM
        if os.path.exists("baseline_model.pkl"):
            self.svm_pipeline = joblib.load("baseline_model.pkl")
            print("SVM loaded.")
        else:
            print("Warning: baseline_model.pkl not found.")

        # 2. Load RoBERTa
        bert_path = "./bert_model_save/"
        # Check if local exist, else use hub
        if not os.path.exists(bert_path):
             bert_path = "hamzab/roberta-fake-news-classification"

        try:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path)
            self.bert_model.to(self.device)
            self.bert_model.eval()
            print("RoBERTa loaded.")
        except Exception as e:
            print(f"Warning: RoBERTa model not found or error loading: {e}")

        # 3. Load GNN
        if os.path.exists("gnn_model.pth"):
            # RoBERTa base dim is 768
            self.gnn_model = GCN(num_features=768, hidden_channels=64, num_classes=2) 
            self.gnn_model.load_state_dict(torch.load("gnn_model.pth", map_location=self.device, weights_only=True))
            self.gnn_model.to(self.device)
            self.gnn_model.eval()
            print("GNN loaded.")
        else:
            print("Warning: GNN model not found.")

    def predict(self, text):
        clean_text = preprocess_text(text)
        
        probs = {}
        
        # --- SVM Prediction ---
        if self.svm_pipeline:
            svm_prob = self.svm_pipeline.decision_function([clean_text]) # Distance
            # Convert to prob via sigmoid roughly or use CalibratedClassifierCV
            # For LinearSVC, we can just use sign or sigmoid of decision function
            svm_prob = 1 / (1 + np.exp(-svm_prob)) # Sigmoid
            probs['SVM'] = float(svm_prob[0]) # Prob of class 1 (Fake)

        # --- RoBERTa Prediction ---
        if self.bert_model:
            inputs = self.bert_tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=128, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                logits = outputs.logits
                sm = torch.nn.Softmax(dim=1)
                # CRITICAL FIX: Tests show Label 1 = REAL, Label 0 = FAKE for this model.
                # We want Probability of FAKE.
                # So we take index 0 (Fake) OR (1 - index 1).
                bert_prob = sm(logits)[0][0].item() # Prob of class 0 (Fake)
                probs['RoBERTa'] = bert_prob

        # --- GNN Prediction ---
        # We use the GNN as a sophisticated classifier on the embeddings, 
        # treating the new article as an isolated node (inductive inference).
        if self.gnn_model and self.bert_model:
            # 1. Get Embeddings using the loaded RoBERTa model
            # Access the base transformer (usually .roberta or .bert) to get raw embeddings
            base_model = getattr(self.bert_model, 'roberta', getattr(self.bert_model, 'bert', None))
            
            if base_model:
                inputs = self.bert_tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=128, padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = base_model(**inputs)
                    # CLS embedding (batch_size, hidden_dim)
                    node_feature = outputs.last_hidden_state[:, 0, :] 
                
                # 2. Pass to GNN
                # GCNConv handling isolated node: requires edge_index. 
                # We pass an empty edge_index for valid syntax.
                edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)
                
                with torch.no_grad():
                    gnn_logits = self.gnn_model(node_feature, edge_index)
                    gnn_prob = torch.nn.functional.softmax(gnn_logits, dim=1)[0][1].item()
                    probs['GNN'] = gnn_prob
            else:
                 # Fallback if structure varies
                 probs['GNN'] = 0.5


        # --- 4. ENSEMBLE ---
        # We combine the votes from all three experts (Models).
        # We assign weights based on how much we trust each expert.
        # RoBERTa gets 80% because it's the smartest pre-trained model.
        # SVM and GNN get 10% each as supporting opinions.
        weights = {'SVM': 0.1, 'RoBERTa': 0.8, 'GNN': 0.1} 
        
        final_score = 0
        total_weight = 0
        
        print("\n--- Individual Model Predictions (Probability of FAKE) ---")
        for model_name, prob in probs.items():
            print(f"{model_name}: {prob:.4f}") # Print individual scores
            
            # Weighted Math
            w = weights.get(model_name, 0)
            final_score += prob * w
            total_weight += w
            
        # Normalize final score (0 to 1)
        if total_weight > 0:
            final_score /= total_weight
        
        return final_score

if __name__ == "__main__":
    ensemble = FakeNewsEnsemble()
    
    # Test
    sample_text = "Donald Trump sends army to the moon to fight aliens."
    score = ensemble.predict(sample_text)
    print(f"\nFinal Ensemble Score (0=True, 1=Fake): {score:.4f}")
    
    if score > 0.5:
        print("Verdict: FAKE NEWS")
    else:
        print("Verdict: REAL NEWS")
