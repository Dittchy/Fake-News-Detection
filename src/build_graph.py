import pandas as pd
import numpy as np
import torch
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import pickle

# Try importing torch_geometric, if not, we will save standard format
try:
    from torch_geometric.data import Data
    from torch_geometric.utils import from_networkx
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("Warning: torch_geometric not found. Will save graph in NetworkX/Pickle format.")

def build_graph():
    print("Building Graph...")
    
    data_path = "processed_data.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    df = df.dropna(subset=['clean_text'])
    
    # PERFORMANCE FIX: Graphs with O(N^2) edges are very heavy on CPU.
    # We subsample to 5000 recent/random articles to make the GNN train in seconds vs hours.
    # If you have a powerful server, you can increase this number.
    sample_size = 5000
    if len(df) > sample_size:
        print(f"Subsampling dataset from {len(df)} to {sample_size} for graph performance...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    # PERFORMANCE FIX: Graphs with O(N^2) edges are very heavy on CPU.
    # We subsample to 5000 recent/random articles to make the GNN train in seconds vs hours.
    # If you have a powerful server, you can increase this number.
    sample_size = 5000
    if len(df) > sample_size:
        print(f"Subsampling dataset from {len(df)} to {sample_size} for graph performance...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    # Subsample for graph demonstration if dataset is too large (Graphs O(N^2) edges can be heavy)
    # df = df.sample(n=3000, random_state=42).reset_index(drop=True)
    # print("DEBUG: Subsampled to 3000 nodes for graph construction.")

    texts = df['clean_text'].values
    labels = df['label'].values
    
    print("Generating node features (using Pre-trained RoBERTa)...")
    from transformers import AutoTokenizer, AutoModel
    
    model_name = "./bert_model_save/"
    # Fallback if download hasn't happened yet, but we expect it to be there
    if not os.path.exists(model_name):
        model_name = "hamzab/roberta-fake-news-classification" 

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Extract embeddings in batches
    embeddings = []
    batch_size = 32
    
    for i in range(0, len(texts), batch_size):
        batch_texts = list(texts[i:i+batch_size])
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token embedding (index 0)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
            
    embeddings = np.vstack(embeddings)
    
    print("Calculating Cosine Similarity on Embeddings...")
    # Math Magic: This calculates how similar every article is to every other article.
    # Result is a giant table (Matrix) where 1.0 means identical and 0.0 means completely different.
    sim_matrix = cosine_similarity(embeddings)
    
    # Threshold: We only draw a line (Edge) if similarity is very high (> 0.5)
    # This creates the "Network" or "Graph" structure.
    threshold = 0.5
    print(f"Constructing edges (Similarity Threshold > {threshold})...")
    
    # Create NetworkX graph object (Empty container)
    G = nx.Graph()
    
    # Add nodes (One for each article)
    num_nodes = len(df)
    G.add_nodes_from(range(num_nodes))
    
    # Add edges (Connect similar articles)
    # np.where gives us coordinates of all similarity scores > 0.5
    rows, cols = np.where(sim_matrix > threshold)
    
    for r, c in zip(rows, cols):
        if r < c: # Ensure we don't duplicate edges (A->B is same as B->A) and skip user self-loops
            G.add_edge(r, c, weight=sim_matrix[r, c])
            
    print(f"Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    # Prepare features and labels
    node_features = torch.tensor(embeddings, dtype=torch.float)
    node_labels = torch.tensor(labels, dtype=torch.long)
    
    output_dir = "graph_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if PYG_AVAILABLE:
        print("Saving as PyG Data object...")
        # PyG Data
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        data = Data(x=node_features, edge_index=edge_index, y=node_labels)
        
        torch.save(data, os.path.join(output_dir, "graph_data.pt"))
        print(f"Saved to {output_dir}/graph_data.pt")
    else:
        print("Saving as NetworkX/Pickle...")
        with open(os.path.join(output_dir, "graph_networkx.pkl"), 'wb') as f:
            pickle.dump(G, f)
        
        # Save features/labels separately
        torch.save({'x': node_features, 'y': node_labels}, os.path.join(output_dir, "graph_features.pt"))
        print(f"Saved graph and features to {output_dir}/")

if __name__ == "__main__":
    build_graph()
