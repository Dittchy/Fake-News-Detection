# implementation_plan.md

## Project Pipeline: Hybrid Fake News Detection System

Based on the synopsis, here is the roadmap to build the Fake News Detector using NLP and GNNs.

### Phase 1: Environment & Data Preparation
- [ ] **Environment Setup**:
    - Install Python 3.8+
    - Install libraries: `pandas`, `numpy`, `scikit-learn`, `nltk`, `spacy`, `torch`, `transformers` (Hugging Face), `networkx`, `torch_geometric` (or `dgl`), `streamlit`, `shap`.
- [ ] **Data Acquisition**:
    - Download **FakeNewsNet** (PolitiFact / GossipCop) or similar dataset (e.g., ISOT, LIAR).
    - The dataset should ideally contain:
        - **News Content**: Text of the articles.
        - **Social Context**: User-article interactions (tweets, shares) for GNN construction.
    - *Note*: If full social graph data is unavailable, we may simulate the graph or use a dataset with existing relations.

### Phase 2: Data Preprocessing
- [ ] **Text Preprocessing**:
    - Cleaning: Remove URLs, special characters.
    - Tokenization & Lemmatization (using `nltk` or `spacy`).
    - Stopword removal.
- [ ] **Feature Engineering (Baseline)**:
    - TF-IDF Vectorization.
- [ ] **Graph Construction (For GNN)**:
    - Nodes: News Articles (and optionally Users/Sources).
    - Edges: Shared by same user, similar content, or followed-by relations.
    - Node Features: Initial embeddings using BERT.

### Phase 3: Model Development (Hybrid Ensemble)
- [ ] **Model A: Baseline (SVM)**: (Completed)
    - TF-IDF + LinearSVC.
- [ ] **Model B: BERT Classifier**:
    - Fine-tune `bert-base-uncased` for binary classification.
    - Save trained model and use it to extract embeddings for the GNN.
- [ ] **Model C: GNN (Graph Neural Network)**:
    - **Graph Construction**:
        - *Option A (Ideal)*: Use FakeNewsNet (if provided) with social context.
        - *Option B (Current Plan)*: Construct a **Content Similarity Graph** (KNN) using TF-IDF/BERT tokens from the `Fake.csv`/`True.csv` dataset.
    - **Architecture**: GCN or GAT using BERT embeddings as node features.
- [ ] **Ensemble Module**:
    - Combine predictions/probabilities from SVM, BERT, and GNN.
    - Weighted average or Meta-Classifier (Logistic Regression).

### Phase 4: Evaluation & Interpretation
- [ ] **Evaluation**:
    - Comparative metrics (Accuracy, F1) for SVM, BERT, GNN, and Ensemble.
- [ ] **Explainability (XAI)**:
    - **Text Explainability**: SHAP on BERT/SVM to highlight keywords.
    - **Graph Explainability**: GNNExplainer (if applicable) or node importance.

### Phase 5: Deployment
- [ ] **Streamlit Dashboard**:
    - Input field for News URL or Text.
    - Display prediction (Real/Fake).
    - Display Confidence Score.
    - Visualizations: SHAP summary plots, maybe a small graph visualization.

---

## Next Steps
1. Download the dataset.
2. Set up the python environment.
3. Start with Preprocessing script.
