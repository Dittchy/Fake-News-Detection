# Fake News Detector using NLP & GNN

This project employs a **Hybrid Ensemble Architecture** to detect fake news. It combines traditional Machine Learning (SVM), Deep Learning (BERT), and Graph Neural Networks (GNN) to analyze both the content and context of simulated news propagation.

## Architecture
1.  **Baseline Model**: Linear SVC on TF-IDF features.
2.  **Text Model**: Fine-tuned **BERT** (`bert-base-uncased`) for semantic understanding.
3.  **Graph Model**: **GCN** (Graph Convolutional Network) trained on a Content Similarity Graph.
4.  **Ensemble**: Methods are combined (Weighted Average) for final prediction.
5.  **Explainability**: **SHAP** is used to explain which words contributed to the decision.

## Prerequisities
- Python 3.8+
- GPU recommended for BERT/GNN training.

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install shap transformers networkx scikit-learn pandas numpy
```

## How to Run

### Phase 1: Data Setup
1.  **Download Data**:
    ```bash
    python download_data.py
    ```
2.  **Preprocess**:
    ```bash
    python src/preprocess.py
    ```
    *Output*: `processed_data.csv`

### Phase 2: Model Setup & Training
3.  **Download Pre-trained Models**:
    ```bash
    python src/download_models.py
    ```
    *Output*: `bert_model_save/` containing RoBERTa weights.

4.  **Train Baseline (SVM)**:
    ```bash
    python src/train_baseline.py
    ```
    *Output*: `baseline_model.pkl`

5.  **Build Graph (using RoBERTa embeddings)**:
    ```bash
    python src/build_graph.py
    ```
    *Output*: `graph_data/` (Nodes=Articles, Features=RoBERTa Embeddings, Edges=Cosine Similarity)

6.  **Train GNN**:
    ```bash
    python src/train_gnn.py
    ```
    *Output*: `gnn_model.pth`

### Phase 3: Inference & Explanation
7.  **Run Inference**:
    ```bash
    python src/inference.py
    ```
    edit the `sample_text` in the script to test different inputs.

8.  **Explain Prediction**:
    ```bash
    python src/explain.py
    ```
    *Output*: `shap_explanation.html`

### Phase 4: Deployment (User Interface)
9.  **Run Streamlit App**:
    ```bash
    streamlit run src/app.py
    ```
    This will open a web interface in your browser where you can paste text and see results interactively.

## Project Structure
- `src/preprocess.py`: data cleaning.
- `src/train_baseline.py`: SVM model.
- `src/train_bert.py`: BERT fine-tuning.
- `src/build_graph.py`: Constructing Similarity Graph from text.
- `src/train_gnn.py`: Training GCN/GAT.
- `src/inference.py`: Ensemble logic.
- `src/explain.py`: SHAP visualization.
