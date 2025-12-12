import torch
import torch.nn.functional as F
import os

# Check for PyG
try:
    from torch_geometric.nn import GCNConv, GATConv
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("Error: torch_geometric not installed. GNN training requires it.")

# --- 2. GNN MODEL DEFINITION ---
# This is the "Brain" of our Graph Neural Network.
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        # Layer 1: Takes raw features and passes messages to immediate neighbors
        self.conv1 = GCNConv(num_features, hidden_channels)
        # Layer 2: Takes the result and passes messages again (Friends of Friends)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # x: Node features (Text contents)
        # edge_index: Graph connections (Who is similar to whom)
        
        # Pass through Layer 1
        x = self.conv1(x, edge_index)
        x = x.relu() # Activation function (adds non-linearity)
        
        # Dropout: Randomly forget some info to prevent overfitting (memoizing)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Pass through Layer 2 (Output Layer)
        x = self.conv2(x, edge_index)
        return x

def train_gnn_model():
    if not PYG_AVAILABLE:
        return

    print("Loading Graph Data...")
    data_path = "graph_data/graph_data.pt"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run src/build_graph.py first.")
        return
    
    data = torch.load(data_path, weights_only=False)
    
    # Basic info
    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.num_edges}")
    print(f"Features: {data.num_features}")
    
    # --- 3. TRAIN/TEST SPLIT ---
    # We hide 20% of the nodes (Mask them out) to test the model later.
    # It's like giving a student a practice exam (80%) and then a final exam (20%).
    if not hasattr(data, 'train_mask'):
        print("Creating train/test masks...")
        num_nodes = data.num_nodes
        indices = torch.randperm(num_nodes) # Shuffle
        
        train_size = int(0.8 * num_nodes)
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]
        
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[train_idx] = True
        
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask[test_idx] = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_features=data.num_features, hidden_channels=64, num_classes=2).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    print("Training GNN...")
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            pred = out.argmax(dim=1)
            correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
            acc = int(correct) / int(data.train_mask.sum())
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {acc:.4f}')

    print("Evaluating...")
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Test Accuracy: {acc:.4f}')

    # --- 6. SAVE MODEL ---
    # We save the "brain state" (learned weights) to a file.
    model_path = "gnn_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"GNN Model saved to {model_path}")

if __name__ == "__main__":
    train_gnn_model()
