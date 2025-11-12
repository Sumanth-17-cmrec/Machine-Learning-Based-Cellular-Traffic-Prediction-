import torch
import torch.nn.functional as F
import pickle
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# Load the standard scaler
with open("standard_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load the trained model
class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GNNModel, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, output_dim)
        self.dropout = dropout
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

# Define input features
features = ["Longitude", "Latitude", "Speed", "RSRP", "RSRQ", "SNR", "CQI", "RSSI", "DL_bitrate", "UL_bitrate", "NRxRSRP", "NRxRSRQ", "ServingCell_Distance"]
input_dim = len(features)
hidden_dim = 128
output_dim = 5  # Change according to the trained model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNNModel(input_dim, hidden_dim, output_dim).to(device)
model.load_state_dict(torch.load("gnn_model.pth", map_location=device))
model.eval()

# Sample input for prediction (modify accordingly)
sample_input = np.array([[-8.471603,51.900334,0,-101,-10,9,12,-81,23642,496,-103,-14,1177.67]])
scaled_input = scaler.transform(sample_input)

# Convert input to torch tensor
node_features = torch.tensor(scaled_input, dtype=torch.float).to(device)
edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)  # Dummy self-loop edge

data = Data(x=node_features, edge_index=edge_index).to(device)

# Make prediction
with torch.no_grad():
    output = model(data)
    prediction = output.argmax(dim=1).item()

print(f"Predicted class: {prediction}")