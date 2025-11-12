from flask import Flask, render_template, request, redirect, url_for
import torch
import torch.nn.functional as F
import pickle
import numpy as np
import sqlite3
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the standard scaler
with open("model/standard_scaler.pkl", "rb") as f:
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

# Model parameters
features = ["Longitude", "Latitude", "Speed", "RSRP", "RSRQ", "SNR", "CQI", "RSSI", "DL_bitrate", "UL_bitrate", "NRxRSRP", "NRxRSRQ", "ServingCell_Distance"]
input_dim = len(features)
hidden_dim = 128
output_dim = 5  # Adjust as per trained model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNNModel(input_dim, hidden_dim, output_dim).to(device)
model.load_state_dict(torch.load("model/gnn_model.pth", map_location=device))
model.eval()

# Database setup
DATABASE = 'predictions.db'

def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_input TEXT,
            predicted_class INTEGER
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Label encoding mapping
label_mapping = {0: "bus", 1: "car", 2: "pedestrian", 3: "static", 4: "train"}

@app.route('/')
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_text = None
    if request.method == 'POST':
        user_input = request.form['input_data']
        input_values = np.array([list(map(float, user_input.split(',')))])
        scaled_input = scaler.transform(input_values)

        # Convert input to torch tensor
        node_features = torch.tensor(scaled_input, dtype=torch.float).to(device)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)

        data = Data(x=node_features, edge_index=edge_index).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(data)
            predicted_class = output.argmax(dim=1).item()

        predicted_label = label_mapping[predicted_class]

        # Store in database
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("INSERT INTO predictions (user_input, predicted_class) VALUES (?, ?)", 
                  (user_input, predicted_class))
        conn.commit()
        conn.close()

        prediction_text = f"Predicted Class: {predicted_label}"

    return render_template('index.html', prediction_text=prediction_text)

@app.route('/predictions')
def predictions():
    page = request.args.get('page', 1, type=int)
    per_page = 5  # Number of records per page

    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM predictions")
    total_records = c.fetchone()[0]

    offset = (page - 1) * per_page
    c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ? OFFSET ?", (per_page, offset))
    records = c.fetchall()
    conn.close()

    total_pages = (total_records + per_page - 1) // per_page

    return render_template('predictions.html', records=records, total_pages=total_pages, current_page=page)

@app.route('/graphs')
def graphs():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT predicted_class FROM predictions")
    predictions = [row[0] for row in c.fetchall()]
    conn.close()

    if not predictions:
        return render_template('graphs.html', message="No data available for graphs.")

    # Count occurrences
    counts = {label_mapping[i]: predictions.count(i) for i in range(output_dim)}

    # Generate Pie Chart
    plt.figure(figsize=(5, 5))
    plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', startangle=140)
    plt.title("Prediction Distribution")
    pie_img = io.BytesIO()
    plt.savefig(pie_img, format='png')
    pie_img.seek(0)
    pie_url = base64.b64encode(pie_img.getvalue()).decode('utf8')

    # Generate Bar Chart
    plt.figure(figsize=(6, 4))
    plt.bar(counts.keys(), counts.values())
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Prediction Counts")
    bar_img = io.BytesIO()
    plt.savefig(bar_img, format='png')
    bar_img.seek(0)
    bar_url = base64.b64encode(bar_img.getvalue()).decode('utf8')

    # Generate Line Chart
    plt.figure(figsize=(6, 4))
    plt.plot(list(counts.keys()), list(counts.values()), marker='o')
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Prediction Trend")
    line_img = io.BytesIO()
    plt.savefig(line_img, format='png')
    line_img.seek(0)
    line_url = base64.b64encode(line_img.getvalue()).decode('utf8')

    return render_template('graphs.html', pie_url=pie_url, bar_url=bar_url, line_url=line_url)

if __name__ == '__main__':
    app.run(debug=True)
