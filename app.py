from flask import Flask, render_template, request, jsonify
import pandas as pd
import networkx as nx
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})

    try:
        # Read data
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.json'):
            df = pd.read_json(file)
        else:
            return jsonify({'success': False, 'error': 'Unsupported file format'})

        # Build network graph
        G = build_transaction_network(df)
        
        # Perform anomaly detection
        anomalies = detect_anomalies(G)
        
        # Prepare visualization data
        graph_data = prepare_visualization_data(G, anomalies)
        
        return jsonify({
            'success': True,
            'graph_data': graph_data
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def build_transaction_network(df):
    # Implement network building logic here
    G = nx.Graph()
    # ... network building code ...
    return G

def detect_anomalies(G):
    # Implement anomaly detection logic here
    anomalies = []
    # ... anomaly detection code ...
    return anomalies

def prepare_visualization_data(G, anomalies):
    # Convert to ECharts compatible format
    nodes = [{'name': str(node), 'value': G.degree(node)} for node in G.nodes()]
    links = [{'source': str(edge[0]), 'target': str(edge[1])} for edge in G.edges()]
    
    return {
        'nodes': nodes,
        'links': links
    }

if __name__ == '__main__':
    app.run(debug=True) 