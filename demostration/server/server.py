from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import json
import os
import sys
import pickle
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural_network import NeuralNetwork
model_path = "trained_mnist_model.pkl"

def load_model(filepath):
    """Load model from file"""
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    arch = model_data['architecture']
    parameters = model_data['parameters']
    
    model = NeuralNetwork(
        input_size=arch['input_size'],
        hidden_layers=arch['hidden_layers'],
        output_size=arch['output_size'],
        learning_rate=arch['learning_rate'],
        activation=arch['activation'],
        cost_function=arch['cost_function']
    )
    
    model.set_parameters(parameters)
    
    print(f"Model loaded from {filepath}")
    print(f"Architecture: {arch['input_size']} -> {arch['hidden_layers']} -> {arch['output_size']}")
    
    return model

model = load_model(model_path)

app = Flask(__name__)
CORS(app)

current_drawing = None
grid_size = 28
current_prediction = None

@app.route('/')
def serve_index():
    return send_from_directory('../src', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('../src', path)

@app.route('/api/update', methods=['POST'])
def update_drawing():
    global current_drawing
    global current_prediction
    global model
    
    try:
        data = request.get_json()
        
        if not data or 'gridData' not in data or 'gridSize' not in data:
            return jsonify({'error': 'Missing gridData or gridSize'}), 400
        
        grid_data = data['gridData']
        grid_size = data['gridSize']
        
        np_array = np.array(grid_data, dtype=np.uint8)
        
        if len(np_array.shape) == 1:
            np_array = np_array.reshape((grid_size, grid_size))
        elif len(np_array.shape) == 2:
            if np_array.shape != (grid_size, grid_size):
                return jsonify({'error': f'Array shape {np_array.shape} does not match grid size {grid_size}'}), 400
        else:
            return jsonify({'error': 'Invalid array dimensions'}), 400
        
        current_drawing = np_array
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Drawing updated - Shape: {np_array.shape}, Mean: {np_array.mean():.2f}")
        

        normalized_arr = np_array.astype(np.float32) / 255.0
        flat_arr = normalized_arr.flatten().reshape(1, -1)  # Shape: (1, 784)
        
        nn_return = model.predict(flat_arr)[0]  # Shape: (10,)
        current_prediction = nn_return.argmax()
        confidence = nn_return[current_prediction]
        
        print('-----------------------------')
        print(f'--- Prediction: {current_prediction}, Confidence: {100*confidence:.1f}')
        print(f'--- All probabilities: {np.round(nn_return, 3)}%')
        
        return jsonify({
            'status': 'success',
            'message': f'Drawing updated - Shape: {np_array.shape}',
            'prediction': int(current_prediction),
            'confidence': float(confidence),
            'probabilities': nn_return.tolist(),
            'stats': {
                'mean': float(np_array.mean()),
                'std': float(np_array.std()),
                'min': int(np_array.min()),
                'max': int(np_array.max()),
            }
        })
    
    except Exception as e:
        print(f"Error updating drawing: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting MNIST Drawing Server...")
    print("Server will be available at: http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  GET  /api/drawing     - Get current drawing")
    print("  POST /api/update      - Update drawing from client")
    print("  POST /api/clear       - Clear current drawing")
    print("  POST /api/save        - Save drawing as .npy file")
    print("  POST /api/load_example - Load example pattern")
    print("  GET  /api/info        - Server information")
    
    app.run(host='0.0.0.0', port=5000, debug=True)