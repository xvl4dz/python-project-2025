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
drawing_save_path = "current_drawing.npy"

app = Flask(__name__)
CORS(app)

CLASS_LABELS = [str(i) for i in range(10)] 

grid_size = 28
current_drawing = None
current_prediction = None

try:
    model = NeuralNetwork.load_model(model_path)
    print(f"Model loaded from {model_path}")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    model = None

@app.route('/')
def serve_index():
    return send_from_directory('../src', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('../src', path)

@app.route('/api/info', methods=['GET'])
def get_info():
    """Returns server status and model configuration (labels)"""
    return jsonify({
        'status': 'online',
        'grid_size': grid_size,
        'model_loaded': model is not None,
        'class_labels': CLASS_LABELS
    })


@app.route('/api/drawing', methods=['GET'])
def get_drawing():
    """
    PULL: Loads the drawing array from a file and returns it to the client.
    """
    if os.path.exists(drawing_save_path):
        try:
            loaded_drawing = np.load(drawing_save_path)
            
            grid_data_list = loaded_drawing.flatten().tolist()
            
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Drawing loaded from file.")
            
            return jsonify({
                'status': 'success',
                'gridData': grid_data_list,
                'gridSize': loaded_drawing.shape[0] 
            })
        except Exception as e:
            return jsonify({'error': f'Could not load drawing file: {str(e)}'}), 500
    else:
        return jsonify({'error': 'No saved drawing found.'}), 404


@app.route('/api/save', methods=['POST'])
def save_drawing():
    """
    PUSH: Saves the current drawing data (`current_drawing` global) to a file.
    """
    global current_drawing
    
    if current_drawing is not None:
        try:
            np.save(drawing_save_path, current_drawing)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Drawing saved to {drawing_save_path}")
            return jsonify({'status': 'success', 'message': 'Drawing saved successfully.'})
        except Exception as e:
            return jsonify({'error': f'Failed to save drawing: {str(e)}'}), 500
    else:
        return jsonify({'error': 'No current drawing data to save.'}), 400


@app.route('/api/update', methods=['POST'])
def update_drawing():
    global current_drawing
    global current_prediction
    
    try:
        data = request.get_json()
        
        if not data or 'gridData' not in data or 'gridSize' not in data:
            return jsonify({'error': 'Missing gridData or gridSize'}), 400
        
        input_grid_size = data['gridSize']
        grid_data = data['gridData']
        
        np_array = np.array(grid_data, dtype=np.uint8)
        
        if len(np_array.shape) == 1:
            np_array = np_array.reshape((input_grid_size, input_grid_size))
        
        current_drawing = np_array
        
        prediction_result = None
        confidence = 0.0
        probabilities = []

        if model:
            normalized_arr = np_array.astype(np.float32) / 255.0
            flat_arr = normalized_arr.flatten().reshape(1, -1)
            
            nn_return = model.predict(flat_arr)[0]
            probabilities = nn_return.tolist()
            
            pred_index = int(nn_return.argmax())
            
            if pred_index < len(CLASS_LABELS):
                prediction_result = CLASS_LABELS[pred_index]
            else:
                prediction_result = str(pred_index)
                
            confidence = float(nn_return[pred_index])
            
            print(f"Prediction: {prediction_result} ({confidence*100:.1f}%)")

        return jsonify({
            'status': 'success',
            'prediction': prediction_result,
            'confidence': confidence,
            'probabilities': probabilities,
            'labels': CLASS_LABELS
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_drawing():
    global current_drawing
    current_drawing = None
    return jsonify({'status': 'cleared'})


if __name__ == '__main__':
    print(f"Starting Server on port 5000...")
    print(f"Model Labels: {CLASS_LABELS}")
    print("\nAvailable endpoints:")
    print("   GET  /api/drawing      - PULL: Get last saved drawing from file")
    print("   POST /api/update       - Update drawing from client & run prediction")
    print("   POST /api/clear        - Clear current drawing in memory")
    print("   POST /api/save         - PUSH: Save current drawing in memory to file")
    print("   GET  /api/info         - Server information & labels")
    app.run(host='0.0.0.0', port=5000, debug=True)
