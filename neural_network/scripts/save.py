import numpy as np
import pandas as pd
import os
import pickle
from .NeuralNetwork import NeuralNetwork
from .Trainer import Trainer

def load_and_preprocess_mnist(train_path, test_path, use_full_dataset=True):
    """Load and preprocess MNIST data"""
    print("Loading MNIST data...")
    
    # Load data
    train_data = pd.read_csv(train_path)
    X_train = train_data.iloc[:, 1:].values
    y_train = train_data.iloc[:, 0].values
    
    test_data = pd.read_csv(test_path)
    X_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values
    
    # Normalize to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    # one-hot encoding
    y_train_one_hot = np.eye(10)[y_train.astype(int)]
    y_test_one_hot = np.eye(10)[y_test.astype(int)]
    
    if not use_full_dataset:
        train_samples = 5000
        test_samples = 1000
        train_indices = np.random.choice(len(X_train), train_samples, replace=False)
        test_indices = np.random.choice(len(X_test), test_samples, replace=False)
        
        X_train = X_train[train_indices]
        y_train_one_hot = y_train_one_hot[train_indices]
        X_test = X_test[test_indices]
        y_test_one_hot = y_test_one_hot[test_indices]
    
    return X_train, y_train_one_hot, X_test, y_test_one_hot

def save_model(model, filepath):
    """Save model architecture and parameters to file"""
    model_data = {
        'architecture': {
            'input_size': model.input_size,
            'hidden_layers': model.hidden_layers,
            'output_size': model.output_size,
            'learning_rate': model.learning_rate,
            'activation': model.layers[0].activation.name,  # same activation for all hidden layers
            'cost_function': model.cost_function_name
        },
        'parameters': model.get_parameters()
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {filepath}")

def train_and_save_model():
    """Train model on full MNIST dataset and save it"""

    train_path = "MNIST_CSV/mnist_train.csv"
    test_path = "MNIST_CSV/mnist_test.csv"
    
    if not os.path.exists(train_path):
        print(f"Error: Training file not found at {train_path}")
        return
    
    if not os.path.exists(test_path):
        print(f"Error: Test file not found at {test_path}")
        return
    
    # use_full_dataset=True for entire dataset
    X_train, y_train, X_test, y_test = load_and_preprocess_mnist(
        train_path, test_path, use_full_dataset=True
    )
    
    print(f"Training on {X_train.shape[0]} samples")
    print(f"Testing on {X_test.shape[0]} samples")
    

    print("\nCreating neural network...")
    model = NeuralNetwork(
        input_size=784,
        hidden_layers=[128, 64],
        output_size=10,
        learning_rate=0.1,
        activation='relu',
        cost_function='cross_entropy'
    )
    

    print("Training model...")
    trainer = Trainer(model, verbose=True)
    history = trainer.train(
        X_train, y_train,
        epochs=20,
        validation_data=(X_test, y_test),
        batch_size=64
    )
    
    train_accuracy = model.compute_accuracy(X_train, y_train)
    test_accuracy = model.compute_accuracy(X_test, y_test)
    
    print(f"\nFinal Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
    
    save_path = "trained_mnist_model.pkl"
    save_model(model, save_path)
    
    print(f"\nModel saved successfully to {save_path}")
    print("You can now use load_demo.py to load and use this model!")

if __name__ == "__main__":
    train_and_save_model()