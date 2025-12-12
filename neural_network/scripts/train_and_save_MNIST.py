import numpy as np
import pandas as pd
import os
import pickle

from NeuralNetwork import NeuralNetwork
from Trainer import Trainer

def apply_transformations(images, params):
    """
    Applies random rotation, scaling, and translation using NumPy 
    with bilinear interpolation.
    
    Args:
        images: Array of shape (N, 784)
        params: Dict containing augmentation parameters
        
    Returns:
        Augmented images of shape (N, 784)
    """
    #Lots of comments here or I'll forget what all of this does ;)

    N, D = images.shape
    H = W = int(np.sqrt(D))
    
    # Reshape to (N, H, W) for geometric ops
    imgs = images.reshape(N, H, W)
    
    # 1. Generate Random Parameters
    angles = np.radians(np.random.uniform(-params['phi'], params['phi'], N))
    scales = np.random.uniform(1.0 - params['scale'], 1.0 + params['scale'], N)
    tx = np.random.uniform(-params['trans_x'], params['trans_x'], N)
    ty = np.random.uniform(-params['trans_y'], params['trans_y'], N)
    
    # 2. Create Grid (Target Coordinates)
    # Grid centered at (0,0) for easier rotation
    x = np.linspace(-W/2, W/2, W)
    y = np.linspace(-H/2, H/2, H)
    xv, yv = np.meshgrid(x, y) # Shapes (H, W)
    
    # Expand dims for broadcasting: (1, H, W)
    xv = xv[np.newaxis, :, :]
    yv = yv[np.newaxis, :, :]
    
    # 3. Inverse Mapping (Target -> Source)
    # To find the pixel value at (x,y) in the output, we find where it came from in the input.
    # Inverse rotation formula + Inverse Scale + Translation
    
    # x_src = (x*cos - y*sin) / scale - shift_x
    # y_src = (x*sin + y*cos) / scale - shift_y
    
    # Reshape params for broadcasting (N, 1, 1)
    cos_a = np.cos(angles)[:, None, None]
    sin_a = np.sin(angles)[:, None, None]
    s = scales[:, None, None]
    tx = tx[:, None, None]
    ty = ty[:, None, None]
    
    # Apply Inverse Transform
    x_src = (xv * cos_a + yv * sin_a) / s - tx
    y_src = (-xv * sin_a + yv * cos_a) / s - ty
    
    # Shift back to image coordinates (0 to 27)
    x_src += W/2
    y_src += H/2
    
    # 4. Bilinear Interpolation
    # Get integer coordinates for the 4 corners
    x0 = np.floor(x_src).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y_src).astype(int)
    y1 = y0 + 1
    
    # Clip coordinates to stay inside image boundaries
    x0 = np.clip(x0, 0, W-1)
    x1 = np.clip(x1, 0, W-1)
    y0 = np.clip(y0, 0, H-1)
    y1 = np.clip(y1, 0, H-1)
    
    # Helper to gather pixels
    # usage: imgs[image_index, y, x]
    b_idx = np.arange(N)[:, None, None] # Batch indices broadcastable
    
    Ia = imgs[b_idx, y0, x0] # Top-left
    Ib = imgs[b_idx, y1, x0] # Bottom-left
    Ic = imgs[b_idx, y0, x1] # Top-right
    Id = imgs[b_idx, y1, x1] # Bottom-right
    
    # Calculate weights based on distance from floor coordinate
    wa = (x1 - x_src) * (y1 - y_src)
    wb = (x1 - x_src) * (y_src - y0)
    wc = (x_src - x0) * (y1 - y_src)
    wd = (x_src - x0) * (y_src - y0)
    
    # Sum weighted values
    aug_imgs = wa*Ia + wb*Ib + wc*Ic + wd*Id
    
    # 5. Add Noise
    if params['noise_std'] > 0:
        noise = np.random.normal(0, params['noise_std'], aug_imgs.shape)
        aug_imgs = aug_imgs + noise
        
    # Clip final result to valid range
    aug_imgs = np.clip(aug_imgs, 0.0, 1.0)
    
    return aug_imgs.reshape(N, D)

def load_and_preprocess_mnist(train_path, test_path, use_full_dataset=True, augmentation_config=None):
    """
    Load and preprocess MNIST data with optional augmentation.
    
    Args:
        augmentation_config (dict): Dict with keys 'phi', 'scale', 'trans_x', 'trans_y', 'noise_std'
    """
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
    
    # One-hot encoding
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

    if augmentation_config:
        print("Applying data augmentation...")
        defaults = {
            'phi': 0,           # Max rotation angle in degrees
            'scale': 0,         # Max scale factor deviation (e.g. 0.1 for 0.9-1.1)
            'trans_x': 0,       # Max translation in X pixels
            'trans_y': 0,       # Max translation in Y pixels
            'noise_std': 0      # Gaussian noise standard deviation
        }

        params = {**defaults, **augmentation_config}
        
        X_train = apply_transformations(X_train, params)
        X_test = apply_transformations(X_test, params)
    
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
    aug_params = {
        'phi': 15,
        'scale': 0.1,
        'trans_x': 2.0,
        'trans_y': 2.0,
        'noise_std': 0.05
    }
    X_train, y_train, X_test, y_test = load_and_preprocess_mnist(
        train_path, 
        test_path, 
        use_full_dataset=True, 
        augmentation_config=aug_params
    )
    
    print(f"Training on {X_train.shape[0]} samples")
    print(f"Testing on {X_test.shape[0]} samples")
    

    print("\nCreating neural network...")
    model = NeuralNetwork(
        input_size=784,
        hidden_layers=[128, 128, 128],
        output_size=10,
        learning_rate=0.1,
        activation='relu',
        cost_function='cross_entropy'
    )
    

    print("Training model...")
    trainer = Trainer(model, verbose=True)
    history = trainer.train(
        X_train, y_train,
        epochs=250,
        validation_data=(X_test, y_test),
        batch_size=16
    )
    
    train_accuracy = model.compute_accuracy(X_train, y_train)
    test_accuracy = model.compute_accuracy(X_test, y_test)
    
    print(f"\nFinal Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
    
    save_path = "trained_mnist_model.pkl"
    save_model(model, save_path)
    
    print(f"\nModel saved successfully to {save_path}")

if __name__ == "__main__":
    train_and_save_model()