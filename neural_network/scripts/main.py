import numpy as np
import pandas as pd
from .NeuralNetwork import NeuralNetwork
from .Trainer import Trainer
from .Visualizer import Visualizer
import os

def load_mnist_data(train_path, test_path):
    """Load MNIST data from CSV files"""
    print("Loading MNIST data...")
    
    # Load data
    train_data = pd.read_csv(train_path)
    X_train = train_data.iloc[:, 1:].values
    y_train = train_data.iloc[:, 0].values
    
    test_data = pd.read_csv(test_path)
    X_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test

def preprocess_mnist_data(X_train, y_train, X_test, y_test):
    """Preprocess MNIST data"""
    print("Preprocessing data...")
    
    # Normalize to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    # one-hot encoding
    y_train_one_hot = np.eye(10)[y_train.astype(int)]
    y_test_one_hot = np.eye(10)[y_test.astype(int)]
    
    print(f"After preprocessing:")
    print(f"X_train range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"X_test range: [{X_test.min():.3f}, {X_test.max():.3f}]")
    print(f"y_train one-hot shape: {y_train_one_hot.shape}")
    print(f"y_test one-hot shape: {y_test_one_hot.shape}")
    
    return X_train, y_train_one_hot, X_test, y_test_one_hot

def create_small_subset(X_train, y_train, X_test, y_test, train_samples=10000, test_samples=2000):
    """Create smaller subset for faster testing"""
    if train_samples < len(X_train):
        indices = np.random.choice(len(X_train), train_samples, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
    
    if test_samples < len(X_test):
        indices = np.random.choice(len(X_test), test_samples, replace=False)
        X_test = X_test[indices]
        y_test = y_test[indices]
    
    return X_train, y_train, X_test, y_test

def display_sample_images(X, y, num_samples=5):
    """Display sample images from the dataset"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        idx = np.random.randint(len(X))
        img = X[idx].reshape(28, 28)
        label = np.argmax(y[idx]) if len(y[idx].shape) == 1 else y[idx]
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    print("=== MNIST Neural Network Demo ===")
    
    train_path = "MNIST_CSV/mnist_train.csv"
    test_path = "MNIST_CSV/mnist_test.csv"
    
    if not os.path.exists(train_path):
        print(f"Error: Training file not found at {train_path}")
        print("Please make sure the MNIST CSV files are in the correct location.")
        return
    
    if not os.path.exists(test_path):
        print(f"Error: Test file not found at {test_path}")
        print("Please make sure the MNIST CSV files are in the correct location.")
        return
    
    X_train, y_train, X_test, y_test = load_mnist_data(train_path, test_path)
    X_train, y_train, X_test, y_test = preprocess_mnist_data(X_train, y_train, X_test, y_test)
    
    use_full_dataset = False
    if not use_full_dataset:
        print("\nUsing smaller subset for faster training...")
        X_train, y_train, X_test, y_test = create_small_subset(
            X_train, y_train, X_test, y_test, 
            train_samples=10000, test_samples=2000
        )
    
    print("\nDisplaying sample images...")
    display_sample_images(X_train, y_train, num_samples=5)
    
    # Create neural network for MNIST
    print("\nCreating neural network...")
    nn_mnist = NeuralNetwork(
        input_size=784,           # 28x28 pixels
        hidden_layers=[128, 64],  # Two hidden layers
        output_size=10,           # 10 digits (0-9)
        learning_rate=0.1,
        activation='relu',        # ReLU
        cost_function='cross_entropy'
    )
    
    print(f"Network architecture: 784 -> 128 -> 64 -> 10")
    print(f"Total training samples: {X_train.shape[0]}")
    print(f"Total test samples: {X_test.shape[0]}")
    
    # Train the network
    print("\nStarting training...")
    trainer_mnist = Trainer(nn_mnist, verbose=True)
    
    history_mnist = trainer_mnist.train(
        X_train, y_train, 
        epochs=50, 
        validation_data=(X_test, y_test),
        batch_size=32
    )
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    train_accuracy = nn_mnist.compute_accuracy(X_train, y_train)
    test_accuracy = nn_mnist.compute_accuracy(X_test, y_test)
    
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Display some predictions
    print("\n=== Sample Predictions ===")
    sample_indices = np.random.choice(len(X_test), 10, replace=False)
    for i, idx in enumerate(sample_indices):
        actual = np.argmax(y_test[idx])
        prediction = nn_mnist.predict(X_test[idx:idx+1])
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
        
        status = "✓" if actual == predicted_class else "✗"
        print(f"Sample {i+1}: Actual={actual}, Predicted={predicted_class}, "
              f"Confidence={confidence:.3f} {status}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    Visualizer.plot_training_history(history_mnist)
    
    # Plot some misclassified examples
    plot_misclassified_examples(nn_mnist, X_test, y_test)


def plot_misclassified_examples(model, X_test, y_test, num_examples=5):
    """Plot misclassified examples"""
    try:
        import matplotlib.pyplot as plt
        
        predictions = model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        misclassified = np.where(predicted_classes != true_classes)[0]
        
        if len(misclassified) > 0:
            print(f"\nFound {len(misclassified)} misclassified examples")
            indices = np.random.choice(misclassified, min(num_examples, len(misclassified)), replace=False)
            
            fig, axes = plt.subplots(1, len(indices), figsize=(15, 3))
            if len(indices) == 1:
                axes = [axes]
            
            for i, idx in enumerate(indices):
                img = X_test[idx].reshape(28, 28)
                actual = true_classes[idx]
                predicted = predicted_classes[idx]
                confidence = np.max(predictions[idx])
                
                axes[i].imshow(img, cmap='gray')
                axes[i].set_title(f'Actual: {actual}, Pred: {predicted}\nConf: {confidence:.3f}')
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()
        else:
            print("No misclassified examples found!")
            
    except Exception as e:
        print(f"Could not plot misclassified examples: {e}")

if __name__ == "__main__":
    main()