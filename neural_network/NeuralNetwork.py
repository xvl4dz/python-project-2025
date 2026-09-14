import numpy as np
import pickle
from typing import List, Tuple, Dict, Optional, Union, Any

try:
    # When running as part of the package
    from .Layer import Layer
    from .Cost import get_cost_function 
except ImportError:
    # When running directly
    from Layer import Layer
    from Cost import get_cost_function 

class NeuralNetwork:
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int,
                 learning_rate: float = 0.1, activation: str = 'sigmoid',
                 output_activation: Optional[str] = None, cost_function: str = 'mse') -> None:
        """
        Initialize neural network
        
        Args:
            input_size (int): Number of input nodes
            hidden_layers (List[int]): List of integers specifying nodes in each hidden layer
            output_size (int): Number of output nodes
            learning_rate (float): Learning rate for training. Defaults to 0.1.
            activation (str): Activation function for hidden layers. Defaults to 'sigmoid'.
            output_activation (Optional[str]): Activation function for output layer (None for automatic). Defaults to None.
            cost_function (str): Cost function to use ('mse', 'cross_entropy', 'binary_cross_entropy'). Defaults to 'mse'.
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.cost_function_name = cost_function
        self.cost_fn, self.cost_derivative = get_cost_function(cost_function)
        
        # Determine output activation automatically if not specified
        if output_activation is None:
            if output_size == 1 and cost_function == 'binary_cross_entropy':
                output_activation = 'sigmoid'
            elif output_size > 1 and cost_function == 'cross_entropy':
                output_activation = 'softmax'
            else:
                output_activation = 'linear'
        
        # Build layers
        self.layers = []
        layer_sizes = [input_size] + hidden_layers + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            if i == len(layer_sizes) - 2:
                layer_activation = output_activation
            else:
                layer_activation = activation
                
            layer = Layer(layer_sizes[i], layer_sizes[i + 1], layer_activation)
            self.layers.append(layer)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Network output of shape (n_samples, output_size)
        """
        current_output = X.T  # batch processing
    
        for i, layer in enumerate(self.layers):
            current_output = layer.forward(current_output)
    
        return current_output.T 
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray) -> None:
        """Backward pass through the network
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            y (np.ndarray): Target labels
            output (np.ndarray): Network predictions from forward pass
        """
        # batch processing
        X = X.T
        y = y.T
        output = output.T
        
        # Calculate initial gradient
        d_output = self.cost_derivative(y, output)
        
        # Backpropagate
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output, self.learning_rate)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predictions of shape (n_samples, output_size)
        """
        return self.forward(X)
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute current loss
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            y (np.ndarray): Target labels
            
        Returns:
            float: Scalar loss value
        """
        predictions = self.predict(X)
        return self.cost_fn(y, predictions)
    
    def compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy for classification tasks
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            y (np.ndarray): Target labels
            
        Returns:
            float: Accuracy between 0 and 1
        """
        predictions = self.predict(X)
        
        if self.output_size == 1:  # Binary classification
            predicted_classes = (predictions > 0.5).astype(int)
            accuracy = np.mean(predicted_classes == y)
        else:  # Multi-class classification
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y, axis=1) if y.shape[1] > 1 else y.flatten()
            accuracy = np.mean(predicted_classes == true_classes)
        
        return accuracy
    
    def get_parameters(self) -> List[Dict[str, np.ndarray]]:
        """Get all network parameters for saving
        
        Returns:
            List[Dict[str, np.ndarray]]: List of dictionaries containing 'weights' and 'biases' for each layer
        """
        parameters = []
        for layer in self.layers:
            weights, biases = layer.get_parameters()
            parameters.append({
                'weights': weights,
                'biases': biases
            })
        return parameters

    def set_parameters(self, parameters: List[Dict[str, np.ndarray]]) -> None:
        """Set network parameters from loaded data
        
        Args:
            parameters (List[Dict[str, np.ndarray]]): List of dictionaries containing 'weights' and 'biases' for each layer
        """
        for layer, param_dict in zip(self.layers, parameters):
            layer.set_parameters(param_dict['weights'], param_dict['biases'])

    def save(self, filepath: str) -> None:
        """Save model to file
        
        Args:
            filepath (str): Path to save the model file
        """
        import pickle
        
        model_data = {
            'architecture': {
                'input_size': self.input_size,
                'hidden_layers': self.hidden_layers,
                'output_size': self.output_size,
                'learning_rate': self.learning_rate,
                'activation': self.layers[0].activation.name,
                'cost_function': self.cost_function_name
            },
            'parameters': self.get_parameters()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'NeuralNetwork':
        """Load model from file and recreate the neural network
        
        Args:
            filepath (str): Path to saved model file
            
        Returns:
            NeuralNetwork: Reconstructed neural network with loaded parameters
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Extract architecture and parameters
        arch = model_data['architecture']
        parameters = model_data['parameters']
        
        # Create new neural network with the same architecture
        model = cls(
            input_size=arch['input_size'],
            hidden_layers=arch['hidden_layers'],
            output_size=arch['output_size'],
            learning_rate=arch['learning_rate'],
            activation=arch['activation'],
            cost_function=arch['cost_function']
        )
        
        # Set the loaded parameters
        model.set_parameters(parameters)
        
        print(f"Model loaded from {filepath}")
        print(f"Architecture: {arch['input_size']} -> {arch['hidden_layers']} -> {arch['output_size']}")
        
        return model
