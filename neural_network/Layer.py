import numpy as np
from typing import Tuple

try:
    from .Activation import Activation
except ImportError:
    from Activation import Activation


class Layer:
    def __init__(self, input_size: int, output_size: int, activation: str = 'sigmoid') -> None:
        """
        Initialize a neural network layer.
        
        Args:
            input_size (int): Number of input nodes
            output_size (int): Number of output nodes
            activation (str): Activation function name ('sigmoid', 'relu', 'tanh', etc.). Defaults to 'sigmoid'.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = Activation(activation)
        
        # Initialize weights and biases
        std = np.sqrt(2.0 / input_size)  # He initialization
        self.weights = np.random.randn(output_size, input_size) * std
        self.biases = np.zeros((output_size, 1))
        
        # Cache for backpropagation
        self.input = None
        self.z = None
        self.output = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        
        Args:
            X (np.ndarray): Input data of shape (input_size, batch_size)
            
        Returns:
            np.ndarray: Output data of shape (output_size, batch_size)
        """
        self.input = X
        self.z = self.weights @ self.input + self.biases
        self.output = self.activation.function(self.z)
        return self.output
    
    def backward(self, d_output: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass through the layer.
        
        Args:
            d_output (np.ndarray): Gradient from next layer of shape (output_size, batch_size)
            learning_rate (float): Learning rate for weight updates
            
        Returns:
            np.ndarray: Gradient to pass to previous layer of shape (input_size, batch_size)
        """
        m = self.input.shape[1]
        
        # Calculate derivative of activation
        d_z = d_output * self.activation.derivative(self.z)
        
        # Calculate gradients
        d_weights = d_z @ self.input.T / m
        d_biases = np.sum(d_z, axis=1, keepdims=True) / m
        d_input = self.weights.T @ d_z
        
        # Update parameters
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        
        return d_input
    
    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get layer parameters.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Weights and biases
        """
        return self.weights, self.biases
    
    def set_parameters(self, weights: np.ndarray, biases: np.ndarray) -> None:
        """
        Set layer parameters.
        
        Args:
            weights (np.ndarray): Weights matrix
            biases (np.ndarray): Biases vector
        """
        self.weights = weights
        self.biases = biases
