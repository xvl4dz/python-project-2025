import numpy as np
from .Activation import Activation

class Layer:
    def __init__(self, input_size, output_size, activation='sigmoid'):
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
    
    def forward(self, X):
        """Forward pass through the layer"""
        self.input = X
        self.z = self.weights @ self.input + self.biases
        self.output = self.activation.function(self.z)
        return self.output
    
    def backward(self, d_output, learning_rate):
        """Backward pass through the layer"""
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
    
    def get_parameters(self):
        """Get layer parameters"""
        return self.weights, self.biases
    
    def set_parameters(self, weights, biases):
        """Set layer parameters"""
        self.weights = weights
        self.biases = biases