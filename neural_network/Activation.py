import numpy as np
from typing import Union

class Activation:
    def __init__(self, activation_name: str) -> None:
        """
        Initialize activation function.
        
        Args:
            activation_name (str): Name of activation function.
                Available functions: 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'softmax', 'linear'.
        """
        self.name = activation_name
    
    def function(self, x: np.ndarray) -> np.ndarray:
        """
        Apply activation function.
        
        Args:
            x (np.ndarray): Input values
            
        Returns:
            np.ndarray: Activated values
            
        Raises:
            ValueError: If activation function name is not supported
            
        Available functions:
            - 'sigmoid': Sigmoid function, outputs between 0 and 1
            - 'tanh': Hyperbolic tangent, outputs between -1 and 1
            - 'relu': Rectified Linear Unit, outputs max(0, x)
            - 'leaky_relu': Leaky ReLU, outputs x if x > 0 else 0.01*x
            - 'softmax': Softmax function for probability distributions
            - 'linear': Identity function, outputs x directly
        """
        if self.name == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
        elif self.name == 'tanh':
            return np.tanh(x)
        elif self.name == 'relu':
            return np.maximum(0, x)
        elif self.name == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)
        elif self.name == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
            return exp_x / np.sum(exp_x, axis=0, keepdims=True)
        elif self.name == 'linear':
            return x
        else:
            raise ValueError(f"Unsupported activation function: {self.name}")
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Compute derivative of activation function.
        
        Args:
            x (np.ndarray): Input values
            
        Returns:
            np.ndarray: Derivative values
            
        Raises:
            ValueError: If activation function name is not supported
            
        Available functions and their derivatives:
            - 'sigmoid': derivative = sigmoid(x) * (1 - sigmoid(x))
            - 'tanh': derivative = 1 - tanh²(x)
            - 'relu': derivative = 1 if x > 0 else 0
            - 'leaky_relu': derivative = 1 if x > 0 else 0.01
            - 'softmax': derivative = softmax(x) * (1 - softmax(x)) [simplified]
            - 'linear': derivative = 1
        """
        if self.name == 'sigmoid':
            sig = self.function(x)
            return sig * (1 - sig)
        elif self.name == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.name == 'relu':
            return (x > 0).astype(float)
        elif self.name == 'leaky_relu':
            return np.where(x > 0, 1, 0.01)
        elif self.name == 'softmax':
            # derivative is simplified 
            softmax = self.function(x)
            return softmax * (1 - softmax)
        elif self.name == 'linear':
            return np.ones_like(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.name}")
