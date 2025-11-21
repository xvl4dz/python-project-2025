import numpy as np

class Activation:
    def __init__(self, activation_name):
        self.name = activation_name
    
    def function(self, x):
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
    
    def derivative(self, x):
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