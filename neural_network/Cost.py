import numpy as np

class Cost:
    @staticmethod
    def mse(y_true, y_pred):
        """Mean Squared Error"""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mse_derivative(y_true, y_pred):
        """Derivative of MSE"""
        return 2 * (y_pred - y_true) / y_true.size
    
    @staticmethod
    def cross_entropy(y_true, y_pred, epsilon=1e-12):
        """Cross Entropy Loss with improved numerical stability"""
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    
        # Calculate cross entropy
        if len(y_true.shape) == 1 or y_true.shape[1] == 1:
            # Binary classification
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            # Multi-class classification
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    @staticmethod
    def cross_entropy_derivative(y_true, y_pred, epsilon=1e-12):
        """Derivative of Cross Entropy (for softmax output)"""
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return y_pred - y_true
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred, epsilon=1e-12):
        """Binary Cross Entropy Loss"""
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def binary_cross_entropy_derivative(y_true, y_pred, epsilon=1e-12):
        """Derivative of Binary Cross Entropy"""
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.size
    
    @staticmethod
    def get_cost_function(cost_name):
        """Get cost function and its derivative by name"""
        cost_functions = {
            'mse': (Cost.mse, Cost.mse_derivative),
            'cross_entropy': (Cost.cross_entropy, Cost.cross_entropy_derivative),
            'binary_cross_entropy': (Cost.binary_cross_entropy, Cost.binary_cross_entropy_derivative)
        }
        return cost_functions.get(cost_name, (Cost.mse, Cost.mse_derivative))