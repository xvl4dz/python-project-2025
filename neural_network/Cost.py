import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod


class BaseCost(ABC):
    """Abstract base class for cost functions."""
    
    @staticmethod
    @abstractmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute cost value.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            float: Cost value
        """
        pass
    
    @staticmethod
    @abstractmethod
    def derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute cost derivative.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            np.ndarray: Derivative of cost
        """
        pass


class MSE(BaseCost):
    """Mean Squared Error cost function."""
    
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Squared Error.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            float: MSE loss
        """
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Derivative of MSE.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            np.ndarray: Gradient of MSE
        """
        return 2 * (y_pred - y_true) / y_true.size


class CrossEntropy(BaseCost):
    """Cross Entropy Loss for multi-class classification."""
    
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-12) -> float:
        """
        Cross Entropy Loss with improved numerical stability.
        
        Args:
            y_true (np.ndarray): True labels (one-hot encoded for multi-class)
            y_pred (np.ndarray): Predicted probabilities
            epsilon (float): Small value to avoid log(0). Defaults to 1e-12.
            
        Returns:
            float: Cross entropy loss
        """
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
    def derivative(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
        """
        Derivative of Cross Entropy (for softmax output).
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted probabilities
            epsilon (float): Small value for numerical stability. Defaults to 1e-12.
            
        Returns:
            np.ndarray: Gradient of cross entropy
        """
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return y_pred - y_true


class BinaryCrossEntropy(BaseCost):
    """Binary Cross Entropy Loss for binary classification."""
    
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-12) -> float:
        """
        Binary Cross Entropy Loss.
        
        Args:
            y_true (np.ndarray): True binary labels (0 or 1)
            y_pred (np.ndarray): Predicted probabilities
            epsilon (float): Small value to avoid log(0). Defaults to 1e-12.
            
        Returns:
            float: Binary cross entropy loss
        """
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def derivative(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
        """
        Derivative of Binary Cross Entropy.
        
        Args:
            y_true (np.ndarray): True binary labels
            y_pred (np.ndarray): Predicted probabilities
            epsilon (float): Small value for numerical stability. Defaults to 1e-12.
            
        Returns:
            np.ndarray: Gradient of binary cross entropy
        """
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.size


def get_cost_function(cost_name: str) -> Tuple[callable, callable]:
    """
    Get cost function and its derivative by name.
    
    Args:
        cost_name (str): Name of cost function ('mse', 'cross_entropy', 'binary_cross_entropy')
        
    Returns:
        Tuple[callable, callable]: (cost_function, cost_derivative)
        
    Raises:
        ValueError: If cost_name is not recognized
    """
    cost_mapping = {
        'mse': (MSE.compute, MSE.derivative),
        'cross_entropy': (CrossEntropy.compute, CrossEntropy.derivative),
        'binary_cross_entropy': (BinaryCrossEntropy.compute, BinaryCrossEntropy.derivative)
    }
    
    if cost_name not in cost_mapping:
        raise ValueError(
            f"Unknown cost function: '{cost_name}'. "
            f"Available options: {list(cost_mapping.keys())}"
        )
    
    return cost_mapping[cost_name]
