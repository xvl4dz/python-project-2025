"""
Neural Network Package
"""

# Import all the main classes
from .NeuralNetwork import NeuralNetwork
from .Layer import Layer
from .Activation import Activation
from .Trainer import Trainer
from .Visualizer import Visualizer

# Import all cost-related items
from .Cost import (
    get_cost_function, 
    BaseCost, 
    MSE, 
    CrossEntropy, 
    BinaryCrossEntropy
)

# Users can do: from neural_network.Cost import get_cost_function
import sys
from . import Cost

from .scripts.test import load_and_preprocess_mnist

__all__ = [
    'NeuralNetwork',
    'Layer',
    'Activation',
    'Trainer', 
    'Visualizer',
    'get_cost_function',
    'BaseCost',
    'MSE',
    'CrossEntropy',
    'BinaryCrossEntropy',
    'Cost',
    'load_and_preprocess_mnist'
]

__version__ = '1.0.0'