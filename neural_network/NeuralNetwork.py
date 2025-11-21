import numpy as np
import pickle
from .Layer import Layer
from .Cost import Cost

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, 
                 learning_rate=0.1, activation='sigmoid', 
                 output_activation=None, cost_function='mse'):
        """
        Initialize neural network
        
        Args:
            input_size: Number of input nodes
            hidden_layers: List of integers specifying nodes in each hidden layer
            output_size: Number of output nodes
            learning_rate: Learning rate for training
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer (None for automatic)
            cost_function: Cost function to use ('mse', 'cross_entropy', 'binary_cross_entropy')
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.cost_function_name = cost_function
        self.cost_fn, self.cost_derivative = Cost.get_cost_function(cost_function)
        
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
    
    def forward(self, X):
        """Forward pass through the network"""
        current_output = X.T  # batch processing
    
        for i, layer in enumerate(self.layers):
            current_output = layer.forward(current_output)
    
        return current_output.T 
    
    def backward(self, X, y, output):
        """Backward pass through the network"""
        # batch processing
        X = X.T
        y = y.T
        output = output.T
        
        # Calculate initial gradient
        d_output = self.cost_derivative(y, output)
        
        # Backpropagate
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output, self.learning_rate)
    
    def predict(self, X):
        """Make predictions"""
        return self.forward(X)
    
    def compute_loss(self, X, y):
        """Compute current loss"""
        predictions = self.predict(X)
        return self.cost_fn(y, predictions)
    
    def compute_accuracy(self, X, y):
        """Compute accuracy for classification tasks"""
        predictions = self.predict(X)
        
        if self.output_size == 1:  # Binary classification
            predicted_classes = (predictions > 0.5).astype(int)
            accuracy = np.mean(predicted_classes == y)
        else:  # Multi-class classification
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y, axis=1) if y.shape[1] > 1 else y.flatten()
            accuracy = np.mean(predicted_classes == true_classes)
        
        return accuracy
    
    def get_parameters(self):
        """Get all network parameters"""
        parameters = []
        for layer in self.layers:
            parameters.append(layer.get_parameters())
        return parameters
    
    def set_parameters(self, parameters):
        """Set all network parameters"""
        for layer, (weights, biases) in zip(self.layers, parameters):
            layer.set_parameters(weights, biases)

    def get_parameters(self):
        """Get all network parameters for saving"""
        parameters = []
        for layer in self.layers:
            weights, biases = layer.get_parameters()
            parameters.append({
                'weights': weights,
                'biases': biases
            })
        return parameters

    def set_parameters(self, parameters):
        """Set network parameters from loaded data"""
        for layer, param_dict in zip(self.layers, parameters):
            layer.set_parameters(param_dict['weights'], param_dict['biases'])

    def save(self, filepath):
        """Save model to file"""
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
    def load_model(cls, filepath):
        """Load model from file and recreate the neural network"""
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