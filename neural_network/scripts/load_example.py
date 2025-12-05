def load_model(filepath):
    """Load model from file and recreate the neural network"""
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    # Extract architecture and parameters
    arch = model_data['architecture']
    parameters = model_data['parameters']
    
    # Create new neural network with the same architecture
    model = NeuralNetwork(
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