import numpy as np
from tqdm import tqdm
import time

class Trainer:
    def __init__(self, model, verbose=True):
        self.model = model
        self.verbose = verbose
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def train(self, X, y, epochs, validation_data=None, batch_size=None):
        """
        Train the neural network
        
        Args:
            X: Training features
            y: Training labels
            epochs: Number of training epochs
            validation_data: Tuple (X_val, y_val) for validation
            batch_size: Batch size for mini-batch gradient descent (None for full batch)
        """
        if self.verbose:
            print(f"Starting training for {epochs} epochs...")
            print(f"Training samples: {X.shape[0]}")
            if validation_data:
                print(f"Validation samples: {validation_data[0].shape[0]}")
            print(f"Batch size: {batch_size if batch_size else 'Full batch'}")
            print("-" * 50)
        
        start_time = time.time()
        
        if batch_size is None:
            batch_size = X.shape[0]
        
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_losses = []
            epoch_accuracies = []
            
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Forward
                output = self.model.forward(X_batch)
                batch_loss = self.model.cost_fn(y_batch, output)
                batch_accuracy = self.model.compute_accuracy(X_batch, y_batch)
                
                epoch_losses.append(batch_loss)
                epoch_accuracies.append(batch_accuracy)
                
                # Backward
                self.model.backward(X_batch, y_batch, output)
            
            # Store metrics
            self.history['train_loss'].append(np.mean(epoch_losses))
            self.history['train_accuracy'].append(np.mean(epoch_accuracies))
            
            if validation_data is not None:
                X_val, y_val = validation_data
                val_loss = self.model.compute_loss(X_val, y_val)
                val_accuracy = self.model.compute_accuracy(X_val, y_val)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)
            
            if self.verbose and (epoch % max(1, epochs // 100) == 0 or epoch == epochs - 1):
                self._print_progress(epoch, epochs, validation_data is not None)
        
        training_time = time.time() - start_time
        if self.verbose:
            print(f"\nTraining completed in {training_time:.2f} seconds")
        
        return self.history
    
    def _print_progress(self, epoch, epochs, has_validation):
        """Print training progress"""
        train_loss = self.history['train_loss'][-1]
        train_acc = self.history['train_accuracy'][-1]
        
        if has_validation:
            val_loss = self.history['val_loss'][-1]
            val_acc = self.history['val_accuracy'][-1]
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f}")