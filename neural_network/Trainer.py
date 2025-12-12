import numpy as np
from tqdm import tqdm
import time
from typing import Tuple, Optional, Dict, Any

class Trainer:
    def __init__(self, model: Any, verbose: bool = True) -> None:
        """
        Initialize trainer for neural network.
        
        Args:
            model (Any): NeuralNetwork model to train
            verbose (bool): Whether to print training progress. Defaults to True.
        """
        self.model = model
        self.verbose = verbose
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, 
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None, 
              batch_size: Optional[int] = None) -> Dict[str, list]:
        """
        Train the neural network.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            epochs (int): Number of training epochs
            validation_data (Optional[Tuple]): Tuple (X_val, y_val) for validation. Defaults to None.
            batch_size (Optional[int]): Batch size for mini-batch gradient descent. Defaults to None (full batch).
            
        Returns:
            Dict[str, list]: Training history
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
    
    def _print_progress(self, epoch: int, epochs: int, has_validation: bool) -> None:
        """
        Print training progress.
        
        Args:
            epoch (int): Current epoch number
            epochs (int): Total number of epochs
            has_validation (bool): Whether validation data is provided
        """
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
