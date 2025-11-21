import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    @staticmethod
    def plot_training_history(history):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # loss
        ax1.plot(history['train_loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history and history['val_loss']:
            ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # accuracy
        ax2.plot(history['train_accuracy'], label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history and history['val_accuracy']:
            ax2.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_decision_boundary(model, X, y, title="Decision Boundary"):
        """Plot decision boundary for 2D data"""
        if X.shape[1] != 2:
            print("Decision boundary plot only available for 2D input")
            return
        
        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                           np.linspace(y_min, y_max, 100))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        
        if model.output_size == 1:
            Z = Z.reshape(xx.shape)
            levels = [0.5]
        else:
            Z = np.argmax(Z, axis=1).reshape(xx.shape)
            levels = np.arange(-0.5, model.output_size)
        
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(xx, yy, Z, alpha=0.8, cmap='RdYlBu')
        plt.colorbar(contour)
        
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), 
                            cmap='RdYlBu', edgecolors='black', s=50)
        plt.colorbar(scatter)
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_predictions_comparison(model, X, y, title="Predictions vs Actual"):
        """Compare predictions with actual values"""
        predictions = model.predict(X)
        
        plt.figure(figsize=(10, 4))
        
        if model.output_size == 1:
            plt.subplot(1, 2, 1)
            plt.scatter(y, predictions, alpha=0.7)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Predictions vs Actual')
            
            plt.subplot(1, 2, 2)
            errors = predictions.flatten() - y.flatten()
            plt.hist(errors, bins=20, alpha=0.7)
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.title('Error Distribution')
        else:
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y, axis=1) if y.shape[1] > 1 else y.flatten()
            
            plt.scatter(true_classes, predicted_classes, alpha=0.7)
            plt.xlabel('True Classes')
            plt.ylabel('Predicted Classes')
            plt.title('True vs Predicted Classes')
        
        plt.tight_layout()
        plt.show()