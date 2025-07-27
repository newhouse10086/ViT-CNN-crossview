"""Visualization utilities for ViT-CNN-crossview."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TrainingVisualizer:
    """Visualizer for training progress and metrics."""
    
    def __init__(self, save_dir: str = "logs/plots", experiment_name: str = "experiment"):
        """
        Initialize training visualizer.
        
        Args:
            save_dir: Directory to save plots
            experiment_name: Name of the experiment
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # Storage for metrics
        self.metrics_history = {
            'epoch': [],
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        # Additional metrics storage
        self.additional_metrics = {}
    
    def update_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Update metrics for current epoch.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics
        """
        self.metrics_history['epoch'].append(epoch)
        
        # Update standard metrics
        for key in ['train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'learning_rate']:
            if key in metrics:
                self.metrics_history[key].append(metrics[key])
            else:
                self.metrics_history[key].append(None)
        
        # Update additional metrics
        for key, value in metrics.items():
            if key not in self.metrics_history:
                if key not in self.additional_metrics:
                    self.additional_metrics[key] = []
                self.additional_metrics[key].append(value)
    
    def plot_training_curves(self, save: bool = True, show: bool = False) -> plt.Figure:
        """
        Plot training curves.
        
        Args:
            save: Whether to save the plot
            show: Whether to show the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Progress - {self.experiment_name}', fontsize=16)
        
        epochs = self.metrics_history['epoch']
        
        # Loss plot
        axes[0, 0].plot(epochs, self.metrics_history['train_loss'], 
                       label='Train Loss', marker='o', markersize=3)
        if any(x is not None for x in self.metrics_history['val_loss']):
            axes[0, 0].plot(epochs, self.metrics_history['val_loss'], 
                           label='Val Loss', marker='s', markersize=3)
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, self.metrics_history['train_accuracy'], 
                       label='Train Accuracy', marker='o', markersize=3)
        if any(x is not None for x in self.metrics_history['val_accuracy']):
            axes[0, 1].plot(epochs, self.metrics_history['val_accuracy'], 
                           label='Val Accuracy', marker='s', markersize=3)
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        if any(x is not None for x in self.metrics_history['learning_rate']):
            axes[1, 0].plot(epochs, self.metrics_history['learning_rate'], 
                           label='Learning Rate', marker='o', markersize=3, color='red')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Additional metrics plot
        if self.additional_metrics:
            metric_names = list(self.additional_metrics.keys())[:4]  # Show up to 4 metrics
            for i, metric_name in enumerate(metric_names):
                color = plt.cm.tab10(i)
                axes[1, 1].plot(epochs[:len(self.additional_metrics[metric_name])], 
                               self.additional_metrics[metric_name], 
                               label=metric_name, marker='o', markersize=3, color=color)
            
            axes[1, 1].set_title('Additional Metrics')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f"{self.experiment_name}_training_curves.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training curves to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def save_metrics_csv(self) -> str:
        """
        Save metrics to CSV file.
        
        Returns:
            Path to saved CSV file
        """
        # Combine all metrics
        all_metrics = self.metrics_history.copy()
        
        # Add additional metrics
        max_len = len(all_metrics['epoch'])
        for key, values in self.additional_metrics.items():
            # Pad with None if shorter
            padded_values = values + [None] * (max_len - len(values))
            all_metrics[key] = padded_values[:max_len]
        
        # Create DataFrame
        df = pd.DataFrame(all_metrics)
        
        # Save to CSV
        csv_path = self.save_dir / f"{self.experiment_name}_metrics.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Saved metrics to {csv_path}")
        return str(csv_path)


def plot_confusion_matrix(cm: np.ndarray, class_names: Optional[List[str]] = None,
                         title: str = "Confusion Matrix", save_path: Optional[str] = None,
                         show: bool = False) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        save_path: Path to save the plot
        show: Whether to show the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_training_curves(metrics_history: Dict[str, List[float]], 
                        title: str = "Training Curves",
                        save_path: Optional[str] = None,
                        show: bool = False) -> plt.Figure:
    """
    Plot training curves from metrics history.
    
    Args:
        metrics_history: Dictionary of metric histories
        title: Plot title
        save_path: Path to save the plot
        show: Whether to show the plot
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)
    
    epochs = metrics_history.get('epoch', range(1, len(list(metrics_history.values())[0]) + 1))
    
    # Loss plot
    if 'train_loss' in metrics_history:
        axes[0].plot(epochs, metrics_history['train_loss'], 
                    label='Train Loss', marker='o', markersize=3)
    if 'val_loss' in metrics_history:
        axes[0].plot(epochs, metrics_history['val_loss'], 
                    label='Val Loss', marker='s', markersize=3)
    
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    if 'train_accuracy' in metrics_history:
        axes[1].plot(epochs, metrics_history['train_accuracy'], 
                    label='Train Accuracy', marker='o', markersize=3)
    if 'val_accuracy' in metrics_history:
        axes[1].plot(epochs, metrics_history['val_accuracy'], 
                    label='Val Accuracy', marker='s', markersize=3)
    
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training curves to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, 
                   class_names: Optional[List[str]] = None,
                   title: str = "ROC Curves", save_path: Optional[str] = None,
                   show: bool = False) -> plt.Figure:
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        class_names: List of class names
        title: Plot title
        save_path: Path to save the plot
        show: Whether to show the plot
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    # Binarize labels for multi-class ROC
    n_classes = y_prob.shape[1]
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        
        class_name = class_names[i] if class_names else f'Class {i}'
        ax.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curves to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig
