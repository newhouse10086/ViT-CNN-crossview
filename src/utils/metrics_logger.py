"""
Enhanced metrics logging and tracking utilities.
"""

import os
import json
import csv
import time
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


class MetricsLogger:
    """Enhanced metrics logger for training tracking."""
    
    def __init__(self, log_dir: str, experiment_name: str = "experiment"):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.log_dir / "metrics"
        self.plots_dir = self.log_dir / "plots"
        self.metrics_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        
        # Initialize storage
        self.metrics_history = []
        self.epoch_times = []
        
        # File paths
        self.csv_path = self.metrics_dir / f"{experiment_name}_metrics.csv"
        self.json_path = self.metrics_dir / f"{experiment_name}_metrics.json"
        
        # Initialize CSV file
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        headers = [
            'epoch', 'train_loss', 'train_accuracy', 'train_precision', 
            'train_recall', 'train_f1_score', 'learning_rate', 'epoch_time',
            'avg_batch_time', 'batches_processed', 'timestamp'
        ]
        
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """Log metrics for an epoch."""
        # Add timestamp
        metrics['timestamp'] = time.time()
        metrics['epoch'] = epoch
        
        # Store in memory
        self.metrics_history.append(metrics.copy())
        
        # Save to CSV
        self._save_to_csv(metrics)
        
        # Save to JSON
        self._save_to_json()
        
        # Generate plots
        self._generate_plots()
    
    def _save_to_csv(self, metrics: Dict[str, Any]):
        """Save metrics to CSV file."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                metrics.get('epoch', 0),
                metrics.get('train_loss', 0.0),
                metrics.get('train_accuracy', 0.0),
                metrics.get('train_precision', 0.0),
                metrics.get('train_recall', 0.0),
                metrics.get('train_f1_score', 0.0),
                metrics.get('learning_rate', 0.0),
                metrics.get('epoch_time', 0.0),
                metrics.get('avg_batch_time', 0.0),
                metrics.get('batches_processed', 0),
                metrics.get('timestamp', time.time())
            ]
            writer.writerow(row)
    
    def _save_to_json(self):
        """Save all metrics to JSON file."""
        with open(self.json_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def _generate_plots(self):
        """Generate training plots."""
        if len(self.metrics_history) < 2:
            return
        
        df = pd.DataFrame(self.metrics_history)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Training Metrics - {self.experiment_name}', fontsize=16)
        
        # Loss plot
        axes[0, 0].plot(df['epoch'], df['train_loss'], 'b-', linewidth=2)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(df['epoch'], df['train_accuracy'], 'g-', linewidth=2)
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[0, 2].plot(df['epoch'], df['learning_rate'], 'r-', linewidth=2)
        axes[0, 2].set_title('Learning Rate')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Precision and Recall
        axes[1, 0].plot(df['epoch'], df['train_precision'], 'c-', linewidth=2, label='Precision')
        axes[1, 0].plot(df['epoch'], df['train_recall'], 'm-', linewidth=2, label='Recall')
        axes[1, 0].set_title('Precision and Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # F1 Score
        axes[1, 1].plot(df['epoch'], df['train_f1_score'], 'orange', linewidth=2)
        axes[1, 1].set_title('F1 Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Epoch Time
        axes[1, 2].plot(df['epoch'], df['epoch_time'], 'purple', linewidth=2)
        axes[1, 2].set_title('Epoch Time')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Time (seconds)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / f"{self.experiment_name}_training_metrics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_best_metrics(self) -> Dict[str, Any]:
        """Get best metrics achieved during training."""
        if not self.metrics_history:
            return {}
        
        df = pd.DataFrame(self.metrics_history)
        
        best_metrics = {
            'best_accuracy': {
                'value': df['train_accuracy'].max(),
                'epoch': df.loc[df['train_accuracy'].idxmax(), 'epoch']
            },
            'best_f1_score': {
                'value': df['train_f1_score'].max(),
                'epoch': df.loc[df['train_f1_score'].idxmax(), 'epoch']
            },
            'lowest_loss': {
                'value': df['train_loss'].min(),
                'epoch': df.loc[df['train_loss'].idxmin(), 'epoch']
            },
            'total_training_time': df['epoch_time'].sum(),
            'avg_epoch_time': df['epoch_time'].mean()
        }
        
        return best_metrics
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of training."""
        if not self.metrics_history:
            return "No metrics available."
        
        best_metrics = self.get_best_metrics()
        latest_metrics = self.metrics_history[-1]
        
        report = f"""
Training Summary Report - {self.experiment_name}
{'='*60}

Final Metrics (Epoch {latest_metrics['epoch']}):
  Loss: {latest_metrics['train_loss']:.6f}
  Accuracy: {latest_metrics['train_accuracy']:.4f}
  Precision: {latest_metrics['train_precision']:.4f}
  Recall: {latest_metrics['train_recall']:.4f}
  F1-Score: {latest_metrics['train_f1_score']:.4f}

Best Metrics Achieved:
  Best Accuracy: {best_metrics['best_accuracy']['value']:.4f} (Epoch {best_metrics['best_accuracy']['epoch']})
  Best F1-Score: {best_metrics['best_f1_score']['value']:.4f} (Epoch {best_metrics['best_f1_score']['epoch']})
  Lowest Loss: {best_metrics['lowest_loss']['value']:.6f} (Epoch {best_metrics['lowest_loss']['epoch']})

Training Time:
  Total Training Time: {best_metrics['total_training_time']:.2f} seconds
  Average Epoch Time: {best_metrics['avg_epoch_time']:.2f} seconds
  Total Epochs: {len(self.metrics_history)}

Files Generated:
  Metrics CSV: {self.csv_path}
  Metrics JSON: {self.json_path}
  Training Plots: {self.plots_dir}
"""
        
        # Save report
        report_path = self.log_dir / f"{self.experiment_name}_summary.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report
    
    def load_metrics(self, json_path: str = None) -> List[Dict[str, Any]]:
        """Load metrics from JSON file."""
        if json_path is None:
            json_path = self.json_path
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                self.metrics_history = json.load(f)
        
        return self.metrics_history
