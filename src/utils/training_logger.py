"""
Enhanced logging utilities for training monitoring.
"""

import os
import sys
import logging
import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


class TrainingLogger:
    """Enhanced logger for training with file and console output."""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 experiment_name: Optional[str] = None,
                 log_level: int = logging.INFO):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory to save log files
            experiment_name: Name of the experiment (auto-generated if None)
            log_level: Logging level
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"fsra_vit_improved_{timestamp}"
        
        self.experiment_name = experiment_name
        self.log_level = log_level
        
        # Create experiment directory
        self.exp_dir = self.log_dir / experiment_name
        self.exp_dir.mkdir(exist_ok=True)
        
        # Setup loggers
        self.setup_loggers()
        
        # Metrics storage
        self.metrics_history = []
        
    def setup_loggers(self):
        """Setup file and console loggers."""
        
        # Main training logger
        self.logger = logging.getLogger(f"training_{self.experiment_name}")
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler for detailed logs
        log_file = self.exp_dir / "training.log"
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for important info
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        
        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Metrics logger (JSON format)
        self.metrics_logger = logging.getLogger(f"metrics_{self.experiment_name}")
        self.metrics_logger.setLevel(logging.INFO)
        self.metrics_logger.handlers.clear()
        
        metrics_file = self.exp_dir / "metrics.jsonl"
        metrics_handler = logging.FileHandler(metrics_file, mode='w', encoding='utf-8')
        metrics_handler.setLevel(logging.INFO)
        metrics_handler.setFormatter(logging.Formatter('%(message)s'))
        self.metrics_logger.addHandler(metrics_handler)
        
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
        
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
        
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
        
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
        
    def log_metrics(self, metrics: Dict[str, Any], epoch: int, batch: Optional[int] = None):
        """
        Log metrics in JSON format.
        
        Args:
            metrics: Dictionary of metrics
            epoch: Current epoch
            batch: Current batch (optional)
        """
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "epoch": epoch,
            "metrics": metrics
        }
        
        if batch is not None:
            log_entry["batch"] = batch
            
        # Log to JSON file
        self.metrics_logger.info(json.dumps(log_entry))
        
        # Store in memory
        self.metrics_history.append(log_entry)
        
    def log_epoch_summary(self, epoch: int, metrics: Dict[str, Any]):
        """Log epoch summary with beautiful formatting."""
        
        self.info("=" * 80)
        self.info(f"🎯 EPOCH {epoch} SUMMARY")
        self.info("=" * 80)
        
        # Performance metrics
        if 'avg_loss' in metrics:
            self.info(f"📊 Average Loss: {metrics['avg_loss']:.6f}")
        if 'accuracy' in metrics:
            self.info(f"🎯 Accuracy: {metrics['accuracy']:.4f}")
        if 'top5_accuracy' in metrics:
            self.info(f"🏆 Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
            
        # Time and memory
        if 'epoch_time' in metrics:
            self.info(f"⏱️  Epoch Time: {metrics['epoch_time']:.1f}s")
        if 'avg_batch_time' in metrics:
            self.info(f"⚡ Avg Batch Time: {metrics['avg_batch_time']:.3f}s")
        if 'gpu_memory_used' in metrics:
            self.info(f"💾 GPU Memory: {metrics['gpu_memory_used']:.1f}MB")
        if 'ram_used' in metrics:
            self.info(f"🧠 RAM Used: {metrics['ram_used']:.2f}GB")
            
        self.info("=" * 80)
        
    def log_training_start(self, config: Dict[str, Any]):
        """Log training start with configuration."""
        
        self.info("🚀 STARTING FSRA VIT IMPROVED TRAINING")
        self.info("=" * 80)
        self.info(f"📁 Experiment: {self.experiment_name}")
        self.info(f"📂 Log Directory: {self.exp_dir}")
        self.info("=" * 80)
        
        # Log configuration
        self.info("⚙️  TRAINING CONFIGURATION:")
        for key, value in config.items():
            self.info(f"   {key}: {value}")
        self.info("=" * 80)
        
        # Save config to file
        config_file = self.exp_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
    def log_training_end(self, total_time: float, best_metrics: Dict[str, Any]):
        """Log training completion."""
        
        self.info("🎉 TRAINING COMPLETED!")
        self.info("=" * 80)
        self.info(f"⏱️  Total Training Time: {total_time:.1f}s ({total_time/3600:.2f}h)")
        
        if best_metrics:
            self.info("🏆 BEST METRICS:")
            for key, value in best_metrics.items():
                if isinstance(value, float):
                    self.info(f"   {key}: {value:.6f}")
                else:
                    self.info(f"   {key}: {value}")
                    
        self.info(f"📁 All logs saved to: {self.exp_dir}")
        self.info("=" * 80)
        
    def save_metrics_summary(self):
        """Save metrics summary to file."""
        if not self.metrics_history:
            return
            
        summary_file = self.exp_dir / "metrics_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
            
        self.info(f"📊 Metrics summary saved to: {summary_file}")
        
    def get_log_dir(self) -> Path:
        """Get experiment log directory."""
        return self.exp_dir


def setup_training_logger(log_dir: str = "logs", 
                         experiment_name: Optional[str] = None) -> TrainingLogger:
    """
    Setup training logger with default configuration.
    
    Args:
        log_dir: Directory to save logs
        experiment_name: Name of experiment
        
    Returns:
        Configured TrainingLogger instance
    """
    return TrainingLogger(log_dir=log_dir, experiment_name=experiment_name)
