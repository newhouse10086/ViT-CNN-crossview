#!/usr/bin/env python3
"""
Simple training script for your innovation method.
FSRA + Community Clustering + PCA
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils import load_config, setup_logger
from src.models import create_model
from src.datasets import make_dataloader
from src.losses import CombinedLoss
from src.optimizers import create_optimizer_with_config
from src.visualization import TrainingVisualizer


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleMetricsCalculator:
    """Simple metrics calculator."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, predictions, targets, loss):
        """Update metrics with batch results."""
        if isinstance(predictions, torch.Tensor):
            pred_classes = torch.argmax(predictions, dim=1)
            self.predictions.extend(pred_classes.cpu().numpy())
        
        if isinstance(targets, torch.Tensor):
            self.targets.extend(targets.cpu().numpy())
        
        if isinstance(loss, torch.Tensor):
            self.losses.append(loss.item())
        else:
            self.losses.append(loss)
    
    def compute_metrics(self):
        """Compute final metrics."""
        if len(self.predictions) == 0 or len(self.targets) == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'avg_loss': np.mean(self.losses) if self.losses else 0.0
            }
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Calculate metrics
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average='weighted', zero_division=0)
        recall = recall_score(targets, predictions, average='weighted', zero_division=0)
        f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        avg_loss = np.mean(self.losses) if self.losses else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_loss': avg_loss
        }


def train_epoch(model, dataloader, criterion, optimizer, device, metrics_calc, epoch, total_epochs, log_interval=10):
    """Train for one epoch."""
    model.train()
    metrics_calc.reset()
    
    epoch_start_time = time.time()
    
    for batch_idx, batch_data in enumerate(dataloader):
        try:
            # Parse batch data
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                (sat_images, sat_labels), (drone_images, drone_labels) = batch_data
                
                sat_images = sat_images.to(device)
                drone_images = drone_images.to(device)
                sat_labels = sat_labels.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                
                if hasattr(model, 'module'):
                    outputs = model.module(sat_images, drone_images)
                else:
                    outputs = model(sat_images, drone_images)
                
                # Compute loss
                losses = criterion(outputs, sat_labels)
                total_loss = losses['total']
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Get predictions for metrics
                if 'satellite' in outputs and outputs['satellite'] is not None:
                    sat_preds = outputs['satellite']['predictions']
                    if isinstance(sat_preds, list) and len(sat_preds) > 0:
                        # Use the first valid prediction
                        for pred in sat_preds:
                            if isinstance(pred, torch.Tensor) and pred.ndim == 2:
                                metrics_calc.update(pred, sat_labels, total_loss)
                                break
                
                # Log progress
                if batch_idx % log_interval == 0:
                    logging.info(
                        f"Epoch [{epoch+1}/{total_epochs}] "
                        f"Batch [{batch_idx}/{len(dataloader)}] "
                        f"Loss: {total_loss.item():.6f}"
                    )
        
        except Exception as e:
            logging.error(f"Error in batch {batch_idx}: {e}")
            continue
    
    # Compute epoch metrics
    epoch_time = time.time() - epoch_start_time
    metrics = metrics_calc.compute_metrics()
    metrics['epoch_time'] = epoch_time
    
    return metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Your Innovation Method')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--num-epochs', type=int, help='Number of epochs')
    parser.add_argument('--gpu-ids', type=str, help='GPU IDs to use')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    if args.gpu_ids:
        config['system']['gpu_ids'] = args.gpu_ids
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Set seed
    set_seed(config['system']['seed'])
    
    logger.info(f"Starting training with your innovation method: {config['model']['name']}")
    logger.info(f"Innovation features:")
    logger.info(f"  - Community Clustering: {config['model'].get('use_community_clustering', False)}")
    logger.info(f"  - PCA Alignment: {config['model'].get('use_pca_alignment', False)}")
    logger.info(f"  - Patch Size: {config['model'].get('patch_size', 'N/A')}")
    logger.info(f"  - Clusters: {config['model'].get('num_final_clusters', 'N/A')}")
    logger.info(f"  - PCA Dim: {config['model'].get('target_pca_dim', 'N/A')}")
    
    # Set device
    device = torch.device(f"cuda:{config['system']['gpu_ids']}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info("Creating your innovation model...")
    model = create_model(config)
    model = model.to(device)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Information:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Create dataloader
    logger.info("Creating dataloader...")
    dataloader, class_names, dataset_sizes = make_dataloader(config)
    logger.info(f"Dataset loaded successfully with {len(class_names)} classes")
    
    # Create loss function
    logger.info("Creating loss function...")
    criterion = CombinedLoss(num_classes=len(class_names))
    
    # Create optimizer and scheduler
    logger.info("Creating optimizer and scheduler...")
    optimizer, scheduler = create_optimizer_with_config(model, config)
    
    # Create metrics calculator
    metrics_calc = SimpleMetricsCalculator()
    
    # Training configuration
    num_epochs = config['training']['num_epochs']
    log_interval = config['system'].get('log_interval', 10)
    
    logger.info("Training Configuration:")
    logger.info(f"  Model: {config['model']['name']}")
    logger.info(f"  Classes: {len(class_names)}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Learning Rate: {config['training']['learning_rate']}")
    logger.info(f"  Batch Size: {config['data']['batch_size']}")
    logger.info(f"  Device: {device}")
    
    # Training loop
    logger.info("Starting training with your innovation...")
    
    for epoch in range(num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"{'='*60}")
        
        # Train epoch
        train_metrics = train_epoch(
            model, dataloader, criterion, optimizer, device, 
            metrics_calc, epoch, num_epochs, log_interval
        )
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch results
        logger.info(f"\nEpoch {epoch+1} Results (Your Innovation):")
        logger.info(f"  Train Loss: {train_metrics['avg_loss']:.6f}")
        logger.info(f"  Train Accuracy: {train_metrics['accuracy']:.4f}")
        logger.info(f"  Train Precision: {train_metrics['precision']:.4f}")
        logger.info(f"  Train Recall: {train_metrics['recall']:.4f}")
        logger.info(f"  Train F1-Score: {train_metrics['f1_score']:.4f}")
        logger.info(f"  Learning Rate: {current_lr:.8f}")
        logger.info(f"  Epoch Time: {train_metrics['epoch_time']:.2f}s")
        
        # Save checkpoint periodically
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                config['system']['checkpoint_dir'],
                f"innovation_checkpoint_epoch_{epoch+1}.pth"
            )
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'metrics': train_metrics,
                'config': config
            }, checkpoint_path)
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(config['system']['checkpoint_dir'], "innovation_final_model.pth")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final innovation model saved: {final_model_path}")
    
    logger.info("Training completed with your innovation method!")


if __name__ == "__main__":
    main()
