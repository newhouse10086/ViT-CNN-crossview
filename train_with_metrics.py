#!/usr/bin/env python3
"""
Enhanced training script with detailed epoch metrics.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils import load_config, setup_logging, set_seed, get_device_info
from src.utils.metrics_logger import MetricsLogger
from src.models import create_model
from src.datasets import make_dataloader
from src.losses import CombinedLoss
from src.optimizers import create_optimizer_with_config
from src.visualization import TrainingVisualizer


class MetricsCalculator:
    """Calculate training and validation metrics."""
    
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
    """Train for one epoch with detailed metrics."""
    model.train()
    metrics_calc.reset()
    
    epoch_start_time = time.time()
    batch_times = []
    
    for batch_idx, batch_data in enumerate(dataloader):
        batch_start_time = time.time()
        
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
                
                # Record batch time
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                
                # Log progress
                if batch_idx % log_interval == 0:
                    logging.info(
                        f"Epoch [{epoch+1}/{total_epochs}] "
                        f"Batch [{batch_idx}/{len(dataloader)}] "
                        f"Loss: {total_loss.item():.6f} "
                        f"Time: {batch_time:.3f}s"
                    )
        
        except Exception as e:
            logging.error(f"Error in batch {batch_idx}: {e}")
            continue
    
    # Compute epoch metrics
    epoch_time = time.time() - epoch_start_time
    avg_batch_time = np.mean(batch_times) if batch_times else 0.0
    
    metrics = metrics_calc.compute_metrics()
    metrics.update({
        'epoch_time': epoch_time,
        'avg_batch_time': avg_batch_time,
        'batches_processed': len(batch_times)
    })
    
    return metrics


def validate_epoch(model, dataloader, criterion, device, metrics_calc):
    """Validate for one epoch."""
    model.eval()
    metrics_calc.reset()
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            try:
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    (sat_images, sat_labels), (drone_images, drone_labels) = batch_data
                    
                    sat_images = sat_images.to(device)
                    drone_images = drone_images.to(device)
                    sat_labels = sat_labels.to(device)
                    
                    # Forward pass
                    if hasattr(model, 'module'):
                        outputs = model.module(sat_images, drone_images)
                    else:
                        outputs = model(sat_images, drone_images)
                    
                    # Compute loss
                    losses = criterion(outputs, sat_labels)
                    total_loss = losses['total']
                    
                    # Get predictions for metrics
                    if 'satellite' in outputs and outputs['satellite'] is not None:
                        sat_preds = outputs['satellite']['predictions']
                        if isinstance(sat_preds, list) and len(sat_preds) > 0:
                            for pred in sat_preds:
                                if isinstance(pred, torch.Tensor) and pred.ndim == 2:
                                    metrics_calc.update(pred, sat_labels, total_loss)
                                    break
            
            except Exception as e:
                logging.error(f"Error in validation batch {batch_idx}: {e}")
                continue
    
    return metrics_calc.compute_metrics()


def main():
    """Main training function with detailed metrics."""
    parser = argparse.ArgumentParser(description='Enhanced ViT-CNN-crossview Training with Metrics')
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
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Set seed
    set_seed(config['system']['seed'])
    
    # Get device info
    device_info = get_device_info()
    logger.info("System Information:")
    for key, value in device_info.items():
        logger.info(f"  {key}: {value}")
    
    # Set device
    device = torch.device(f"cuda:{config['system']['gpu_ids']}" if torch.cuda.is_available() else "cpu")
    
    # Create model
    logger.info("Creating model...")
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
    
    # Create metrics calculators
    train_metrics_calc = MetricsCalculator()
    
    # Create visualizer and metrics logger
    visualizer = TrainingVisualizer(config)
    metrics_logger = MetricsLogger(
        log_dir=config['system']['log_dir'],
        experiment_name=f"{config['model']['name'].lower()}_{int(time.time())}"
    )
    
    # Training configuration
    num_epochs = config['training']['num_epochs']
    log_interval = config['system'].get('log_interval', 10)
    save_interval = config['system'].get('save_interval', 10)
    
    logger.info("Training Configuration:")
    logger.info(f"  Model: {config['model']['name']}")
    logger.info(f"  Classes: {len(class_names)}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Learning Rate: {config['training']['learning_rate']}")
    logger.info(f"  Batch Size: {config['data']['batch_size']}")
    logger.info(f"  Device: {device}")
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"{'='*60}")
        
        # Train epoch
        train_metrics = train_epoch(
            model, dataloader, criterion, optimizer, device, 
            train_metrics_calc, epoch, num_epochs, log_interval
        )
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch results
        logger.info(f"\nEpoch {epoch+1} Results:")
        logger.info(f"  Train Loss: {train_metrics['avg_loss']:.6f}")
        logger.info(f"  Train Accuracy: {train_metrics['accuracy']:.4f}")
        logger.info(f"  Train Precision: {train_metrics['precision']:.4f}")
        logger.info(f"  Train Recall: {train_metrics['recall']:.4f}")
        logger.info(f"  Train F1-Score: {train_metrics['f1_score']:.4f}")
        logger.info(f"  Learning Rate: {current_lr:.8f}")
        logger.info(f"  Epoch Time: {train_metrics['epoch_time']:.2f}s")
        logger.info(f"  Avg Batch Time: {train_metrics['avg_batch_time']:.3f}s")
        logger.info(f"  Batches Processed: {train_metrics['batches_processed']}")
        
        # Prepare metrics for visualizer
        epoch_metrics = {
            'train_loss': train_metrics['avg_loss'],
            'train_accuracy': train_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'train_f1_score': train_metrics['f1_score'],
            'learning_rate': current_lr,
            'epoch_time': train_metrics['epoch_time'],
            'epoch': epoch + 1
        }
        
        # Update visualizer and metrics logger
        visualizer.update_metrics(epoch + 1, epoch_metrics)
        metrics_logger.log_epoch_metrics(epoch + 1, epoch_metrics)
        
        # Save plots periodically
        if (epoch + 1) % config['evaluation'].get('plot_interval', 10) == 0:
            visualizer.plot_training_curves(save=True, show=False)
        
        # Save checkpoint periodically
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(
                config['system']['checkpoint_dir'],
                f"checkpoint_epoch_{epoch+1}.pth"
            )
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'metrics': epoch_metrics,
                'config': config
            }, checkpoint_path)
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Final plots and summary
    visualizer.plot_training_curves(save=True, show=False)

    # Generate final summary report
    summary_report = metrics_logger.generate_summary_report()
    logger.info("\n" + summary_report)

    # Save final model
    final_model_path = os.path.join(config['system']['checkpoint_dir'], "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved: {final_model_path}")

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
