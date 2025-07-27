#!/usr/bin/env python3
"""
Simplified training script with enhanced logging.
"""

import os
import sys
import time
import argparse
import logging
import datetime
from pathlib import Path
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Direct imports to avoid __init__.py issues
from src.utils.config_utils import load_config
from src.models import create_model
from src.datasets import make_dataloader
from src.losses import CombinedLoss
from src.optimizers import create_optimizer_with_config
from src.utils.training_logger import setup_training_logger


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_memory_usage():
    """Get current memory usage."""
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
    
    import psutil
    ram_memory = psutil.virtual_memory().used / 1024**3  # GB
    
    return gpu_memory, ram_memory


def train_epoch_simple(model, dataloader, criterion, optimizer, device, training_logger, epoch, total_epochs):
    """Simple training epoch with logging."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    epoch_start_time = time.time()
    
    for batch_idx, batch_data in enumerate(dataloader):
        batch_start_time = time.time()
        
        try:
            # Handle different data formats
            if isinstance(batch_data, (list, tuple)):
                if len(batch_data) == 2:
                    # Format: ((sat_images, sat_labels), (drone_images, drone_labels))
                    (sat_images, sat_labels), (drone_images, drone_labels) = batch_data
                elif len(batch_data) == 3:
                    # Format: (sat_images, drone_images, labels)
                    sat_images, drone_images, sat_labels = batch_data
                    drone_labels = sat_labels  # Same labels for both views
                else:
                    raise ValueError(f"Unexpected batch_data length: {len(batch_data)}")
                
                sat_images = sat_images.to(device)
                drone_images = drone_images.to(device)
                sat_labels = sat_labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                if hasattr(model, 'module'):
                    outputs = model.module(sat_images, drone_images)
                else:
                    outputs = model(sat_images, drone_images)
                
                # Compute loss
                losses = criterion(outputs, sat_labels)
                loss = losses['total']
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Batch timing and memory
                batch_time = time.time() - batch_start_time
                gpu_mem, ram_mem = get_memory_usage()
                
                # Log every 10 batches
                if batch_idx % 10 == 0:
                    log_message = (
                        f"Epoch [{epoch+1}/{total_epochs}] "
                        f"Batch [{batch_idx}/{num_batches}] "
                        f"Loss: {loss.item():.6f} "
                        f"Time: {batch_time:.3f}s "
                        f"GPU: {gpu_mem:.1f}MB "
                        f"RAM: {ram_mem:.2f}GB"
                    )
                    
                    print(log_message)
                    training_logger.info(log_message)
                    
                    # Log batch metrics
                    batch_metrics = {
                        'batch_loss': loss.item(),
                        'batch_time': batch_time,
                        'gpu_memory': gpu_mem,
                        'ram_memory': ram_mem,
                        'classification_loss': losses.get('classification', 0.0),
                        'triplet_loss': losses.get('triplet', 0.0)
                    }
                    training_logger.log_metrics(batch_metrics, epoch+1, batch_idx)
                
        except Exception as e:
            error_msg = f"Error in batch {batch_idx}: {e}"
            print(error_msg)
            training_logger.error(error_msg)
            continue
    
    # Epoch summary
    epoch_time = time.time() - epoch_start_time
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    epoch_metrics = {
        'avg_loss': avg_loss,
        'epoch_time': epoch_time,
        'num_batches': num_batches,
        'total_loss': total_loss
    }
    
    return epoch_metrics


def main():
    parser = argparse.ArgumentParser(description='FSRA ViT Improved Training with Logs')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--data-dir', type=str, help='Data directory')
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
    
    # Setup enhanced logging
    experiment_name = f"fsra_vit_improved_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    training_logger = setup_training_logger(log_dir="logs", experiment_name=experiment_name)
    
    set_seed(config['system']['seed'])
    
    # Log training start
    training_config = {
        'model_name': config['model']['name'],
        'batch_size': config['data']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'num_epochs': config['training']['num_epochs'],
        'device': f"cuda:{config['system']['gpu_ids']}" if torch.cuda.is_available() else "cpu",
        'seed': config['system']['seed'],
        'data_dir': config['data']['data_dir']
    }
    training_logger.log_training_start(training_config)
    
    device = torch.device(f"cuda:{config['system']['gpu_ids']}" if torch.cuda.is_available() else "cpu")
    training_logger.info(f"Device: {device}")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    training_logger.info(f"Model created: {config['model']['name']}")
    
    # Create dataloader
    dataloader, class_names, dataset_sizes = make_dataloader(config)
    training_logger.info(f"Dataset loaded: {len(class_names)} classes")
    
    # Create loss function
    criterion = CombinedLoss(num_classes=len(class_names))
    training_logger.info("Loss function created")
    
    # Create optimizer
    optimizer, scheduler = create_optimizer_with_config(model, config)
    training_logger.info(f"Optimizer created: {type(optimizer).__name__}")
    
    # Training parameters
    num_epochs = config['training']['num_epochs']
    
    # Training loop
    training_start_time = time.time()
    
    for epoch in range(num_epochs):
        training_logger.info(f"\n{'='*80}")
        training_logger.info(f"🎯 Epoch {epoch+1}/{num_epochs}")
        training_logger.info(f"{'='*80}")
        
        # Train epoch
        epoch_metrics = train_epoch_simple(
            model, dataloader, criterion, optimizer, device, 
            training_logger, epoch, num_epochs
        )
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Log epoch summary
        training_logger.log_metrics(epoch_metrics, epoch+1)
        training_logger.log_epoch_summary(epoch+1, epoch_metrics)
        
        print(f"Epoch {epoch+1}/{num_epochs} completed - Loss: {epoch_metrics['avg_loss']:.6f}")
    
    # Training completion
    total_training_time = time.time() - training_start_time
    best_metrics = {
        'total_epochs': num_epochs,
        'final_loss': epoch_metrics['avg_loss']
    }
    
    training_logger.log_training_end(total_training_time, best_metrics)
    training_logger.save_metrics_summary()
    
    print(f"🎉 Training completed! Logs saved to: {training_logger.get_log_dir()}")


if __name__ == "__main__":
    main()
