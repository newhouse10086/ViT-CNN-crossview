#!/usr/bin/env python3
"""CPU-optimized training script for ViT-CNN-crossview."""

import sys
import os
import time
import argparse
from pathlib import Path
import torch
import torch.nn as nn

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import with error handling
try:
    from src.models import create_model
    from src.datasets import make_dataloader, create_dummy_dataset
    from src.losses import CombinedLoss
    from src.optimizers import create_optimizer_with_config
    from src.utils import (
        setup_logger, get_logger, load_config, validate_config,
        TrainingVisualizer, MetricsCalculator, log_system_info
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please check your installation and dependencies.")
    sys.exit(1)


def create_cpu_optimized_config():
    """Create CPU-optimized configuration."""
    return {
        'model': {
            'name': 'ViTCNN',
            'num_classes': 10,
            'use_pretrained_resnet': False,
            'use_pretrained_vit': False,
            'num_final_clusters': 3,
            'resnet_layers': 18,
            'vit_patch_size': 16,
            'vit_embed_dim': 384
        },
        'data': {
            'batch_size': 4,
            'num_workers': 2,
            'image_height': 128,
            'image_width': 128,
            'views': 2,
            'sample_num': 4
        },
        'training': {
            'num_epochs': 5,
            'learning_rate': 0.01,
            'weight_decay': 0.0005,
            'optimizer': 'sgd',
            'scheduler': 'step',
            'use_fp16': False,
            'use_autocast': False
        },
        'system': {
            'use_gpu': False,
            'seed': 42,
            'log_interval': 5,
            'save_interval': 5,
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs'
        },
        'experiment_name': 'cpu_test'
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, config):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, batch_data in enumerate(dataloader):
        try:
            # Parse batch data
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                (sat_images, sat_labels), (drone_images, drone_labels) = batch_data
                
                # Move to device
                sat_images = sat_images.to(device)
                drone_images = drone_images.to(device)
                sat_labels = sat_labels.to(device)
                
                # Forward pass
                outputs = model(sat_images, drone_images)
                
                # Compute loss
                losses = criterion(outputs, sat_labels)
                loss = losses['total']
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                
                # Get predictions
                if 'satellite' in outputs and outputs['satellite'] is not None:
                    sat_preds = outputs['satellite']['predictions']
                    if isinstance(sat_preds, list):
                        pred = torch.argmax(sat_preds[0], dim=1)
                    else:
                        pred = torch.argmax(sat_preds, dim=1)
                    
                    correct += (pred == sat_labels).sum().item()
                    total += sat_labels.size(0)
                
                # Log progress
                if batch_idx % config.get('system', {}).get('log_interval', 5) == 0:
                    elapsed = time.time() - start_time
                    print(f'Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, '
                          f'Loss: {loss.item():.4f}, '
                          f'Acc: {100.*correct/total if total > 0 else 0:.2f}%, '
                          f'Time: {elapsed:.1f}s')
        
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = 100. * correct / total if total > 0 else 0
    
    return epoch_loss, epoch_acc


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='CPU Training for ViT-CNN-crossview')
    parser.add_argument('--config', type=str, default='config/cpu_config.yaml',
                       help='Path to config file')
    parser.add_argument('--create-dummy-data', action='store_true',
                       help='Create dummy dataset for testing')
    parser.add_argument('--experiment-name', type=str, default='cpu_test',
                       help='Experiment name')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(
        name='ViT-CNN-CPU',
        log_file='logs/cpu_training.log',
        level='INFO'
    )
    logger = get_logger('ViT-CNN-CPU')
    
    logger.info("Starting CPU training for ViT-CNN-crossview")
    
    # Log system info
    log_system_info()
    
    # Force CPU usage
    device = torch.device('cpu')
    logger.info("Using CPU for training")
    
    # Load or create config
    if args.config and Path(args.config).exists():
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = create_cpu_optimized_config()
        logger.info("Using default CPU-optimized config")
    
    # Override experiment name
    config['experiment_name'] = args.experiment_name
    
    # Ensure CPU usage
    config['system']['use_gpu'] = False
    
    try:
        # Create model
        logger.info("Creating model...")
        model = create_model(config)
        model = model.to(device)
        logger.info(f"Model created: {type(model).__name__}")
        
        # Create loss function
        logger.info("Creating loss function...")
        criterion = CombinedLoss(num_classes=config['model']['num_classes'])
        
        # Create optimizer
        logger.info("Creating optimizer...")
        optimizer, scheduler = create_optimizer_with_config(model, config)
        
        # Create dataset
        if args.create_dummy_data:
            logger.info("Creating dummy dataset...")
            train_loader, val_loader = create_dummy_dataset(config)
        else:
            logger.info("Creating data loaders...")
            train_loader = make_dataloader(config, split='train')
            val_loader = make_dataloader(config, split='val')
        
        logger.info(f"Training batches: {len(train_loader)}")
        if val_loader:
            logger.info(f"Validation batches: {len(val_loader)}")
        
        # Training loop
        num_epochs = config['training']['num_epochs']
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch + 1, config
            )
            
            # Update scheduler
            if scheduler:
                scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            logger.info(f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_time:.1f}s")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            # Save checkpoint
            if (epoch + 1) % config.get('system', {}).get('save_interval', 5) == 0:
                checkpoint_dir = Path(config.get('system', {}).get('checkpoint_dir', 'checkpoints'))
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'config': config
                }
                
                checkpoint_path = checkpoint_dir / f'cpu_checkpoint_epoch_{epoch + 1}.pth'
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
