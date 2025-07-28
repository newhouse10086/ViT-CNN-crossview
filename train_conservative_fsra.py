#!/usr/bin/env python3
"""
Conservative FSRA training - exactly following original FSRA training strategy.
Only algorithm innovation, same training approach.
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
import torch.optim as optim

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config_utils import load_config
from src.models import create_model
from src.datasets import make_dataloader
from src.losses import CombinedLoss  # Use your original combined loss
from src.utils.training_logger import setup_training_logger


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)


def create_fsra_optimizer(model, config):
    """Create optimizer following FSRA strategy."""
    training_config = config['training']
    
    # FSRA uses different learning rates for backbone and classifier
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name.lower() or 'resnet' in name.lower():
            backbone_params.append(param)
        else:
            classifier_params.append(param)
    
    # FSRA strategy: backbone lr = 0.1 * base_lr, classifier lr = base_lr
    base_lr = training_config['learning_rate']
    backbone_lr = base_lr * 0.1
    
    optimizer = optim.SGD([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': classifier_params, 'lr': base_lr}
    ], momentum=0.9, weight_decay=5e-4)
    
    return optimizer


def create_fsra_scheduler(optimizer, config):
    """Create scheduler following FSRA strategy."""
    # FSRA uses step scheduler: decay by 0.1 at epochs 40, 70
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[40, 70], 
        gamma=0.1
    )
    return scheduler


def train_epoch_conservative(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """Conservative training epoch - exactly like original FSRA."""
    model.train()
    
    total_loss = 0.0
    total_cls_loss = 0.0
    total_triplet_loss = 0.0
    total_kl_loss = 0.0
    num_batches = len(dataloader)
    
    for batch_idx, batch_data in enumerate(dataloader):
        try:
            # Data loading - handle both formats
            if isinstance(batch_data, (list, tuple)):
                if len(batch_data) == 2:
                    (sat_images, sat_labels), (drone_images, drone_labels) = batch_data
                elif len(batch_data) == 3:
                    sat_images, drone_images, sat_labels = batch_data
                    drone_labels = sat_labels
                else:
                    continue
                
                sat_images = sat_images.to(device)
                drone_images = drone_images.to(device)
                sat_labels = sat_labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(sat_images, drone_images)
                
                # Compute loss using your combined loss (like original FSRA)
                losses = criterion(outputs, sat_labels)
                loss = losses['total']
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Accumulate losses
                total_loss += loss.item()
                total_cls_loss += losses.get('classification', 0.0)
                total_triplet_loss += losses.get('triplet', 0.0)
                total_kl_loss += losses.get('kl_divergence', 0.0)
                
                # Log every 50 batches (like original FSRA)
                if batch_idx % 50 == 0:
                    print(f"Epoch [{epoch+1}/{total_epochs}] "
                          f"Batch [{batch_idx}/{num_batches}] "
                          f"Loss: {loss.item():.6f} "
                          f"(Cls: {losses.get('classification', 0.0):.4f}, "
                          f"Tri: {losses.get('triplet', 0.0):.4f}, "
                          f"KL: {losses.get('kl_divergence', 0.0):.4f})")
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    # Epoch metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_cls_loss = total_cls_loss / num_batches if num_batches > 0 else 0.0
    avg_triplet_loss = total_triplet_loss / num_batches if num_batches > 0 else 0.0
    avg_kl_loss = total_kl_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        'avg_loss': avg_loss,
        'avg_cls_loss': avg_cls_loss,
        'avg_triplet_loss': avg_triplet_loss,
        'avg_kl_loss': avg_kl_loss
    }


def main():
    parser = argparse.ArgumentParser(description='Conservative FSRA Training')
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
    
    # Setup logging
    experiment_name = f"conservative_fsra_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    training_logger = setup_training_logger(log_dir="logs", experiment_name=experiment_name)
    
    set_seed(config['system']['seed'])
    
    # Log training start
    training_config = {
        'model_name': config['model']['name'],
        'batch_size': config['data']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'num_epochs': config['training']['num_epochs'],
        'training_strategy': 'Conservative_FSRA_Original',
        'device': f"cuda:{config['system']['gpu_ids']}" if torch.cuda.is_available() else "cpu"
    }
    training_logger.log_training_start(training_config)
    
    device = torch.device(f"cuda:{config['system']['gpu_ids']}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    print(f"Model created: {config['model']['name']}")
    
    # Create dataloader
    dataloader, class_names, dataset_sizes = make_dataloader(config)
    print(f"Dataset loaded: {len(class_names)} classes")
    
    # Create loss function (your original combined loss)
    criterion = CombinedLoss(num_classes=len(class_names))
    print("Combined loss function created (original FSRA style)")
    
    # Create optimizer and scheduler (FSRA strategy)
    optimizer = create_fsra_optimizer(model, config)
    scheduler = create_fsra_scheduler(optimizer, config)
    print("FSRA-style optimizer and scheduler created")
    
    # Print optimizer info
    print(f"\nOptimizer Information:")
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"  Group {i}: {len([p for p in param_group['params']])} parameters, lr={param_group['lr']}")
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Train epoch
        epoch_metrics = train_epoch_conservative(
            model, dataloader, criterion, optimizer, device, epoch, num_epochs
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log epoch summary
        training_logger.log_metrics(epoch_metrics, epoch+1)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {epoch_metrics['avg_loss']:.6f}")
        print(f"  Classification Loss: {epoch_metrics['avg_cls_loss']:.6f}")
        print(f"  Triplet Loss: {epoch_metrics['avg_triplet_loss']:.6f}")
        print(f"  KL Loss: {epoch_metrics['avg_kl_loss']:.6f}")
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_path = training_logger.get_log_dir() / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_metrics['avg_loss'],
                'config': config
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_model_path = training_logger.get_log_dir() / "final_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_metrics': epoch_metrics
    }, final_model_path)
    
    print(f"\n🎉 Conservative FSRA training completed!")
    print(f"📁 Logs saved to: {training_logger.get_log_dir()}")
    print(f"💾 Final model saved to: {final_model_path}")
    
    # Training completion
    training_logger.log_training_end(
        total_time=num_epochs * 3600,  # Estimate
        best_metrics=epoch_metrics
    )
    training_logger.save_metrics_summary()


if __name__ == "__main__":
    main()
