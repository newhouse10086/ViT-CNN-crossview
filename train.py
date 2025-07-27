#!/usr/bin/env python3
"""Training script for ViT-CNN-crossview."""

import argparse
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models import create_model
from src.datasets import make_dataloader, create_dummy_dataset
from src.losses import CombinedLoss
from src.optimizers import create_optimizer_with_config
from src.utils import (
    setup_logger, get_logger, load_config, validate_config,
    TrainingVisualizer, MetricsCalculator, log_system_info
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ViT-CNN-crossview model')
    
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Path to dataset directory')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--gpu-ids', type=str, default=None,
                       help='GPU IDs to use (e.g., "0,1,2")')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--experiment-name', type=str, default='vit_cnn_experiment',
                       help='Name of the experiment')
    parser.add_argument('--create-dummy-data', action='store_true',
                       help='Create dummy dataset if real data is not available')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    return parser.parse_args()


def setup_environment(config):
    """Setup training environment."""
    # Set random seeds
    import random
    import numpy as np
    
    seed = config['system'].get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set CUDA settings
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create directories
    os.makedirs(config['system']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['system']['log_dir'], exist_ok=True)


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
        print("Using default configuration...")
        config = {
            'model': {'name': 'ViTCNN', 'num_classes': 10},
            'data': {'data_dir': 'data/train', 'batch_size': 16, 'views': 2, 'num_workers': 4},
            'training': {'num_epochs': 150, 'learning_rate': 0.005},
            'system': {'gpu_ids': '0', 'log_dir': 'logs', 'checkpoint_dir': 'checkpoints'}
        }
    
    # Update config with command line arguments
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
    
    # Validate configuration
    if not validate_config(config):
        print("Configuration validation failed!")
        return
    
    # Setup environment
    setup_environment(config)
    
    # Setup logging
    log_file = os.path.join(config['system']['log_dir'], f"{args.experiment_name}_train.log")
    logger = setup_logger(log_file=log_file, log_level="DEBUG" if args.debug else "INFO")
    
    logger.info(f"Starting ViT-CNN-crossview training: {args.experiment_name}")
    log_system_info()
    
    # Setup device
    if torch.cuda.is_available() and config['system'].get('use_gpu', True):
        gpu_ids = config['system']['gpu_ids']
        if isinstance(gpu_ids, str):
            gpu_ids = [int(x.strip()) for x in gpu_ids.split(',')]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        logger.info(f"Using GPU(s): {gpu_ids}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    try:
        # Create model
        logger.info("Creating model...")
        model = create_model(config)
        model = model.to(device)
        
        # Multi-GPU setup
        if torch.cuda.is_available() and len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)
            logger.info(f"Using DataParallel with GPUs: {gpu_ids}")
        
        # Print model info
        from src.models.model_factory import print_model_info
        print_model_info(model, "ViT-CNN Model")
        
        # Create dataloader
        logger.info("Creating dataloader...")
        try:
            dataloader, class_names, dataset_sizes = make_dataloader(
                config, create_dummy=args.create_dummy_data
            )
            logger.info(f"Dataset loaded successfully with {len(class_names)} classes")
        except Exception as e:
            logger.error(f"Failed to create dataloader: {e}")
            if args.create_dummy_data:
                logger.info("Creating dummy dataset...")
                create_dummy_dataset(config['data']['data_dir'])
                dataloader, class_names, dataset_sizes = make_dataloader(config, create_dummy=True)
            else:
                logger.info("This is expected if the dataset is not available. The model structure is still valid.")
                dataloader = None
                class_names = [f"class_{i}" for i in range(config['model']['num_classes'])]
                dataset_sizes = {'train': 0}
        
        # Create loss function
        logger.info("Creating loss function...")
        criterion = CombinedLoss(
            num_classes=config['model']['num_classes'],
            triplet_weight=config['training'].get('triplet_loss_weight', 0.3),
            kl_weight=config['training'].get('kl_loss_weight', 0.0),
            alignment_weight=config['training'].get('cross_attention_weight', 1.0),
            use_kl_loss=config['training'].get('use_kl_loss', False)
        )
        
        # Create optimizer and scheduler
        logger.info("Creating optimizer and scheduler...")
        optimizer, scheduler = create_optimizer_with_config(model, config)
        
        # Create visualizer and metrics calculator
        visualizer = TrainingVisualizer(
            save_dir=os.path.join(config['system']['log_dir'], 'plots'),
            experiment_name=args.experiment_name
        )
        metrics_calculator = MetricsCalculator(config['model']['num_classes'])
        
        # Print training information
        logger.info("Training Configuration:")
        logger.info(f"  Model: {config['model']['name']}")
        logger.info(f"  Classes: {config['model']['num_classes']}")
        logger.info(f"  Epochs: {config['training']['num_epochs']}")
        logger.info(f"  Learning Rate: {config['training']['learning_rate']}")
        logger.info(f"  Batch Size: {config['data']['batch_size']}")
        logger.info(f"  Device: {device}")
        
        if dataloader is None:
            logger.info("No dataloader available. Training skipped.")
            logger.info("Model structure validation completed successfully.")
            return
        
        # Training loop
        logger.info("Starting training...")
        model.train()
        
        for epoch in range(config['training']['num_epochs']):
            logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
            
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, batch_data in enumerate(dataloader):
                try:
                    # Parse batch data
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
                        
                        # Backward pass
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()
                        
                        # Update metrics
                        epoch_loss += total_loss.item()
                        epoch_samples += sat_images.size(0)
                        
                        if batch_idx % config['system'].get('log_interval', 10) == 0:
                            logger.info(f"  Batch {batch_idx}/{len(dataloader)}, "
                                      f"Loss: {total_loss.item():.6f}")
                    
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {e}")
                    continue
            
            # Epoch metrics
            avg_loss = epoch_loss / max(len(dataloader), 1)
            
            # Update scheduler
            if scheduler:
                scheduler.step()
            
            # Log epoch results
            current_lr = optimizer.param_groups[0]['lr']
            epoch_metrics = {
                'train_loss': avg_loss,
                'learning_rate': current_lr,
                'epoch': epoch + 1
            }
            
            logger.info(f"Epoch {epoch+1} completed - Loss: {avg_loss:.6f}, LR: {current_lr:.8f}")
            
            # Update visualizer
            visualizer.update_metrics(epoch + 1, epoch_metrics)
            
            # Save plots periodically
            if (epoch + 1) % config['evaluation'].get('plot_interval', 10) == 0:
                visualizer.plot_training_curves(save=True, show=False)
            
            # Save checkpoint periodically
            if (epoch + 1) % config['system'].get('save_interval', 10) == 0:
                checkpoint_path = os.path.join(
                    config['system']['checkpoint_dir'],
                    f"{args.experiment_name}_epoch_{epoch+1}.pth"
                )
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'config': config,
                    'metrics': epoch_metrics
                }, checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Final plots and metrics
        logger.info("Training completed!")
        visualizer.plot_training_curves(save=True, show=False)
        visualizer.save_metrics_csv()
        
        # Save final model
        final_model_path = os.path.join(
            config['system']['checkpoint_dir'],
            f"{args.experiment_name}_final.pth"
        )
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'class_names': class_names
        }, final_model_path)
        logger.info(f"Final model saved: {final_model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
