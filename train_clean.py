#!/usr/bin/env python3
"""
Clean training script with comprehensive metrics for your innovation method.
No debug prints, only essential training metrics.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils import load_config
from src.models import create_model
from src.datasets import make_dataloader
from src.losses import CombinedLoss
from src.optimizers import create_optimizer_with_config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MetricsCalculator:
    """Clean metrics calculator with comprehensive evaluation."""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.losses = []
    
    def update(self, predictions, targets, loss):
        """Update metrics with batch results."""
        with torch.no_grad():
            if isinstance(predictions, torch.Tensor):
                # Get probabilities for AUC
                probs = torch.softmax(predictions, dim=1)
                self.probabilities.extend(probs.detach().cpu().numpy())
                
                # Get predicted classes
                pred_classes = torch.argmax(predictions, dim=1)
                self.predictions.extend(pred_classes.detach().cpu().numpy())
            
            if isinstance(targets, torch.Tensor):
                self.targets.extend(targets.detach().cpu().numpy())
            
            if isinstance(loss, torch.Tensor):
                self.losses.append(loss.detach().item())
            else:
                self.losses.append(loss)
    
    def compute_metrics(self):
        """Compute all metrics."""
        if len(self.predictions) == 0 or len(self.targets) == 0:
            return {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 
                'f1_score': 0.0, 'auc': 0.0, 'avg_loss': 0.0
            }
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities)
        
        try:
            accuracy = accuracy_score(targets, predictions)
            precision = precision_score(targets, predictions, average='weighted', zero_division=0)
            recall = recall_score(targets, predictions, average='weighted', zero_division=0)
            f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
            
            # AUC calculation
            try:
                if self.num_classes == 2:
                    auc = roc_auc_score(targets, probabilities[:, 1])
                else:
                    auc = roc_auc_score(targets, probabilities, multi_class='ovr', average='weighted')
            except:
                auc = 0.0
                
        except Exception:
            accuracy = precision = recall = f1 = auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'avg_loss': np.mean(self.losses) if self.losses else 0.0
        }


def train_epoch(model, dataloader, criterion, optimizer, device, metrics_calc, epoch, total_epochs, log_interval=20):
    """Train one epoch."""
    model.train()
    metrics_calc.reset()
    
    epoch_start_time = time.time()
    successful_batches = 0
    
    for batch_idx, batch_data in enumerate(dataloader):
        try:
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                (sat_images, sat_labels), (drone_images, drone_labels) = batch_data
                
                sat_images = sat_images.to(device)
                drone_images = drone_images.to(device)
                sat_labels = sat_labels.to(device)
                
                optimizer.zero_grad()
                
                if hasattr(model, 'module'):
                    outputs = model.module(sat_images, drone_images)
                else:
                    outputs = model(sat_images, drone_images)
                
                losses = criterion(outputs, sat_labels)
                total_loss = losses['total']
                
                total_loss.backward()
                optimizer.step()
                
                # Update metrics
                if 'satellite' in outputs and outputs['satellite'] is not None:
                    sat_preds = outputs['satellite']['predictions']
                    if isinstance(sat_preds, list) and len(sat_preds) > 0:
                        for pred in reversed(sat_preds):
                            if isinstance(pred, torch.Tensor) and pred.ndim == 2:
                                metrics_calc.update(pred, sat_labels, total_loss)
                                break
                
                successful_batches += 1
                
                if batch_idx % log_interval == 0:
                    logging.info(
                        f"Epoch [{epoch+1}/{total_epochs}] "
                        f"Batch [{batch_idx}/{len(dataloader)}] "
                        f"Loss: {total_loss.item():.6f}"
                    )
        
        except Exception as e:
            logging.error(f"Error in batch {batch_idx}: {e}")
            continue
    
    epoch_time = time.time() - epoch_start_time
    metrics = metrics_calc.compute_metrics()
    metrics['epoch_time'] = epoch_time
    metrics['successful_batches'] = successful_batches
    
    return metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Clean Training for Innovation Method')
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
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    set_seed(config['system']['seed'])
    
    logger.info(f"🚀 TRAINING YOUR INNOVATION: {config['model']['name']}")
    logger.info(f"Innovation Features:")
    logger.info(f"  🔬 Community Clustering: {config['model'].get('use_community_clustering', False)}")
    logger.info(f"  📊 PCA Alignment: {config['model'].get('use_pca_alignment', False)}")
    logger.info(f"  🧩 Patch Size: {config['model'].get('patch_size', 'N/A')}")
    logger.info(f"  🎯 Clusters: {config['model'].get('num_final_clusters', 'N/A')}")
    
    device = torch.device(f"cuda:{config['system']['gpu_ids']}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_params:,} parameters ({total_params * 4 / 1024 / 1024:.1f} MB)")
    
    # Create dataloader
    dataloader, class_names, dataset_sizes = make_dataloader(config)
    logger.info(f"Dataset: {len(class_names)} classes, {sum(dataset_sizes.values())} samples")
    
    # Create loss and optimizer
    criterion = CombinedLoss(num_classes=len(class_names))
    optimizer, scheduler = create_optimizer_with_config(model, config)
    
    # Create metrics calculator
    metrics_calc = MetricsCalculator(num_classes=len(class_names))
    
    num_epochs = config['training']['num_epochs']
    log_interval = config['system'].get('log_interval', 20)
    
    logger.info(f"Training: {num_epochs} epochs, LR={config['training']['learning_rate']}, Batch={config['data']['batch_size']}")
    
    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 Epoch {epoch+1}/{num_epochs} - YOUR INNOVATION")
        logger.info(f"{'='*60}")
        
        train_metrics = train_epoch(
            model, dataloader, criterion, optimizer, device, 
            metrics_calc, epoch, num_epochs, log_interval
        )
        
        if scheduler:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log comprehensive results
        logger.info(f"\n📊 Epoch {epoch+1} Results:")
        logger.info(f"  Loss: {train_metrics['avg_loss']:.6f}")
        logger.info(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {train_metrics['precision']:.4f}")
        logger.info(f"  Recall: {train_metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {train_metrics['f1_score']:.4f}")
        logger.info(f"  AUC: {train_metrics['auc']:.4f}")
        logger.info(f"  LR: {current_lr:.8f}")
        logger.info(f"  Time: {train_metrics['epoch_time']:.1f}s")
        logger.info(f"  Success: {train_metrics['successful_batches']}/{len(dataloader)} ({train_metrics['successful_batches']/len(dataloader)*100:.1f}%)")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(
                config['system']['checkpoint_dir'],
                f"innovation_clean_epoch_{epoch+1}.pth"
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
            
            logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(config['system']['checkpoint_dir'], "innovation_clean_final.pth")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"🎉 Final model saved: {final_model_path}")
    
    logger.info("🎊 TRAINING COMPLETED!")
    logger.info("🚀 Your Community Clustering + PCA innovation is ready!")


if __name__ == "__main__":
    main()
