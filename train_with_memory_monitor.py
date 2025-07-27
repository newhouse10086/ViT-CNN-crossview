#!/usr/bin/env python3
"""
Training script with memory monitoring to detect leaks.
"""

import os
import sys
import time
import argparse
import logging
import gc
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils import load_config
from src.models import create_model
from src.datasets import make_dataloader
from src.losses import CombinedLoss
from src.optimizers import create_optimizer_with_config


def get_memory_usage():
    """Get current memory usage."""
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
    
    import psutil
    ram_memory = psutil.virtual_memory().used / 1024**3  # GB
    
    return gpu_memory, ram_memory


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MemoryMonitoringMetrics:
    """Metrics calculator with memory monitoring."""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, predictions, targets, loss):
        """Update metrics with batch results."""
        with torch.no_grad():
            if isinstance(predictions, torch.Tensor):
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
            return {'accuracy': 0.0, 'avg_loss': 0.0}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        try:
            accuracy = accuracy_score(targets, predictions)
        except:
            accuracy = 0.0
        
        return {
            'accuracy': accuracy,
            'avg_loss': np.mean(self.losses) if self.losses else 0.0
        }


def train_epoch_with_monitoring(model, dataloader, criterion, optimizer, device, metrics_calc, epoch, total_epochs, log_interval=10):
    """Train one epoch with memory monitoring."""
    model.train()
    metrics_calc.reset()
    
    epoch_start_time = time.time()
    successful_batches = 0
    
    # Initial memory
    initial_gpu, initial_ram = get_memory_usage()
    
    batch_times = []
    memory_usage = []
    
    for batch_idx, batch_data in enumerate(dataloader):
        batch_start_time = time.time()
        
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
                
                # Memory monitoring
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                
                current_gpu, current_ram = get_memory_usage()
                memory_usage.append((current_gpu, current_ram))
                
                # Force garbage collection every 10 batches
                if batch_idx % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Detailed logging
                if batch_idx % log_interval == 0:
                    avg_batch_time = np.mean(batch_times[-10:]) if len(batch_times) >= 10 else np.mean(batch_times)
                    gpu_increase = current_gpu - initial_gpu
                    ram_increase = current_ram - initial_ram
                    
                    logging.info(
                        f"Epoch [{epoch+1}/{total_epochs}] "
                        f"Batch [{batch_idx}/{len(dataloader)}] "
                        f"Loss: {total_loss.item():.6f} "
                        f"Time: {batch_time:.3f}s "
                        f"AvgTime: {avg_batch_time:.3f}s "
                        f"GPU: +{gpu_increase:.1f}MB "
                        f"RAM: +{ram_increase:.2f}GB"
                    )
                
                # Clean up batch data
                del sat_images, drone_images, sat_labels, outputs, losses, total_loss
        
        except Exception as e:
            logging.error(f"Error in batch {batch_idx}: {e}")
            continue
    
    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    epoch_time = time.time() - epoch_start_time
    metrics = metrics_calc.compute_metrics()
    
    # Memory analysis
    final_gpu, final_ram = get_memory_usage()
    
    # Batch time analysis
    if len(batch_times) > 10:
        first_10_avg = np.mean(batch_times[:10])
        last_10_avg = np.mean(batch_times[-10:])
        time_degradation = (last_10_avg - first_10_avg) / first_10_avg * 100
    else:
        time_degradation = 0
    
    metrics.update({
        'epoch_time': epoch_time,
        'successful_batches': successful_batches,
        'avg_batch_time': np.mean(batch_times) if batch_times else 0,
        'time_degradation_percent': time_degradation,
        'gpu_memory_increase': final_gpu - initial_gpu,
        'ram_memory_increase': final_ram - initial_ram,
        'final_gpu_memory': final_gpu,
        'final_ram_memory': final_ram
    })
    
    return metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Memory Monitoring Training')
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
    
    logger.info(f"🔍 MEMORY MONITORING TRAINING: {config['model']['name']}")
    
    device = torch.device(f"cuda:{config['system']['gpu_ids']}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Create dataloader
    dataloader, class_names, dataset_sizes = make_dataloader(config)
    logger.info(f"Dataset: {len(class_names)} classes")
    
    # Create loss and optimizer
    criterion = CombinedLoss(num_classes=len(class_names))
    optimizer, scheduler = create_optimizer_with_config(model, config)
    
    # Create metrics calculator
    metrics_calc = MemoryMonitoringMetrics(num_classes=len(class_names))
    
    num_epochs = config['training']['num_epochs']
    log_interval = config['system'].get('log_interval', 10)
    
    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 Epoch {epoch+1}/{num_epochs} - MEMORY MONITORING")
        logger.info(f"{'='*80}")
        
        train_metrics = train_epoch_with_monitoring(
            model, dataloader, criterion, optimizer, device, 
            metrics_calc, epoch, num_epochs, log_interval
        )
        
        if scheduler:
            scheduler.step()
        
        # Log comprehensive results
        logger.info(f"\n📊 Epoch {epoch+1} Results:")
        logger.info(f"  Loss: {train_metrics['avg_loss']:.6f}")
        logger.info(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        logger.info(f"  Avg Batch Time: {train_metrics['avg_batch_time']:.3f}s")
        logger.info(f"  Time Degradation: {train_metrics['time_degradation_percent']:.1f}%")
        logger.info(f"  GPU Memory Increase: {train_metrics['gpu_memory_increase']:.1f}MB")
        logger.info(f"  RAM Memory Increase: {train_metrics['ram_memory_increase']:.2f}GB")
        logger.info(f"  Final GPU Memory: {train_metrics['final_gpu_memory']:.1f}MB")
        
        # Warning for memory issues
        if train_metrics['time_degradation_percent'] > 20:
            logger.warning(f"⚠️  Significant time degradation detected: {train_metrics['time_degradation_percent']:.1f}%")
        
        if train_metrics['gpu_memory_increase'] > 500:  # 500MB increase
            logger.warning(f"⚠️  Large GPU memory increase: {train_metrics['gpu_memory_increase']:.1f}MB")
    
    logger.info("🎊 Memory monitoring training completed!")


if __name__ == "__main__":
    main()
