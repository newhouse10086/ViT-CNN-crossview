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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
try:
    from sklearn.metrics import top_k_accuracy_score
except ImportError:
    # Fallback for older sklearn versions
    def top_k_accuracy_score(y_true, y_score, k=5):
        """Simple top-k accuracy implementation."""
        top_k_preds = np.argsort(y_score, axis=1)[:, -k:]
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_preds[i]:
                correct += 1
        return correct / len(y_true)

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


def print_detailed_metrics_with_memory(metrics: dict, epoch: int, total_epochs: int, current_lr: float):
    """Print detailed metrics with memory monitoring in a beautiful format."""
    print(f"\n{'='*90}")
    print(f"📊 EPOCH {epoch}/{total_epochs} COMPREHENSIVE METRICS & MEMORY REPORT")
    print(f"{'='*90}")

    # Performance Metrics
    print(f"🎯 CLASSIFICATION PERFORMANCE:")
    print(f"   Top-1 Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   Top-5 Accuracy:     {metrics['top5_accuracy']:.4f} ({metrics['top5_accuracy']*100:.2f}%)")
    print(f"   AUC Score:          {metrics['auc']:.4f}")

    # Weighted Metrics (considers class imbalance)
    print(f"\n📊 WEIGHTED METRICS (Class-balanced):")
    print(f"   Precision:          {metrics['precision']:.4f}")
    print(f"   Recall:             {metrics['recall']:.4f}")
    print(f"   F1-Score:           {metrics['f1_score']:.4f}")

    # Macro Metrics (treats all classes equally)
    print(f"\n📐 MACRO METRICS (Class-equal):")
    print(f"   Macro Precision:    {metrics['macro_precision']:.4f}")
    print(f"   Macro Recall:       {metrics['macro_recall']:.4f}")
    print(f"   Macro F1-Score:     {metrics['macro_f1']:.4f}")

    # Training Metrics
    print(f"\n📈 TRAINING METRICS:")
    print(f"   Average Loss:       {metrics['avg_loss']:.6f}")
    print(f"   Learning Rate:      {current_lr:.8f}")
    print(f"   Samples Processed:  {metrics['num_samples']:,}")

    # Memory & Performance Metrics
    print(f"\n💾 MEMORY & PERFORMANCE:")
    print(f"   Epoch Time:         {metrics['epoch_time']:.1f}s")
    print(f"   Avg Batch Time:     {metrics['avg_batch_time']:.3f}s")
    print(f"   Time Degradation:   {metrics['time_degradation_percent']:.1f}%")
    print(f"   GPU Memory Inc:     {metrics['gpu_memory_increase']:.1f}MB")
    print(f"   RAM Memory Inc:     {metrics['ram_memory_increase']:.2f}GB")
    print(f"   Final GPU Memory:   {metrics['final_gpu_memory']:.1f}MB")
    print(f"   Final RAM Memory:   {metrics['final_ram_memory']:.2f}GB")
    print(f"   Success Rate:       {metrics['successful_batches']}/{metrics.get('total_batches', 'N/A')} batches")

    # Performance Assessment
    print(f"\n🎪 PERFORMANCE ASSESSMENT:")
    if metrics['accuracy'] >= 0.8:
        acc_status = "🟢 EXCELLENT"
    elif metrics['accuracy'] >= 0.6:
        acc_status = "🟡 GOOD"
    elif metrics['accuracy'] >= 0.4:
        acc_status = "🟠 FAIR"
    else:
        acc_status = "🔴 NEEDS IMPROVEMENT"

    if metrics['auc'] >= 0.9:
        auc_status = "🟢 EXCELLENT"
    elif metrics['auc'] >= 0.8:
        auc_status = "🟡 GOOD"
    elif metrics['auc'] >= 0.7:
        auc_status = "🟠 FAIR"
    else:
        auc_status = "🔴 NEEDS IMPROVEMENT"

    # Memory status
    if metrics['time_degradation_percent'] < 10:
        memory_status = "🟢 STABLE"
    elif metrics['time_degradation_percent'] < 25:
        memory_status = "🟡 MINOR DEGRADATION"
    else:
        memory_status = "🔴 MEMORY LEAK DETECTED"

    print(f"   Accuracy Status:    {acc_status}")
    print(f"   AUC Status:         {auc_status}")
    print(f"   Memory Status:      {memory_status}")

    # Warnings
    if metrics['time_degradation_percent'] > 20:
        print(f"\n⚠️  WARNING: Significant time degradation detected!")
        print(f"   Consider checking for memory leaks or reducing batch size.")

    if metrics['gpu_memory_increase'] > 500:  # 500MB increase
        print(f"\n⚠️  WARNING: Large GPU memory increase detected!")
        print(f"   GPU memory increased by {metrics['gpu_memory_increase']:.1f}MB this epoch.")

    print(f"{'='*90}")


def print_epoch_summary_with_memory(metrics: dict, epoch: int, total_epochs: int):
    """Print a compact epoch summary with memory info."""
    acc_pct = metrics['accuracy'] * 100
    top5_pct = metrics['top5_accuracy'] * 100

    print(f"📋 Epoch {epoch:2d}/{total_epochs} | "
          f"Loss: {metrics['avg_loss']:.4f} | "
          f"Acc: {acc_pct:5.2f}% | "
          f"Top5: {top5_pct:5.2f}% | "
          f"AUC: {metrics['auc']:.3f} | "
          f"F1: {metrics['f1_score']:.3f} | "
          f"Time: {metrics['epoch_time']:.1f}s | "
          f"GPU: +{metrics['gpu_memory_increase']:.0f}MB | "
          f"Degradation: {metrics['time_degradation_percent']:.1f}%")


def print_memory_training_summary(training_history: list, best_accuracy: float, best_auc: float, total_epochs: int):
    """Print comprehensive training summary with memory analysis."""
    print(f"\n{'='*90}")
    print(f"🎊 MEMORY-MONITORED TRAINING COMPLETED - FINAL SUMMARY")
    print(f"{'='*90}")

    if not training_history:
        print("No training history available.")
        return

    # Best metrics
    print(f"🏆 BEST PERFORMANCE ACHIEVED:")
    print(f"   Best Accuracy:      {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"   Best AUC:           {best_auc:.4f}")

    # Final epoch metrics
    final_metrics = training_history[-1]
    print(f"\n📊 FINAL EPOCH METRICS:")
    print(f"   Final Accuracy:     {final_metrics['accuracy']:.4f} ({final_metrics['accuracy']*100:.2f}%)")
    print(f"   Final AUC:          {final_metrics['auc']:.4f}")
    print(f"   Final Loss:         {final_metrics['loss']:.6f}")
    print(f"   Final F1-Score:     {final_metrics['f1_score']:.4f}")
    print(f"   Final Recall:       {final_metrics['recall']:.4f}")
    print(f"   Final Precision:    {final_metrics['precision']:.4f}")

    # Memory analysis
    total_gpu_increase = sum(h['gpu_memory_increase'] for h in training_history)
    max_degradation = max(h['time_degradation'] for h in training_history)
    avg_degradation = np.mean([h['time_degradation'] for h in training_history])

    print(f"\n💾 MEMORY ANALYSIS:")
    print(f"   Total GPU Memory Increase:  {total_gpu_increase:.1f}MB")
    print(f"   Max Time Degradation:       {max_degradation:.1f}%")
    print(f"   Avg Time Degradation:       {avg_degradation:.1f}%")
    print(f"   Final GPU Memory Increase:  {final_metrics['gpu_memory_increase']:.1f}MB")
    print(f"   Final Time Degradation:     {final_metrics['time_degradation']:.1f}%")

    # Memory health assessment
    print(f"\n🔍 MEMORY HEALTH ASSESSMENT:")
    if max_degradation < 10:
        memory_health = "🟢 EXCELLENT - No memory issues detected"
    elif max_degradation < 25:
        memory_health = "🟡 GOOD - Minor degradation observed"
    elif max_degradation < 50:
        memory_health = "🟠 FAIR - Moderate memory issues"
    else:
        memory_health = "🔴 POOR - Significant memory leaks detected"

    print(f"   Memory Health:      {memory_health}")

    # Training progress analysis
    if len(training_history) > 1:
        first_metrics = training_history[0]
        improvement_acc = final_metrics['accuracy'] - first_metrics['accuracy']
        improvement_auc = final_metrics['auc'] - first_metrics['auc']
        improvement_loss = first_metrics['loss'] - final_metrics['loss']

        print(f"\n📈 TRAINING PROGRESS:")
        print(f"   Accuracy Improvement:   {improvement_acc:+.4f} ({improvement_acc*100:+.2f}%)")
        print(f"   AUC Improvement:        {improvement_auc:+.4f}")
        print(f"   Loss Reduction:         {improvement_loss:+.6f}")

    # Performance assessment
    print(f"\n🎯 OVERALL PERFORMANCE ASSESSMENT:")
    if best_accuracy >= 0.9:
        acc_grade = "🟢 EXCELLENT (A+)"
    elif best_accuracy >= 0.8:
        acc_grade = "🟢 VERY GOOD (A)"
    elif best_accuracy >= 0.7:
        acc_grade = "🟡 GOOD (B)"
    elif best_accuracy >= 0.6:
        acc_grade = "🟠 FAIR (C)"
    else:
        acc_grade = "🔴 NEEDS IMPROVEMENT (D)"

    if best_auc >= 0.95:
        auc_grade = "🟢 EXCELLENT (A+)"
    elif best_auc >= 0.9:
        auc_grade = "🟢 VERY GOOD (A)"
    elif best_auc >= 0.8:
        auc_grade = "🟡 GOOD (B)"
    elif best_auc >= 0.7:
        auc_grade = "🟠 FAIR (C)"
    else:
        auc_grade = "🔴 NEEDS IMPROVEMENT (D)"

    print(f"   Accuracy Grade:     {acc_grade}")
    print(f"   AUC Grade:          {auc_grade}")
    print(f"   Memory Grade:       {memory_health.split(' - ')[0]}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if max_degradation > 25:
        print(f"   ⚠️  Consider reducing batch size or model complexity")
        print(f"   ⚠️  Check for memory leaks in data loading or model")
    if total_gpu_increase > 1000:  # 1GB
        print(f"   ⚠️  Large GPU memory increase detected")
        print(f"   ⚠️  Consider gradient checkpointing or model optimization")
    if best_accuracy < 0.7:
        print(f"   📚 Consider longer training or hyperparameter tuning")
        print(f"   📚 Try different learning rates or optimizers")

    print(f"{'='*90}")
    print(f"🚀 Your memory-monitored innovation model is ready!")
    print(f"{'='*90}")


class MemoryMonitoringMetrics:
    """Comprehensive metrics calculator with memory monitoring."""

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
        """Compute comprehensive metrics."""
        if len(self.predictions) == 0 or len(self.targets) == 0:
            return {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                'f1_score': 0.0, 'auc': 0.0, 'top5_accuracy': 0.0,
                'avg_loss': 0.0, 'num_samples': 0
            }

        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities)

        try:
            # Basic metrics
            accuracy = accuracy_score(targets, predictions)
            precision = precision_score(targets, predictions, average='weighted', zero_division=0)
            recall = recall_score(targets, predictions, average='weighted', zero_division=0)
            f1 = f1_score(targets, predictions, average='weighted', zero_division=0)

            # Top-5 accuracy
            try:
                top5_accuracy = top_k_accuracy_score(targets, probabilities, k=5)
            except Exception:
                top5_accuracy = 0.0

            # AUC calculation
            try:
                if self.num_classes == 2:
                    # Binary classification
                    auc = roc_auc_score(targets, probabilities[:, 1])
                else:
                    # Multi-class classification
                    auc = roc_auc_score(targets, probabilities, multi_class='ovr', average='weighted')
            except Exception:
                auc = 0.0

            # Additional metrics
            macro_precision = precision_score(targets, predictions, average='macro', zero_division=0)
            macro_recall = recall_score(targets, predictions, average='macro', zero_division=0)
            macro_f1 = f1_score(targets, predictions, average='macro', zero_division=0)

        except Exception as e:
            print(f"Warning: Error computing metrics: {e}")
            accuracy = precision = recall = f1 = auc = top5_accuracy = 0.0
            macro_precision = macro_recall = macro_f1 = 0.0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'top5_accuracy': top5_accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'avg_loss': np.mean(self.losses) if self.losses else 0.0,
            'num_samples': len(targets)
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
                
                if hasattr(model, 'module'):
                    outputs = model.module(sat_images, drone_images)
                else:
                    outputs = model(sat_images, drone_images)

                # Quick debug: Check outputs and labels structure
                if batch_idx == 0:
                    print(f"TEMP DEBUG: Labels type: {type(sat_labels)}")
                    if hasattr(sat_labels, 'shape'):
                        print(f"  Labels shape: {sat_labels.shape}")
                    elif isinstance(sat_labels, dict):
                        print(f"  Labels dict keys: {list(sat_labels.keys())}")

                    if 'satellite' in outputs and 'predictions' in outputs['satellite']:
                        predictions = outputs['satellite']['predictions']
                        print(f"  Predictions list length: {len(predictions)}")
                        for i, pred in enumerate(predictions):
                            print(f"    Pred {i}: type={type(pred)}")
                            if hasattr(pred, 'shape'):
                                print(f"      Shape: {pred.shape}")

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

    # Training progress tracking
    best_accuracy = 0.0
    best_auc = 0.0
    training_history = []

    logger.info(f"Training: {num_epochs} epochs, LR={config['training']['learning_rate']}, Batch={config['data']['batch_size']}")
    print(f"\n🚀 STARTING MEMORY-MONITORED TRAINING WITH COMPREHENSIVE METRICS")
    print(f"{'='*90}")

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
        
        # Print detailed metrics report with memory monitoring
        print_detailed_metrics_with_memory(train_metrics, epoch+1, num_epochs, current_lr)

        # Also log compact summary
        print_epoch_summary_with_memory(train_metrics, epoch+1, num_epochs)

        # Update best metrics
        if train_metrics['accuracy'] > best_accuracy:
            best_accuracy = train_metrics['accuracy']
            print(f"🎉 NEW BEST ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

        if train_metrics['auc'] > best_auc:
            best_auc = train_metrics['auc']
            print(f"🎉 NEW BEST AUC: {best_auc:.4f}")

        # Store training history
        training_history.append({
            'epoch': epoch + 1,
            'accuracy': train_metrics['accuracy'],
            'auc': train_metrics['auc'],
            'loss': train_metrics['avg_loss'],
            'f1_score': train_metrics['f1_score'],
            'recall': train_metrics['recall'],
            'precision': train_metrics['precision'],
            'gpu_memory_increase': train_metrics['gpu_memory_increase'],
            'time_degradation': train_metrics['time_degradation_percent']
        })
    
    # Print final training summary with memory analysis
    print_memory_training_summary(training_history, best_accuracy, best_auc, num_epochs)

    logger.info("🎊 Memory monitoring training completed!")


if __name__ == "__main__":
    main()
