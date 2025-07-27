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
        return correct / len(y_true), roc_auc_score

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


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    """Print a progress bar."""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    if iteration == total:
        print()  # Print new line on completion


def print_detailed_metrics(metrics: dict, epoch: int, total_epochs: int, current_lr: float):
    """Print detailed metrics in a beautiful format."""
    print(f"\n{'='*80}")
    print(f"📊 EPOCH {epoch}/{total_epochs} DETAILED METRICS REPORT")
    print(f"{'='*80}")

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

    # Performance Indicators
    print(f"\n⚡ PERFORMANCE INDICATORS:")
    print(f"   Epoch Time:         {metrics['epoch_time']:.1f}s")
    print(f"   Samples/Second:     {metrics['num_samples']/metrics['epoch_time']:.1f}")
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

    print(f"   Accuracy Status:    {acc_status}")
    print(f"   AUC Status:         {auc_status}")

    print(f"{'='*80}")


def print_epoch_summary(metrics: dict, epoch: int, total_epochs: int):
    """Print a compact epoch summary."""
    acc_pct = metrics['accuracy'] * 100
    top5_pct = metrics['top5_accuracy'] * 100

    print(f"📋 Epoch {epoch:2d}/{total_epochs} | "
          f"Loss: {metrics['avg_loss']:.4f} | "
          f"Acc: {acc_pct:5.2f}% | "
          f"Top5: {top5_pct:5.2f}% | "
          f"AUC: {metrics['auc']:.3f} | "
          f"F1: {metrics['f1_score']:.3f} | "
          f"Time: {metrics['epoch_time']:.1f}s")


def print_training_summary(training_history: list, best_accuracy: float, best_auc: float, total_epochs: int):
    """Print comprehensive training summary."""
    print(f"\n{'='*80}")
    print(f"🎊 TRAINING COMPLETED - FINAL SUMMARY")
    print(f"{'='*80}")

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

    # Training progress analysis
    if len(training_history) > 1:
        first_metrics = training_history[0]
        improvement_acc = final_metrics['accuracy'] - first_metrics['accuracy']
        improvement_auc = final_metrics['auc'] - first_metrics['auc']
        improvement_loss = first_metrics['loss'] - final_metrics['loss']  # Loss should decrease

        print(f"\n📈 TRAINING PROGRESS:")
        print(f"   Accuracy Improvement:   {improvement_acc:+.4f} ({improvement_acc*100:+.2f}%)")
        print(f"   AUC Improvement:        {improvement_auc:+.4f}")
        print(f"   Loss Reduction:         {improvement_loss:+.6f}")

    # Performance assessment
    print(f"\n🎯 PERFORMANCE ASSESSMENT:")
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

    # Training statistics
    total_samples = sum(h.get('num_samples', 0) for h in training_history if 'num_samples' in h)
    avg_epoch_time = np.mean([h.get('epoch_time', 0) for h in training_history if 'epoch_time' in h])

    print(f"\n📊 TRAINING STATISTICS:")
    print(f"   Total Epochs:       {total_epochs}")
    print(f"   Total Samples:      {total_samples:,}")
    print(f"   Avg Epoch Time:     {avg_epoch_time:.1f}s")
    print(f"   Total Training Time: {avg_epoch_time * total_epochs:.1f}s ({avg_epoch_time * total_epochs / 60:.1f}m)")

    print(f"{'='*80}")
    print(f"🚀 Your innovation model is ready for deployment!")
    print(f"{'='*80}")


class MetricsCalculator:
    """Comprehensive metrics calculator with detailed evaluation."""

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
                    progress = (batch_idx + 1) / len(dataloader) * 100
                    logging.info(
                        f"Epoch [{epoch+1}/{total_epochs}] "
                        f"Batch [{batch_idx+1}/{len(dataloader)}] ({progress:.1f}%) "
                        f"Loss: {total_loss.item():.6f}"
                    )

                    # Print progress bar
                    if batch_idx % (log_interval * 2) == 0:  # Less frequent progress bar
                        print_progress_bar(batch_idx + 1, len(dataloader),
                                         prefix=f'Epoch {epoch+1}/{total_epochs}',
                                         suffix=f'Loss: {total_loss.item():.4f}',
                                         length=40)
        
        except Exception as e:
            logging.error(f"Error in batch {batch_idx}: {e}")
            continue
    
    epoch_time = time.time() - epoch_start_time
    metrics = metrics_calc.compute_metrics()
    metrics['epoch_time'] = epoch_time
    metrics['successful_batches'] = successful_batches
    metrics['total_batches'] = len(dataloader)

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

    # Training progress tracking
    best_accuracy = 0.0
    best_auc = 0.0
    training_history = []

    print(f"\n🚀 STARTING TRAINING WITH COMPREHENSIVE METRICS")
    print(f"{'='*80}")

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

        # Print detailed metrics report
        print_detailed_metrics(train_metrics, epoch+1, num_epochs, current_lr)

        # Also log compact summary
        print_epoch_summary(train_metrics, epoch+1, num_epochs)

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
            'precision': train_metrics['precision']
        })

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

    # Print final training summary
    print_training_summary(training_history, best_accuracy, best_auc, num_epochs)

    logger.info("🎊 TRAINING COMPLETED!")
    logger.info("🚀 Your Community Clustering + PCA innovation is ready!")


if __name__ == "__main__":
    main()
