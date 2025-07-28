#!/usr/bin/env python3
"""
FSRA-aligned training script - exactly matching original FSRA training setup.
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
import torch.nn.functional as F
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config_utils import load_config
from src.models import create_model
from src.datasets import make_dataloader
from src.utils.training_logger import setup_training_logger


class FSRAAlignedLoss(nn.Module):
    """Loss function exactly matching original FSRA."""
    
    def __init__(self, num_classes, triplet_margin=0.3):
        super(FSRAAlignedLoss, self).__init__()
        self.num_classes = num_classes
        self.triplet_margin = triplet_margin
        
        # Classification loss
        self.cls_loss = nn.CrossEntropyLoss()
        
        # Triplet loss
        self.triplet_loss = nn.TripletMarginLoss(margin=triplet_margin)
        
    def forward(self, outputs, labels):
        """Compute FSRA-aligned loss."""
        losses = {}
        total_loss = 0.0
        
        # Extract predictions
        if 'satellite' in outputs:
            predictions = outputs['satellite']['predictions']
            features_dict = outputs['satellite']['features']
        else:
            raise ValueError("No satellite outputs found")
        
        # 1. Classification Loss (sum over all predictions)
        cls_loss = 0.0
        for i, pred in enumerate(predictions):
            if isinstance(pred, torch.Tensor) and pred.dim() == 2:
                cls_loss += self.cls_loss(pred, labels)
        
        # Average classification loss
        cls_loss = cls_loss / len(predictions)
        total_loss += cls_loss
        losses['cls_loss'] = cls_loss.item()
        
        # 2. Triplet Loss (on global features with hard mining)
        triplet_loss = 0.0
        if 'global' in features_dict:
            global_features = features_dict['global']
        elif 'final' in features_dict:
            global_features = features_dict['final']
        else:
            global_features = list(features_dict.values())[0]
        
        if isinstance(global_features, torch.Tensor) and global_features.dim() == 2:
            # Normalize features for better triplet loss
            normalized_features = F.normalize(global_features, p=2, dim=1)
            
            # Create triplets from batch with hard mining
            batch_size = global_features.shape[0]
            if batch_size >= 3:  # Need at least 3 samples for triplet
                # Compute pairwise distances
                dist_mat = torch.cdist(normalized_features, normalized_features, p=2)
                
                # Create positive and negative masks
                labels_expanded = labels.view(-1, 1)
                pos_mask = (labels_expanded == labels_expanded.T).float()
                neg_mask = (labels_expanded != labels_expanded.T).float()
                
                # Remove self-comparison
                pos_mask = pos_mask - torch.eye(batch_size, device=labels.device)
                
                # Hard positive mining (furthest positive)
                pos_distances = dist_mat * pos_mask
                pos_distances[pos_mask == 0] = -1  # Set non-positives to -1
                hardest_pos_dist, _ = torch.max(pos_distances, dim=1)
                
                # Hard negative mining (closest negative) 
                neg_distances = dist_mat * neg_mask
                neg_distances[neg_mask == 0] = float('inf')  # Set non-negatives to inf
                hardest_neg_dist, _ = torch.min(neg_distances, dim=1)
                
                # Only use valid triplets (where we found both pos and neg)
                valid_triplets = (hardest_pos_dist > -1) & (hardest_neg_dist < float('inf'))
                
                if valid_triplets.sum() > 0:
                    triplet_loss = F.relu(hardest_pos_dist[valid_triplets] - 
                                        hardest_neg_dist[valid_triplets] + 
                                        self.triplet_margin).mean()
        
        total_loss += triplet_loss
        losses['triplet_loss'] = triplet_loss.item() if isinstance(triplet_loss, torch.Tensor) else 0.0
        
        # 3. KL Loss (实现真正的KL散度)
        kl_loss = 0.0
        if len(predictions) > 1:
            # 计算不同预测器输出之间的KL散度，促进一致性
            prob_distributions = [F.softmax(pred, dim=1) for pred in predictions]
            
            # 计算主要预测（第一个）与其他预测之间的KL散度
            main_pred = prob_distributions[0]
            for i, other_pred in enumerate(prob_distributions[1:], 1):
                kl_div = F.kl_div(other_pred.log(), main_pred, reduction='batchmean')
                kl_loss += kl_div
            
            # 平均KL散度
            kl_loss = kl_loss / (len(prob_distributions) - 1)
        
        total_loss += 0.1 * kl_loss  # 降低KL损失权重
        losses['kl_loss'] = kl_loss.item() if isinstance(kl_loss, torch.Tensor) else 0.0
        
        losses['total'] = total_loss
        return losses


def calculate_accuracy(outputs, labels):
    """Calculate accuracy for both satellite and drone independently."""
    sat_acc = 0.0
    drone_acc = 0.0

    # 卫星精度计算
    if 'satellite' in outputs:
        sat_predictions = outputs['satellite']['predictions']
        if len(sat_predictions) > 0:
            sat_pred = sat_predictions[0]  # 使用第一个预测器（global）
            if sat_pred.dim() == 2 and sat_pred.size(1) > 1:
                _, sat_predicted = torch.max(sat_pred.data, 1)
                sat_correct = (sat_predicted == labels).sum().item()
                sat_acc = sat_correct / labels.size(0)

    # 无人机精度计算（如果有独立输出）
    if 'drone' in outputs and outputs['drone'] is not None:
        drone_predictions = outputs['drone']['predictions']
        if len(drone_predictions) > 0:
            drone_pred = drone_predictions[0]  # 使用第一个预测器（global）
            if drone_pred.dim() == 2 and drone_pred.size(1) > 1:
                _, drone_predicted = torch.max(drone_pred.data, 1)
                drone_correct = (drone_predicted == labels).sum().item()
                drone_acc = drone_correct / labels.size(0)
    else:
        # 如果没有独立的无人机输出，使用最后一个预测器作为融合预测
        if 'satellite' in outputs and len(outputs['satellite']['predictions']) > 1:
            final_pred = outputs['satellite']['predictions'][-1]  # 最后一个预测器（融合后）
            if final_pred.dim() == 2 and final_pred.size(1) > 1:
                _, final_predicted = torch.max(final_pred.data, 1)
                final_correct = (final_predicted == labels).sum().item()
                drone_acc = final_correct / labels.size(0)
        else:
            drone_acc = sat_acc  # 作为备选方案

    return sat_acc, drone_acc


def train_epoch_fsra_aligned(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """Training epoch exactly matching FSRA style."""
    model.train()
    
    running_loss = 0.0
    running_cls_loss = 0.0
    running_triplet_loss = 0.0
    running_kl_loss = 0.0
    running_sat_corrects = 0
    running_drone_corrects = 0
    total_samples = 0
    
    for batch_idx, batch_data in enumerate(dataloader):
        try:
            # Handle data format
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
                
                # Compute loss
                losses = criterion(outputs, sat_labels)
                loss = losses['total']
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * sat_images.size(0)
                running_cls_loss += losses['cls_loss'] * sat_images.size(0)
                running_triplet_loss += losses['triplet_loss'] * sat_images.size(0)
                running_kl_loss += losses['kl_loss'] * sat_images.size(0)
                
                # Accuracy
                sat_accuracy, drone_accuracy = calculate_accuracy(outputs, sat_labels)
                running_sat_corrects += sat_accuracy * sat_images.size(0)
                running_drone_corrects += drone_accuracy * sat_images.size(0)
                total_samples += sat_images.size(0)
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    # Epoch statistics
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_cls_loss = running_cls_loss / total_samples if total_samples > 0 else 0.0
    epoch_triplet_loss = running_triplet_loss / total_samples if total_samples > 0 else 0.0
    epoch_kl_loss = running_kl_loss / total_samples if total_samples > 0 else 0.0
    epoch_sat_acc = running_sat_corrects / total_samples if total_samples > 0 else 0.0
    epoch_drone_acc = running_drone_corrects / total_samples if total_samples > 0 else 0.0

    return {
        'loss': epoch_loss,
        'cls_loss': epoch_cls_loss,
        'triplet_loss': epoch_triplet_loss,
        'kl_loss': epoch_kl_loss,
        'sat_accuracy': epoch_sat_acc,
        'drone_accuracy': epoch_drone_acc
    }


def main():
    parser = argparse.ArgumentParser(description='FSRA-Aligned Training')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--data-dir', type=str, help='Data directory')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--num-epochs', type=int, help='Number of epochs')
    parser.add_argument('--gpu-ids', type=str, help='GPU IDs to use')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    if args.gpu_ids:
        config['system']['gpu_ids'] = args.gpu_ids
    
    # Setup logging
    experiment_name = f"fsra_aligned_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    training_logger = setup_training_logger(log_dir="logs", experiment_name=experiment_name)
    
    # Set seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device(f"cuda:{config['system']['gpu_ids']}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    print(f"Model created: {config['model']['name']}")
    
    # Create dataloader
    dataloader, class_names, dataset_sizes = make_dataloader(config)
    print(f"Dataset loaded: {len(class_names)} classes")
    
    # Create FSRA-aligned loss
    criterion = FSRAAlignedLoss(num_classes=len(class_names))
    print("FSRA-aligned loss function created")
    
    # Create optimizer with unified learning rate
    learning_rate = config['training']['learning_rate']
    
    # 统一学习率，避免backbone/other分离的复杂性
    optimizer = optim.SGD(
        model.parameters(), 
        lr=learning_rate,
        momentum=0.9, 
        weight_decay=5e-4
    )
    
    print(f"Optimizer created with unified lr={learning_rate}")
    
    # Create scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70], gamma=0.1)
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs-1}")
        print("-" * 10)
        
        start_time = time.time()
        
        # Train epoch
        metrics = train_epoch_fsra_aligned(
            model, dataloader, criterion, optimizer, device, epoch, num_epochs
        )
        
        # Update scheduler
        scheduler.step()
        
        # Print results in FSRA format
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"train Loss: {metrics['loss']:.4f} "
              f"Cls_Loss:{metrics['cls_loss']:.4f} "
              f"KL_Loss:{metrics['kl_loss']:.4f} "
              f"Triplet_Loss {metrics['triplet_loss']:.4f} "
              f"Satellite_Acc: {metrics['sat_accuracy']:.4f} "
              f"Drone_Acc: {metrics['drone_accuracy']:.4f} "
              f"lr_backbone:{current_lr:.6f} "
              f"lr_other {current_lr:.6f}")
        print(f"Training complete in {epoch_time//60:.0f}m {epoch_time%60:.0f}s")
        
        # Log metrics
        training_logger.log_metrics(metrics, epoch+1)
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_path = training_logger.get_log_dir() / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': metrics
            }, checkpoint_path)
    
    print(f"\n🎉 FSRA-aligned training completed!")


if __name__ == "__main__":
    main()
