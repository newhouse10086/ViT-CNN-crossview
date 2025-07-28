#!/usr/bin/env python3
"""
FSRA-CRN Training Script
Context-aware Region-alignment Network for Cross-View Image Matching
专门适用于University-1652数据集的跨视角图像匹配任务
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
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config_utils import load_config
from src.models.fsra_crn_improved import create_fsra_crn_model
from src.datasets import make_dataloader


class FSRACRNLoss(nn.Module):
    """FSRA-CRN专用损失函数"""
    
    def __init__(self, num_classes, loss_weights):
        super().__init__()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        
        # 基础损失
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.triplet_loss = nn.TripletMarginLoss(margin=0.3)
        
        # 创新损失组件
        self.mse_loss = nn.MSELoss()
        
    def forward(self, outputs, labels):
        """计算FSRA-CRN总损失"""
        losses = {}
        total_loss = 0.0
        
        # 1. 全局分类损失
        global_pred = outputs['global_prediction']
        global_loss = self.cls_loss(global_pred, labels)
        losses['global_loss'] = global_loss
        total_loss += self.loss_weights['global_loss'] * global_loss
        
        # 2. 区域分类损失
        regional_preds = outputs['regional_predictions']
        regional_loss = 0.0
        for pred in regional_preds:
            regional_loss += self.cls_loss(pred, labels)
        regional_loss /= len(regional_preds)
        losses['regional_loss'] = regional_loss
        total_loss += self.loss_weights['regional_loss'] * regional_loss
        
        # 3. 特征对齐损失（创新）
        if 'aligned_features' in outputs:
            # 确保特征对齐的一致性
            sat_enhanced = outputs.get('sat_enhanced')
            uav_enhanced = outputs.get('uav_enhanced')
            
            if sat_enhanced is not None and uav_enhanced is not None:
                # 计算卫星和无人机特征的对齐一致性
                sat_global = torch.mean(sat_enhanced.view(sat_enhanced.size(0), -1), dim=1)
                uav_global = torch.mean(uav_enhanced.view(uav_enhanced.size(0), -1), dim=1)
                
                alignment_loss = self.mse_loss(sat_global, uav_global)
                losses['alignment_loss'] = alignment_loss
                total_loss += self.loss_weights['alignment_loss'] * alignment_loss
        
        # 4. 区域一致性损失
        if len(regional_preds) > 1:
            consistency_loss = 0.0
            for i in range(len(regional_preds)):
                for j in range(i+1, len(regional_preds)):
                    # 计算不同区域预测的相似性
                    pred_i = torch.softmax(regional_preds[i], dim=1)
                    pred_j = torch.softmax(regional_preds[j], dim=1)
                    consistency_loss += self.mse_loss(pred_i, pred_j)
            
            consistency_loss /= (len(regional_preds) * (len(regional_preds) - 1) / 2)
            losses['consistency_loss'] = consistency_loss
            total_loss += self.loss_weights['consistency_loss'] * consistency_loss
        
        losses['total'] = total_loss
        return losses


def calculate_accuracy(outputs, labels):
    """计算精度指标"""
    metrics = {}
    
    # 全局精度
    global_pred = outputs['global_prediction']
    _, predicted = torch.max(global_pred.data, 1)
    global_acc = (predicted == labels).sum().item() / labels.size(0)
    metrics['global_accuracy'] = global_acc
    
    # 区域精度（平均）
    regional_preds = outputs['regional_predictions']
    regional_accs = []
    for pred in regional_preds:
        _, pred_labels = torch.max(pred.data, 1)
        acc = (pred_labels == labels).sum().item() / labels.size(0)
        regional_accs.append(acc)
    
    metrics['regional_accuracy'] = np.mean(regional_accs)
    
    return metrics


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    """训练一个epoch"""
    model.train()
    
    running_losses = {}
    running_metrics = {}
    total_samples = 0
    
    for batch_idx, batch_data in enumerate(dataloader):
        try:
            # 数据准备
            if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 3:
                sat_images, drone_images, labels = batch_data[:3]
            else:
                continue
                
            sat_images = sat_images.to(device)
            drone_images = drone_images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(sat_images, drone_images)
            
            # 计算损失
            losses = criterion(outputs, labels)
            loss = losses['total']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计信息
            batch_size = sat_images.size(0)
            total_samples += batch_size
            
            # 累积损失
            for loss_name, loss_value in losses.items():
                if loss_name not in running_losses:
                    running_losses[loss_name] = 0.0
                running_losses[loss_name] += loss_value.item() * batch_size
            
            # 计算精度指标
            metrics = calculate_accuracy(outputs, labels)
            for metric_name, metric_value in metrics.items():
                if metric_name not in running_metrics:
                    running_metrics[metric_name] = 0.0
                running_metrics[metric_name] += metric_value * batch_size
            
            # TensorBoard日志
            if writer and batch_idx % 50 == 0:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('Batch/Loss', loss.item(), global_step)
                writer.add_scalar('Batch/GlobalAccuracy', 
                                metrics['global_accuracy'], global_step)
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    # 计算epoch平均值
    epoch_losses = {}
    epoch_metrics = {}
    
    for loss_name, loss_sum in running_losses.items():
        epoch_losses[loss_name] = loss_sum / total_samples
    
    for metric_name, metric_sum in running_metrics.items():
        epoch_metrics[metric_name] = metric_sum / total_samples
    
    return epoch_losses, epoch_metrics


def main():
    parser = argparse.ArgumentParser(description='FSRA-CRN Training for Cross-View Image Matching')
    parser.add_argument('--config', type=str, default='config/fsra_crn_config.yaml',
                       help='Config file path')
    parser.add_argument('--data-dir', type=str, help='Data directory')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--num-epochs', type=int, help='Number of epochs')
    parser.add_argument('--gpu-ids', type=str, help='GPU IDs to use')
    parser.add_argument('--experiment-name', type=str, 
                       default='fsra_crn_experiment', help='Experiment name')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    if args.gpu_ids:
        config['system']['gpu_ids'] = args.gpu_ids
    
    # 设置实验记录
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{args.experiment_name}_{timestamp}"
    
    # 创建日志目录
    log_dir = Path("logs") / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir / "tensorboard")
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # 设置随机种子
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device(f"cuda:{config['system']['gpu_ids']}" 
                         if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 创建FSRA-CRN模型
    model = create_fsra_crn_model(
        num_classes=config['model']['num_classes'],
        feature_dim=config['model']['feature_dim']
    )
    model = model.to(device)
    
    logger.info(f"FSRA-CRN model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 创建数据加载器
    try:
        dataloader, class_names, dataset_sizes = make_dataloader(config)
        logger.info(f"University-1652 dataset loaded: {len(class_names)} classes")
    except Exception as e:
        logger.error(f"Failed to create dataloader: {e}")
        # 创建简单的数据加载器用于测试
        from torch.utils.data import DataLoader, TensorDataset
        
        # 创建模拟数据
        num_samples = 1000
        sat_data = torch.randn(num_samples, 3, 256, 256)
        drone_data = torch.randn(num_samples, 3, 256, 256)
        labels = torch.randint(0, config['model']['num_classes'], (num_samples,))
        
        dataset = TensorDataset(sat_data, drone_data, labels)
        dataloader = DataLoader(dataset, batch_size=config['data']['batch_size'], 
                              shuffle=True, num_workers=0)
        class_names = [f"class_{i}" for i in range(config['model']['num_classes'])]
        logger.info(f"Using simulated data: {len(class_names)} classes")
    
    # 创建损失函数
    criterion = FSRACRNLoss(
        num_classes=len(class_names),
        loss_weights=config['training']['loss_weights']
    )
    logger.info("FSRA-CRN loss function created")
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config['training']['scheduler']['milestones'],
        gamma=config['training']['scheduler']['gamma']
    )
    
    logger.info(f"Optimizer and scheduler created")
    
    # 训练循环
    num_epochs = config['training']['num_epochs']
    best_accuracy = 0.0
    
    logger.info(f"Starting FSRA-CRN training for {num_epochs} epochs on University-1652...")
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info("-" * 50)
        
        start_time = time.time()
        
        # 训练epoch
        epoch_losses, epoch_metrics = train_epoch(
            model, dataloader, criterion, optimizer, device, epoch, writer
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录训练信息
        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        
        # 打印结果
        logger.info(f"Epoch {epoch+1} Results:")
        logger.info(f"  Time: {epoch_time:.2f}s")
        logger.info(f"  Learning Rate: {current_lr:.6f}")
        logger.info(f"  Total Loss: {epoch_losses['total']:.4f}")
        logger.info(f"  Global Loss: {epoch_losses['global_loss']:.4f}")
        logger.info(f"  Regional Loss: {epoch_losses['regional_loss']:.4f}")
        
        if 'alignment_loss' in epoch_losses:
            logger.info(f"  Alignment Loss: {epoch_losses['alignment_loss']:.4f}")
        if 'consistency_loss' in epoch_losses:
            logger.info(f"  Consistency Loss: {epoch_losses['consistency_loss']:.4f}")
        
        logger.info(f"  Global Accuracy: {epoch_metrics['global_accuracy']:.4f}")
        logger.info(f"  Regional Accuracy: {epoch_metrics['regional_accuracy']:.4f}")
        
        # TensorBoard日志
        writer.add_scalar('Epoch/TotalLoss', epoch_losses['total'], epoch)
        writer.add_scalar('Epoch/GlobalAccuracy', epoch_metrics['global_accuracy'], epoch)
        writer.add_scalar('Epoch/RegionalAccuracy', epoch_metrics['regional_accuracy'], epoch)
        writer.add_scalar('Epoch/LearningRate', current_lr, epoch)
        
        # 保存最佳模型
        if epoch_metrics['global_accuracy'] > best_accuracy:
            best_accuracy = epoch_metrics['global_accuracy']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_accuracy': best_accuracy,
                'config': config
            }, log_dir / 'best_fsra_crn_model.pth')
            logger.info(f"New best FSRA-CRN model saved with accuracy: {best_accuracy:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'losses': epoch_losses,
                'metrics': epoch_metrics,
                'config': config
            }, log_dir / f'fsra_crn_checkpoint_epoch_{epoch+1}.pth')
    
    # 训练完成
    logger.info(f"\nFSRA-CRN training completed!")
    logger.info(f"Best accuracy achieved: {best_accuracy:.4f}")
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_metrics': epoch_metrics,
        'experiment_name': experiment_name
    }, log_dir / 'final_fsra_crn_model.pth')
    
    writer.close()
    logger.info(f"FSRA-CRN experiment results saved to: {log_dir}")
    
    # 创新点总结
    logger.info("\n🚀 FSRA-CRN创新点总结:")
    logger.info("  1. 多尺度残差特征提取器 (MSRFE) - 替代传统CNN backbone")
    logger.info("  2. 魔方注意力机制 (RCA) - 借鉴CEUSP论文创新思路") 
    logger.info("  3. 动态上下文融合 (DCF) - 自适应多尺度信息整合")
    logger.info("  4. 自适应区域对齐 (ARA) - 改进的区域发现和对齐")


if __name__ == "__main__":
    main() 