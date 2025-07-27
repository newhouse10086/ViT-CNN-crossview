"""Trainer module for ViT-CNN-crossview."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from ..utils import MetricsCalculator, AverageMeter, TrainingVisualizer
from ..losses import CombinedLoss

logger = logging.getLogger(__name__)


class Trainer:
    """Main trainer class for ViT-CNN-crossview."""
    
    def __init__(self, model: nn.Module, criterion: CombinedLoss,
                 optimizer: torch.optim.Optimizer, scheduler: Optional[Any] = None,
                 device: torch.device = torch.device('cpu'),
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to use for training
            config: Configuration dictionary
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config or {}
        
        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.training_history = []
        
        # Metrics and visualization
        self.metrics_calculator = MetricsCalculator(
            num_classes=self.config.get('model', {}).get('num_classes', 10)
        )
        
        if 'experiment_name' in self.config:
            self.visualizer = TrainingVisualizer(
                save_dir=self.config.get('system', {}).get('log_dir', 'logs'),
                experiment_name=self.config['experiment_name']
            )
        else:
            self.visualizer = None
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training dataloader
            
        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        
        # Metrics tracking
        loss_meter = AverageMeter('Loss')
        batch_time = AverageMeter('Time')
        
        self.metrics_calculator.reset()
        
        end_time = time.time()
        
        for batch_idx, batch_data in enumerate(dataloader):
            try:
                # Parse batch data
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    (sat_images, sat_labels), (drone_images, drone_labels) = batch_data
                    
                    # Move to device
                    sat_images = sat_images.to(self.device)
                    drone_images = drone_images.to(self.device)
                    sat_labels = sat_labels.to(self.device)
                    
                    # Forward pass
                    if hasattr(self.model, 'module'):
                        outputs = self.model.module(sat_images, drone_images)
                    else:
                        outputs = self.model(sat_images, drone_images)
                    
                    # Compute loss
                    losses = self.criterion(outputs, sat_labels)
                    total_loss = losses['total']
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
                    
                    # Update metrics
                    loss_meter.update(total_loss.item(), sat_images.size(0))
                    
                    # Collect predictions for accuracy calculation
                    if 'satellite' in outputs and outputs['satellite'] is not None:
                        sat_preds = outputs['satellite']['predictions']
                        if isinstance(sat_preds, list):
                            pred = torch.argmax(sat_preds[0], dim=1)
                        else:
                            pred = torch.argmax(sat_preds, dim=1)
                        
                        self.metrics_calculator.update(
                            pred.cpu().numpy(),
                            sat_labels.cpu().numpy()
                        )
                    
                    # Measure elapsed time
                    batch_time.update(time.time() - end_time)
                    end_time = time.time()
                    
                    # Log progress
                    if batch_idx % self.config.get('system', {}).get('log_interval', 10) == 0:
                        logger.info(
                            f'Epoch: [{self.current_epoch}][{batch_idx}/{len(dataloader)}] '
                            f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})'
                        )
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Compute epoch metrics
        epoch_metrics = self.metrics_calculator.compute_metrics()
        epoch_metrics['loss'] = loss_meter.avg
        epoch_metrics['batch_time'] = batch_time.avg
        
        return epoch_metrics
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            dataloader: Validation dataloader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        loss_meter = AverageMeter('Loss')
        self.metrics_calculator.reset()
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                try:
                    # Parse batch data
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                        (sat_images, sat_labels), (drone_images, drone_labels) = batch_data
                        
                        # Move to device
                        sat_images = sat_images.to(self.device)
                        drone_images = drone_images.to(self.device)
                        sat_labels = sat_labels.to(self.device)
                        
                        # Forward pass
                        if hasattr(self.model, 'module'):
                            outputs = self.model.module(sat_images, drone_images)
                        else:
                            outputs = self.model(sat_images, drone_images)
                        
                        # Compute loss
                        losses = self.criterion(outputs, sat_labels)
                        total_loss = losses['total']
                        
                        # Update metrics
                        loss_meter.update(total_loss.item(), sat_images.size(0))
                        
                        # Collect predictions
                        if 'satellite' in outputs and outputs['satellite'] is not None:
                            sat_preds = outputs['satellite']['predictions']
                            if isinstance(sat_preds, list):
                                pred = torch.argmax(sat_preds[0], dim=1)
                            else:
                                pred = torch.argmax(sat_preds, dim=1)
                            
                            self.metrics_calculator.update(
                                pred.cpu().numpy(),
                                sat_labels.cpu().numpy()
                            )
                
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        # Compute validation metrics
        val_metrics = self.metrics_calculator.compute_metrics()
        val_metrics['loss'] = loss_meter.avg
        
        return val_metrics
    
    def train(self, train_dataloader: DataLoader, 
              val_dataloader: Optional[DataLoader] = None,
              num_epochs: int = 100) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader (optional)
            num_epochs: Number of epochs to train
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_dataloader)
            
            # Validate epoch
            val_metrics = {}
            if val_dataloader is not None:
                val_metrics = self.validate_epoch(val_dataloader)
            
            # Update scheduler
            if self.scheduler is not None:
                if hasattr(self.scheduler, 'step'):
                    self.scheduler.step()
            
            # Log epoch results
            current_lr = self.optimizer.param_groups[0]['lr']
            
            epoch_summary = {
                'epoch': epoch + 1,
                'train_loss': train_metrics.get('loss', 0.0),
                'train_accuracy': train_metrics.get('accuracy', 0.0),
                'val_loss': val_metrics.get('loss', 0.0),
                'val_accuracy': val_metrics.get('accuracy', 0.0),
                'learning_rate': current_lr
            }
            
            self.training_history.append(epoch_summary)
            
            # Update visualizer
            if self.visualizer is not None:
                self.visualizer.update_metrics(epoch + 1, epoch_summary)
            
            # Log progress
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics.get('loss', 0.0):.4f}, "
                f"Train Acc: {train_metrics.get('accuracy', 0.0):.4f}, "
                f"Val Loss: {val_metrics.get('loss', 0.0):.4f}, "
                f"Val Acc: {val_metrics.get('accuracy', 0.0):.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # Save best model
            current_accuracy = val_metrics.get('accuracy', train_metrics.get('accuracy', 0.0))
            if current_accuracy > self.best_accuracy:
                self.best_accuracy = current_accuracy
                self.save_checkpoint(epoch + 1, is_best=True)
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.get('system', {}).get('save_interval', 10) == 0:
                self.save_checkpoint(epoch + 1, is_best=False)
        
        # Generate final plots
        if self.visualizer is not None:
            self.visualizer.plot_training_curves(save=True, show=False)
            self.visualizer.save_metrics_csv()
        
        logger.info("Training completed!")
        
        return {
            'training_history': self.training_history,
            'best_accuracy': self.best_accuracy
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint_dir = Path(self.config.get('system', {}).get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_accuracy': self.best_accuracy,
            'config': self.config,
            'training_history': self.training_history
        }
        
        if is_best:
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved best model checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint information
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        self.training_history = checkpoint.get('training_history', [])
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        
        return checkpoint
