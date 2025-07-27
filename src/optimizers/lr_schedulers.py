"""Custom learning rate schedulers for ViT-CNN-crossview."""

import torch
from torch.optim import lr_scheduler
import math
from typing import List


class WarmupCosineScheduler(lr_scheduler._LRScheduler):
    """Cosine annealing scheduler with warmup."""
    
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, 
                 eta_min: float = 0, last_epoch: int = -1):
        """
        Initialize warmup cosine scheduler.
        
        Args:
            optimizer: Optimizer
            warmup_epochs: Number of warmup epochs
            max_epochs: Maximum number of epochs
            eta_min: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Get learning rates for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                   for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.eta_min + (base_lr - self.eta_min) * 
                   (1 + math.cos(math.pi * progress)) / 2 
                   for base_lr in self.base_lrs]


class WarmupLinearScheduler(lr_scheduler._LRScheduler):
    """Linear decay scheduler with warmup."""
    
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, 
                 eta_min: float = 0, last_epoch: int = -1):
        """
        Initialize warmup linear scheduler.
        
        Args:
            optimizer: Optimizer
            warmup_epochs: Number of warmup epochs
            max_epochs: Maximum number of epochs
            eta_min: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super(WarmupLinearScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Get learning rates for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                   for base_lr in self.base_lrs]
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.eta_min + (base_lr - self.eta_min) * (1 - progress) 
                   for base_lr in self.base_lrs]


class WarmupStepScheduler(lr_scheduler._LRScheduler):
    """Step scheduler with warmup."""
    
    def __init__(self, optimizer, warmup_epochs: int, milestones: List[int], 
                 gamma: float = 0.1, last_epoch: int = -1):
        """
        Initialize warmup step scheduler.
        
        Args:
            optimizer: Optimizer
            warmup_epochs: Number of warmup epochs
            milestones: List of epoch indices for learning rate decay
            gamma: Multiplicative factor of learning rate decay
            last_epoch: Last epoch index
        """
        self.warmup_epochs = warmup_epochs
        self.milestones = milestones
        self.gamma = gamma
        super(WarmupStepScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Get learning rates for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                   for base_lr in self.base_lrs]
        else:
            # Step decay
            decay_factor = self.gamma ** sum([self.last_epoch >= m for m in self.milestones])
            return [base_lr * decay_factor for base_lr in self.base_lrs]


class PolynomialScheduler(lr_scheduler._LRScheduler):
    """Polynomial learning rate scheduler."""
    
    def __init__(self, optimizer, max_epochs: int, power: float = 0.9, 
                 eta_min: float = 0, last_epoch: int = -1):
        """
        Initialize polynomial scheduler.
        
        Args:
            optimizer: Optimizer
            max_epochs: Maximum number of epochs
            power: Power of polynomial decay
            eta_min: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.max_epochs = max_epochs
        self.power = power
        self.eta_min = eta_min
        super(PolynomialScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Get learning rates for current epoch."""
        progress = self.last_epoch / self.max_epochs
        decay_factor = (1 - progress) ** self.power
        return [self.eta_min + (base_lr - self.eta_min) * decay_factor 
               for base_lr in self.base_lrs]


class CyclicCosineScheduler(lr_scheduler._LRScheduler):
    """Cyclic cosine annealing scheduler."""
    
    def __init__(self, optimizer, cycle_epochs: int, eta_min: float = 0, 
                 last_epoch: int = -1):
        """
        Initialize cyclic cosine scheduler.
        
        Args:
            optimizer: Optimizer
            cycle_epochs: Number of epochs per cycle
            eta_min: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.cycle_epochs = cycle_epochs
        self.eta_min = eta_min
        super(CyclicCosineScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Get learning rates for current epoch."""
        cycle_progress = (self.last_epoch % self.cycle_epochs) / self.cycle_epochs
        return [self.eta_min + (base_lr - self.eta_min) * 
               (1 + math.cos(math.pi * cycle_progress)) / 2 
               for base_lr in self.base_lrs]


class OneCycleScheduler(lr_scheduler._LRScheduler):
    """One cycle learning rate scheduler."""
    
    def __init__(self, optimizer, max_epochs: int, max_lr: float, 
                 pct_start: float = 0.3, div_factor: float = 25.0, 
                 final_div_factor: float = 10000.0, last_epoch: int = -1):
        """
        Initialize one cycle scheduler.
        
        Args:
            optimizer: Optimizer
            max_epochs: Maximum number of epochs
            max_lr: Maximum learning rate
            pct_start: Percentage of cycle spent increasing learning rate
            div_factor: Initial learning rate divisor
            final_div_factor: Final learning rate divisor
            last_epoch: Last epoch index
        """
        self.max_epochs = max_epochs
        self.max_lr = max_lr
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        super(OneCycleScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Get learning rates for current epoch."""
        progress = self.last_epoch / self.max_epochs
        
        if progress <= self.pct_start:
            # Increasing phase
            phase_progress = progress / self.pct_start
            lr = self.max_lr / self.div_factor + \
                 (self.max_lr - self.max_lr / self.div_factor) * phase_progress
        else:
            # Decreasing phase
            phase_progress = (progress - self.pct_start) / (1 - self.pct_start)
            lr = self.max_lr - (self.max_lr - self.max_lr / self.final_div_factor) * phase_progress
        
        return [lr for _ in self.base_lrs]


class ExponentialWarmupScheduler(lr_scheduler._LRScheduler):
    """Exponential decay scheduler with warmup."""
    
    def __init__(self, optimizer, warmup_epochs: int, gamma: float = 0.95, 
                 last_epoch: int = -1):
        """
        Initialize exponential warmup scheduler.
        
        Args:
            optimizer: Optimizer
            warmup_epochs: Number of warmup epochs
            gamma: Multiplicative factor of learning rate decay
            last_epoch: Last epoch index
        """
        self.warmup_epochs = warmup_epochs
        self.gamma = gamma
        super(ExponentialWarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Get learning rates for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                   for base_lr in self.base_lrs]
        else:
            # Exponential decay
            decay_epochs = self.last_epoch - self.warmup_epochs
            decay_factor = self.gamma ** decay_epochs
            return [base_lr * decay_factor for base_lr in self.base_lrs]


class AdaptiveScheduler(lr_scheduler._LRScheduler):
    """Adaptive learning rate scheduler based on loss."""
    
    def __init__(self, optimizer, patience: int = 10, factor: float = 0.5, 
                 threshold: float = 1e-4, min_lr: float = 1e-6, 
                 last_epoch: int = -1):
        """
        Initialize adaptive scheduler.
        
        Args:
            optimizer: Optimizer
            patience: Number of epochs to wait before reducing learning rate
            factor: Factor by which to reduce learning rate
            threshold: Threshold for measuring improvement
            min_lr: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.patience = patience
        self.factor = factor
        self.threshold = threshold
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.wait_count = 0
        super(AdaptiveScheduler, self).__init__(optimizer, last_epoch)
    
    def step(self, loss: float):
        """
        Step the scheduler with current loss.
        
        Args:
            loss: Current loss value
        """
        if loss < self.best_loss - self.threshold:
            self.best_loss = loss
            self.wait_count = 0
        else:
            self.wait_count += 1
            
        if self.wait_count >= self.patience:
            self._reduce_lr()
            self.wait_count = 0
        
        super().step()
    
    def _reduce_lr(self):
        """Reduce learning rate for all parameter groups."""
        for group in self.optimizer.param_groups:
            old_lr = group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            group['lr'] = new_lr
    
    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]
