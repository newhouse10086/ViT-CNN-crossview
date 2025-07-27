"""Focal loss implementation for ViT-CNN-crossview."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0,
                 num_classes: Optional[int] = None, size_average: bool = True):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (default: None)
            gamma: Focusing parameter (default: 2.0)
            num_classes: Number of classes
            size_average: Whether to average the loss
        """
        super(FocalLoss, self).__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.ones(num_classes) * alpha
            elif isinstance(alpha, list):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of focal loss.
        
        Args:
            inputs: Predictions of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = at * focal_loss
        
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class BinaryFocalLoss(nn.Module):
    """Binary Focal Loss for binary classification."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, 
                 size_average: bool = True):
        """
        Initialize Binary Focal Loss.
        
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter
            size_average: Whether to average the loss
        """
        super(BinaryFocalLoss, self).__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of binary focal loss.
        
        Args:
            inputs: Predictions of shape (batch_size,)
            targets: Ground truth labels of shape (batch_size,)
            
        Returns:
            Binary focal loss value
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class AdaptiveFocalLoss(nn.Module):
    """Adaptive Focal Loss with learnable parameters."""
    
    def __init__(self, num_classes: int, initial_gamma: float = 2.0,
                 learn_alpha: bool = True, learn_gamma: bool = True):
        """
        Initialize Adaptive Focal Loss.
        
        Args:
            num_classes: Number of classes
            initial_gamma: Initial value for gamma
            learn_alpha: Whether to learn alpha parameters
            learn_gamma: Whether to learn gamma parameter
        """
        super(AdaptiveFocalLoss, self).__init__()
        
        self.num_classes = num_classes
        
        if learn_alpha:
            self.alpha = nn.Parameter(torch.ones(num_classes))
        else:
            self.register_buffer('alpha', torch.ones(num_classes))
        
        if learn_gamma:
            self.gamma = nn.Parameter(torch.tensor(initial_gamma))
        else:
            self.register_buffer('gamma', torch.tensor(initial_gamma))
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of adaptive focal loss.
        
        Args:
            inputs: Predictions of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
            
        Returns:
            Adaptive focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Get alpha for each sample
        alpha_t = self.alpha.gather(0, targets)
        
        # Compute focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class ClassBalancedFocalLoss(nn.Module):
    """Class-balanced Focal Loss using effective number of samples."""
    
    def __init__(self, num_classes: int, samples_per_class: torch.Tensor,
                 beta: float = 0.9999, gamma: float = 2.0):
        """
        Initialize Class-balanced Focal Loss.
        
        Args:
            num_classes: Number of classes
            samples_per_class: Number of samples per class
            beta: Hyperparameter for re-weighting
            gamma: Focusing parameter
        """
        super(ClassBalancedFocalLoss, self).__init__()
        
        self.num_classes = num_classes
        self.gamma = gamma
        
        # Compute effective number of samples
        effective_num = 1.0 - torch.pow(beta, samples_per_class)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes
        
        self.register_buffer('weights', weights)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of class-balanced focal loss.
        
        Args:
            inputs: Predictions of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
            
        Returns:
            Class-balanced focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Get weights for each sample
        weights_t = self.weights.gather(0, targets)
        
        # Compute focal loss
        focal_loss = weights_t * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class MultiClassFocalLoss(nn.Module):
    """Multi-class Focal Loss with different gamma for each class."""
    
    def __init__(self, num_classes: int, alpha: Optional[torch.Tensor] = None,
                 gamma: Optional[torch.Tensor] = None):
        """
        Initialize Multi-class Focal Loss.
        
        Args:
            num_classes: Number of classes
            alpha: Class weights
            gamma: Focusing parameters for each class
        """
        super(MultiClassFocalLoss, self).__init__()
        
        self.num_classes = num_classes
        
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        else:
            self.alpha = alpha
        
        if gamma is None:
            self.gamma = torch.ones(num_classes) * 2.0
        else:
            self.gamma = gamma
        
        self.register_buffer('alpha_buffer', self.alpha)
        self.register_buffer('gamma_buffer', self.gamma)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-class focal loss.
        
        Args:
            inputs: Predictions of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
            
        Returns:
            Multi-class focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Get alpha and gamma for each sample
        alpha_t = self.alpha_buffer.gather(0, targets)
        gamma_t = self.gamma_buffer.gather(0, targets)
        
        # Compute focal loss
        focal_loss = alpha_t * (1 - pt) ** gamma_t * ce_loss
        
        return focal_loss.mean()
