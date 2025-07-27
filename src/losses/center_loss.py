"""Center loss implementation for ViT-CNN-crossview."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CenterLoss(nn.Module):
    """Center loss for feature learning."""
    
    def __init__(self, num_classes: int, feat_dim: int, use_gpu: bool = True):
        """
        Initialize Center Loss.
        
        Args:
            num_classes: Number of classes
            feat_dim: Feature dimension
            use_gpu: Whether to use GPU
        """
        super(CenterLoss, self).__init__()
        
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of center loss.
        
        Args:
            x: Feature tensor of shape (batch_size, feat_dim)
            labels: Label tensor of shape (batch_size,)
            
        Returns:
            Center loss value
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


class AdaptiveCenterLoss(nn.Module):
    """Adaptive Center Loss with learnable margin."""
    
    def __init__(self, num_classes: int, feat_dim: int, margin: float = 0.5,
                 use_gpu: bool = True, learn_margin: bool = True):
        """
        Initialize Adaptive Center Loss.
        
        Args:
            num_classes: Number of classes
            feat_dim: Feature dimension
            margin: Margin for center loss
            use_gpu: Whether to use GPU
            learn_margin: Whether to learn margin parameter
        """
        super(AdaptiveCenterLoss, self).__init__()
        
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        
        if learn_margin:
            self.margin = nn.Parameter(torch.tensor(margin))
        else:
            self.register_buffer('margin', torch.tensor(margin))
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of adaptive center loss.
        
        Args:
            x: Feature tensor of shape (batch_size, feat_dim)
            labels: Label tensor of shape (batch_size,)
            
        Returns:
            Adaptive center loss value
        """
        batch_size = x.size(0)
        
        # Compute distances to centers
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        # Apply margin
        dist = torch.clamp(distmat - self.margin, min=0) * mask.float()
        loss = dist.sum() / batch_size

        return loss


class MultiCenterLoss(nn.Module):
    """Multi-center loss with multiple centers per class."""
    
    def __init__(self, num_classes: int, feat_dim: int, num_centers_per_class: int = 2,
                 use_gpu: bool = True):
        """
        Initialize Multi-center Loss.
        
        Args:
            num_classes: Number of classes
            feat_dim: Feature dimension
            num_centers_per_class: Number of centers per class
            use_gpu: Whether to use GPU
        """
        super(MultiCenterLoss, self).__init__()
        
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.num_centers_per_class = num_centers_per_class
        self.use_gpu = use_gpu
        
        total_centers = num_classes * num_centers_per_class
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(total_centers, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(total_centers, self.feat_dim))
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-center loss.
        
        Args:
            x: Feature tensor of shape (batch_size, feat_dim)
            labels: Label tensor of shape (batch_size,)
            
        Returns:
            Multi-center loss value
        """
        batch_size = x.size(0)
        total_loss = 0.0
        
        for i in range(batch_size):
            label = labels[i].item()
            feature = x[i].unsqueeze(0)  # (1, feat_dim)
            
            # Get centers for this class
            start_idx = label * self.num_centers_per_class
            end_idx = start_idx + self.num_centers_per_class
            class_centers = self.centers[start_idx:end_idx]  # (num_centers_per_class, feat_dim)
            
            # Compute distances to all centers of this class
            distances = torch.pow(feature - class_centers, 2).sum(dim=1)
            
            # Use minimum distance (closest center)
            min_distance = torch.min(distances)
            total_loss += min_distance
        
        return total_loss / batch_size


class WeightedCenterLoss(nn.Module):
    """Weighted Center Loss with class-specific weights."""
    
    def __init__(self, num_classes: int, feat_dim: int, 
                 class_weights: Optional[torch.Tensor] = None, use_gpu: bool = True):
        """
        Initialize Weighted Center Loss.
        
        Args:
            num_classes: Number of classes
            feat_dim: Feature dimension
            class_weights: Weights for each class
            use_gpu: Whether to use GPU
        """
        super(WeightedCenterLoss, self).__init__()
        
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        
        if class_weights is None:
            class_weights = torch.ones(num_classes)
        
        if self.use_gpu:
            self.register_buffer('class_weights', class_weights.cuda())
        else:
            self.register_buffer('class_weights', class_weights)
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of weighted center loss.
        
        Args:
            x: Feature tensor of shape (batch_size, feat_dim)
            labels: Label tensor of shape (batch_size,)
            
        Returns:
            Weighted center loss value
        """
        batch_size = x.size(0)
        
        # Compute distances to centers
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        # Apply class weights
        weights = self.class_weights.unsqueeze(0).expand(batch_size, self.num_classes)
        dist = distmat * mask.float() * weights
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


class NormalizedCenterLoss(nn.Module):
    """Normalized Center Loss with L2 normalization."""
    
    def __init__(self, num_classes: int, feat_dim: int, use_gpu: bool = True):
        """
        Initialize Normalized Center Loss.
        
        Args:
            num_classes: Number of classes
            feat_dim: Feature dimension
            use_gpu: Whether to use GPU
        """
        super(NormalizedCenterLoss, self).__init__()
        
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of normalized center loss.
        
        Args:
            x: Feature tensor of shape (batch_size, feat_dim)
            labels: Label tensor of shape (batch_size,)
            
        Returns:
            Normalized center loss value
        """
        # Normalize features and centers
        x_norm = F.normalize(x, p=2, dim=1)
        centers_norm = F.normalize(self.centers, p=2, dim=1)
        
        batch_size = x_norm.size(0)
        
        # Compute cosine distances
        cosine_sim = torch.mm(x_norm, centers_norm.t())
        cosine_dist = 1 - cosine_sim
        
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = cosine_dist * mask.float()
        loss = dist.sum() / batch_size

        return loss
