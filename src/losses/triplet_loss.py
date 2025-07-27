"""Triplet loss implementations for ViT-CNN-crossview."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute euclidean distance between two tensors.
    
    Args:
        x: Tensor of shape (m, d)
        y: Tensor of shape (n, d)
        
    Returns:
        Distance matrix of shape (m, n)
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine distance between two tensors.
    
    Args:
        x: Tensor of shape (m, d)
        y: Tensor of shape (n, d)
        
    Returns:
        Distance matrix of shape (m, n)
    """
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    similarity = torch.mm(x_norm, y_norm.t())
    dist = 1 - similarity
    return dist


def hard_example_mining(dist_mat: torch.Tensor, labels: torch.Tensor, 
                       return_inds: bool = False):
    """
    Hard example mining for triplet loss.
    
    Args:
        dist_mat: Distance matrix of shape (batch_size, batch_size)
        labels: Labels of shape (batch_size,)
        return_inds: Whether to return indices
        
    Returns:
        Tuple of (dist_ap, dist_an) or (dist_ap, dist_an, p_inds, n_inds)
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    pos_mask_reshaped = dist_mat[is_pos].contiguous().view(N, -1)
    if pos_mask_reshaped.size(1) == 0:
        # No positive pairs found, return zeros
        dist_ap = torch.zeros(N, 1, device=dist_mat.device)
        relative_p_inds = torch.zeros(N, 1, dtype=torch.long, device=dist_mat.device)
    else:
        dist_ap, relative_p_inds = torch.max(pos_mask_reshaped, 1, keepdim=True)

    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    neg_mask_reshaped = dist_mat[is_neg].contiguous().view(N, -1)
    if neg_mask_reshaped.size(1) == 0:
        # No negative pairs found, return large distances
        dist_an = torch.full((N, 1), float('inf'), device=dist_mat.device)
        relative_n_inds = torch.zeros(N, 1, dtype=torch.long, device=dist_mat.device)
    else:
        dist_an, relative_n_inds = torch.min(neg_mask_reshaped, 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining."""
    
    def __init__(self, margin: float = 0.3, distance_metric: str = 'euclidean',
                 normalize_feature: bool = False, hard_factor: float = 0.0):
        """
        Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss
            distance_metric: Distance metric ('euclidean' or 'cosine')
            normalize_feature: Whether to normalize features
            hard_factor: Hard mining factor
        """
        super(TripletLoss, self).__init__()
        
        self.margin = margin
        self.distance_metric = distance_metric
        self.normalize_feature = normalize_feature
        self.hard_factor = hard_factor
        
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()
    
    def forward(self, global_feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of triplet loss.
        
        Args:
            global_feat: Feature tensor of shape (batch_size, feat_dim)
            labels: Label tensor of shape (batch_size,)
            
        Returns:
            Triplet loss value
        """
        # Safety check for batch size
        if global_feat.size(0) < 2:
            # Not enough samples for triplet loss, return zero loss
            return torch.tensor(0.0, device=global_feat.device, requires_grad=True)

        # Check if we have multiple classes
        unique_labels = torch.unique(labels)
        if len(unique_labels) < 2:
            # All samples are from the same class, return zero loss
            return torch.tensor(0.0, device=global_feat.device, requires_grad=True)

        if self.normalize_feature:
            global_feat = F.normalize(global_feat, p=2, dim=1)

        if self.distance_metric == 'euclidean':
            dist_mat = euclidean_dist(global_feat, global_feat)
        elif self.distance_metric == 'cosine':
            dist_mat = cosine_dist(global_feat, global_feat)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        
        # Apply hard factor
        if self.hard_factor > 0:
            dist_ap *= (1.0 + self.hard_factor)
            dist_an *= (1.0 - self.hard_factor)
        
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        
        return loss


class HardTripletLoss(nn.Module):
    """Hard triplet loss with batch hard mining."""
    
    def __init__(self, margin: float = 0.3, normalize_feature: bool = True):
        """
        Initialize hard triplet loss.
        
        Args:
            margin: Margin for triplet loss
            normalize_feature: Whether to normalize features
        """
        super(HardTripletLoss, self).__init__()
        
        self.margin = margin
        self.normalize_feature = normalize_feature
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of hard triplet loss.
        
        Args:
            embeddings: Feature embeddings of shape (batch_size, feat_dim)
            labels: Labels of shape (batch_size,)
            
        Returns:
            Hard triplet loss value
        """
        if self.normalize_feature:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise distance matrix
        dist_mat = euclidean_dist(embeddings, embeddings)
        
        # For each anchor, find the hardest positive and negative
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        
        # Compute triplet loss
        loss = F.relu(dist_ap - dist_an + self.margin)
        
        return loss.mean()


class BatchHardTripletLoss(nn.Module):
    """Batch hard triplet loss."""
    
    def __init__(self, margin: float = 0.3, normalize_feature: bool = True,
                 squared: bool = False):
        """
        Initialize batch hard triplet loss.
        
        Args:
            margin: Margin for triplet loss
            normalize_feature: Whether to normalize features
            squared: Whether to use squared euclidean distance
        """
        super(BatchHardTripletLoss, self).__init__()
        
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.squared = squared
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of batch hard triplet loss.
        
        Args:
            embeddings: Feature embeddings of shape (batch_size, feat_dim)
            labels: Labels of shape (batch_size,)
            
        Returns:
            Batch hard triplet loss value
        """
        if self.normalize_feature:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise distance matrix
        if self.squared:
            dist_mat = torch.pow(euclidean_dist(embeddings, embeddings), 2)
        else:
            dist_mat = euclidean_dist(embeddings, embeddings)
        
        # Get the hardest positive and negative for each anchor
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        
        # Compute triplet loss
        loss = F.relu(dist_ap - dist_an + self.margin)
        
        return loss.mean()


class SoftTripletLoss(nn.Module):
    """Soft triplet loss using log-sum-exp."""
    
    def __init__(self, margin: float = 0.3, normalize_feature: bool = True,
                 alpha: float = 1.0):
        """
        Initialize soft triplet loss.
        
        Args:
            margin: Margin for triplet loss
            normalize_feature: Whether to normalize features
            alpha: Temperature parameter for softmax
        """
        super(SoftTripletLoss, self).__init__()
        
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.alpha = alpha
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of soft triplet loss.
        
        Args:
            embeddings: Feature embeddings of shape (batch_size, feat_dim)
            labels: Labels of shape (batch_size,)
            
        Returns:
            Soft triplet loss value
        """
        if self.normalize_feature:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise distance matrix
        dist_mat = euclidean_dist(embeddings, embeddings)
        
        N = dist_mat.size(0)
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
        
        # Soft positive and negative distances
        dist_ap = torch.logsumexp(-self.alpha * dist_mat[is_pos].view(N, -1), dim=1) / (-self.alpha)
        dist_an = torch.logsumexp(-self.alpha * (-dist_mat[is_neg].view(N, -1)), dim=1) / self.alpha
        
        # Compute triplet loss
        loss = F.relu(dist_ap - dist_an + self.margin)
        
        return loss.mean()
