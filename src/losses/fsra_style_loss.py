"""
FSRA-style loss function for cross-view geo-localization.
Based on standard practices in cross-view geo-localization literature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional


class TripletLossWithBatchHard(nn.Module):
    """
    Triplet loss with batch hard mining.
    This is the standard approach used in cross-view geo-localization.
    """
    
    def __init__(self, margin=0.3):
        super(TripletLossWithBatchHard, self).__init__()
        self.margin = margin
        
    def forward(self, features, labels):
        """
        Args:
            features: (N, D) feature vectors
            labels: (N,) class labels
        """
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute pairwise distances
        dist_mat = torch.cdist(features, features, p=2)
        
        # For each anchor, find hardest positive and hardest negative
        batch_size = features.size(0)
        
        # Create masks for positive and negative pairs
        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.T).float()
        mask_neg = (labels != labels.T).float()
        
        # Remove self-comparison
        mask_pos = mask_pos - torch.eye(batch_size, device=features.device)
        
        # Find hardest positive (maximum distance among positives)
        dist_pos = dist_mat * mask_pos
        dist_pos[mask_pos == 0] = -np.inf
        hardest_pos_dist = torch.max(dist_pos, dim=1)[0]
        
        # Find hardest negative (minimum distance among negatives)
        dist_neg = dist_mat * mask_neg
        dist_neg[mask_neg == 0] = np.inf
        hardest_neg_dist = torch.min(dist_neg, dim=1)[0]
        
        # Compute triplet loss
        loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
        
        return loss.mean()


class CenterLoss(nn.Module):
    """
    Center loss for learning discriminative features.
    """
    
    def __init__(self, num_classes, feature_dim, alpha=0.5):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha
        
        # Initialize centers
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
        
    def forward(self, features, labels):
        """
        Args:
            features: (N, D) feature vectors
            labels: (N,) class labels
        """
        batch_size = features.size(0)

        # Ensure centers are on the same device as features
        if self.centers.device != features.device:
            self.centers.data = self.centers.data.to(features.device)

        # Get centers for current batch
        centers_batch = self.centers[labels]  # (N, D)

        # Compute center loss
        loss = F.mse_loss(features, centers_batch)

        return loss


class FSRAStyleLoss(nn.Module):
    """
    FSRA-style combined loss function for cross-view geo-localization.
    """
    
    def __init__(self, 
                 num_classes,
                 classification_weight=1.0,
                 triplet_weight=1.0,
                 center_weight=0.0005,
                 triplet_margin=0.3):
        super(FSRAStyleLoss, self).__init__()
        
        self.num_classes = num_classes
        self.classification_weight = classification_weight
        self.triplet_weight = triplet_weight
        self.center_weight = center_weight
        
        # Classification loss (Cross Entropy)
        self.classification_loss = nn.CrossEntropyLoss()
        
        # Triplet loss with batch hard mining
        self.triplet_loss = TripletLossWithBatchHard(margin=triplet_margin)
        
        # Center loss
        if center_weight > 0:
            self.center_loss = CenterLoss(num_classes, feature_dim=256)  # Assume 256-dim features
        else:
            self.center_loss = None
            
    def forward(self, outputs: Dict, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute FSRA-style combined loss.
        
        Args:
            outputs: Model outputs dictionary
            labels: Ground truth labels (N,)
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = 0.0
        
        # Extract predictions and features
        if 'satellite' in outputs:
            predictions = outputs['satellite']['predictions']  # List of prediction tensors
            features_dict = outputs['satellite']['features']   # Dict of feature tensors
        else:
            raise ValueError("No satellite outputs found")
        
        # 1. Classification Loss
        # Use all predictions for classification loss
        classification_loss = 0.0
        num_predictions = len(predictions)
        
        for i, pred in enumerate(predictions):
            if isinstance(pred, torch.Tensor) and pred.dim() == 2:
                cls_loss = self.classification_loss(pred, labels)
                classification_loss += cls_loss
                losses[f'classification_loss_{i}'] = cls_loss.item()
        
        classification_loss = classification_loss / num_predictions
        total_loss += self.classification_weight * classification_loss
        losses['classification_loss'] = classification_loss.item()
        
        # 2. Triplet Loss
        # Use global features for triplet loss
        if 'global' in features_dict:
            global_features = features_dict['global']
        elif 'final' in features_dict:
            global_features = features_dict['final']
        else:
            # Use the first available feature
            global_features = list(features_dict.values())[0]
        
        if isinstance(global_features, torch.Tensor) and global_features.dim() == 2:
            triplet_loss = self.triplet_loss(global_features, labels)
            total_loss += self.triplet_weight * triplet_loss
            losses['triplet_loss'] = triplet_loss.item()
        else:
            losses['triplet_loss'] = 0.0
        
        # 3. Center Loss (optional)
        if self.center_loss is not None and isinstance(global_features, torch.Tensor):
            center_loss = self.center_loss(global_features, labels)
            total_loss += self.center_weight * center_loss
            losses['center_loss'] = center_loss.item()
        else:
            losses['center_loss'] = 0.0
        
        losses['total'] = total_loss
        
        return losses


def create_fsra_style_loss(num_classes, 
                          classification_weight=1.0,
                          triplet_weight=1.0, 
                          center_weight=0.0005,
                          triplet_margin=0.3):
    """
    Create FSRA-style loss function.
    
    Args:
        num_classes: Number of classes
        classification_weight: Weight for classification loss
        triplet_weight: Weight for triplet loss
        center_weight: Weight for center loss
        triplet_margin: Margin for triplet loss
        
    Returns:
        FSRAStyleLoss instance
    """
    return FSRAStyleLoss(
        num_classes=num_classes,
        classification_weight=classification_weight,
        triplet_weight=triplet_weight,
        center_weight=center_weight,
        triplet_margin=triplet_margin
    )
