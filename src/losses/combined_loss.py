"""Combined loss functions for ViT-CNN-crossview."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Union
import logging

from .triplet_loss import TripletLoss

logger = logging.getLogger(__name__)


class CombinedLoss(nn.Module):
    """Combined loss for multi-task learning."""
    
    def __init__(self, num_classes: int, triplet_weight: float = 0.3,
                 kl_weight: float = 0.0, alignment_weight: float = 1.0,
                 use_kl_loss: bool = False, use_focal_loss: bool = False):
        """
        Initialize combined loss.
        
        Args:
            num_classes: Number of classes
            triplet_weight: Weight for triplet loss
            kl_weight: Weight for KL divergence loss
            alignment_weight: Weight for alignment loss
            use_kl_loss: Whether to use KL divergence loss
            use_focal_loss: Whether to use focal loss instead of cross-entropy
        """
        super(CombinedLoss, self).__init__()
        
        self.triplet_weight = triplet_weight
        self.kl_weight = kl_weight
        self.alignment_weight = alignment_weight
        self.use_kl_loss = use_kl_loss
        
        # Classification loss
        if use_focal_loss:
            from .focal_loss import FocalLoss
            self.classification_loss = FocalLoss(num_classes=num_classes)
        else:
            self.classification_loss = nn.CrossEntropyLoss()
        
        # Triplet loss
        if triplet_weight > 0:
            self.triplet_loss = TripletLoss(margin=triplet_weight)
        else:
            self.triplet_loss = None
        
        # KL divergence loss for mutual learning
        if use_kl_loss:
            self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        else:
            self.kl_loss = None
        
        # Alignment loss
        self.alignment_loss = AlignmentLoss()
    
    def forward(self, outputs: Dict[str, any], labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            outputs: Model outputs dictionary
            labels: Ground truth labels
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = 0.0
        
        # Extract predictions and features
        if 'satellite' in outputs and outputs['satellite'] is not None:
            sat_predictions = outputs['satellite']['predictions']
            sat_features = outputs['satellite']['features']
        else:
            sat_predictions = None
            sat_features = None
        
        if 'drone' in outputs and outputs['drone'] is not None:
            drone_predictions = outputs['drone']['predictions']
            drone_features = outputs['drone']['features']
        else:
            drone_predictions = None
            drone_features = None
        
        # Classification loss
        cls_loss = 0.0
        if sat_predictions is not None:
            if isinstance(sat_predictions, list):
                for pred in sat_predictions:
                    # Handle case where pred might still be a list [tensor, features]
                    if isinstance(pred, list):
                        pred_tensor = pred[0]  # Take the prediction tensor
                    else:
                        pred_tensor = pred

                    # Additional safety check for tensor dimensions
                    if isinstance(pred_tensor, torch.Tensor):
                        if pred_tensor.ndim == 1:
                            # Skip 1D tensors with warning
                            logger.warning(f"Skipping 1D prediction tensor with shape {pred_tensor.shape}")
                            continue
                        elif pred_tensor.ndim != 2:
                            logger.warning(f"Unexpected prediction tensor dimensions: {pred_tensor.shape}")
                            continue

                    cls_loss += self.classification_loss(pred_tensor, labels)
            else:
                # Handle case where sat_predictions might be [tensor, features]
                if isinstance(sat_predictions, list) and len(sat_predictions) == 2:
                    pred_tensor = sat_predictions[0]
                else:
                    pred_tensor = sat_predictions
                cls_loss += self.classification_loss(pred_tensor, labels)

        if drone_predictions is not None:
            if isinstance(drone_predictions, list):
                for pred in drone_predictions:
                    # Handle case where pred might still be a list [tensor, features]
                    if isinstance(pred, list):
                        pred_tensor = pred[0]  # Take the prediction tensor
                    else:
                        pred_tensor = pred

                    # Additional safety check for tensor dimensions
                    if isinstance(pred_tensor, torch.Tensor):
                        if pred_tensor.ndim == 1:
                            # Skip 1D tensors with warning
                            logger.warning(f"Skipping 1D drone prediction tensor with shape {pred_tensor.shape}")
                            continue
                        elif pred_tensor.ndim != 2:
                            logger.warning(f"Unexpected drone prediction tensor dimensions: {pred_tensor.shape}")
                            continue

                    cls_loss += self.classification_loss(pred_tensor, labels)
            else:
                # Handle case where drone_predictions might be [tensor, features]
                if isinstance(drone_predictions, list) and len(drone_predictions) == 2:
                    pred_tensor = drone_predictions[0]
                else:
                    pred_tensor = drone_predictions
                cls_loss += self.classification_loss(pred_tensor, labels)
        
        losses['classification'] = cls_loss
        total_loss += cls_loss
        
        # Triplet loss
        if self.triplet_loss is not None and self.triplet_weight > 0:
            triplet_loss = 0.0
            
            if sat_features is not None:
                if isinstance(sat_features, list):
                    for feat in sat_features:
                        triplet_loss += self.triplet_loss(feat, labels)
                else:
                    triplet_loss += self.triplet_loss(sat_features, labels)
            
            if drone_features is not None:
                if isinstance(drone_features, list):
                    for feat in drone_features:
                        triplet_loss += self.triplet_loss(feat, labels)
                else:
                    triplet_loss += self.triplet_loss(drone_features, labels)
            
            losses['triplet'] = triplet_loss
            total_loss += self.triplet_weight * triplet_loss
        else:
            losses['triplet'] = torch.tensor(0.0, device=labels.device)
        
        # KL divergence loss
        if self.kl_loss is not None and self.use_kl_loss and self.kl_weight > 0:
            if sat_predictions is not None and drone_predictions is not None:
                if isinstance(sat_predictions, list) and isinstance(drone_predictions, list):
                    kl_loss = 0.0
                    for sat_pred, drone_pred in zip(sat_predictions, drone_predictions):
                        kl_loss += self.kl_loss(
                            F.log_softmax(sat_pred, dim=1),
                            F.softmax(drone_pred, dim=1)
                        )
                        kl_loss += self.kl_loss(
                            F.log_softmax(drone_pred, dim=1),
                            F.softmax(sat_pred, dim=1)
                        )
                else:
                    kl_loss = self.kl_loss(
                        F.log_softmax(sat_predictions, dim=1),
                        F.softmax(drone_predictions, dim=1)
                    )
                    kl_loss += self.kl_loss(
                        F.log_softmax(drone_predictions, dim=1),
                        F.softmax(sat_predictions, dim=1)
                    )
                
                losses['kl_divergence'] = kl_loss
                total_loss += self.kl_weight * kl_loss
            else:
                losses['kl_divergence'] = torch.tensor(0.0, device=labels.device)
        else:
            losses['kl_divergence'] = torch.tensor(0.0, device=labels.device)
        
        # Alignment loss
        if 'alignment' in outputs and outputs['alignment'] is not None:
            alignment_loss = self.alignment_loss(outputs['alignment'])
            losses['alignment'] = alignment_loss
            total_loss += self.alignment_weight * alignment_loss
        else:
            losses['alignment'] = torch.tensor(0.0, device=labels.device)
        
        losses['total'] = total_loss
        
        return losses


class AlignmentLoss(nn.Module):
    """Cross-view alignment loss."""
    
    def __init__(self, loss_type: str = 'mse'):
        """
        Initialize alignment loss.
        
        Args:
            loss_type: Type of loss ('mse', 'cosine', 'kl')
        """
        super(AlignmentLoss, self).__init__()
        
        self.loss_type = loss_type
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'cosine':
            self.loss_fn = nn.CosineEmbeddingLoss()
        elif loss_type == 'kl':
            self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, alignment_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute alignment loss.
        
        Args:
            alignment_info: Dictionary containing alignment information
            
        Returns:
            Alignment loss value
        """
        sat_aligned = alignment_info['satellite_aligned']
        drone_aligned = alignment_info['drone_aligned']
        sat_original = alignment_info['satellite_original']
        drone_original = alignment_info['drone_original']
        
        if self.loss_type == 'mse':
            # MSE between aligned features and target features
            loss = self.loss_fn(sat_aligned, drone_original) + \
                   self.loss_fn(drone_aligned, sat_original)
        
        elif self.loss_type == 'cosine':
            # Cosine embedding loss
            batch_size = sat_aligned.size(0)
            target = torch.ones(batch_size, device=sat_aligned.device)
            
            loss = self.loss_fn(sat_aligned, drone_original, target) + \
                   self.loss_fn(drone_aligned, sat_original, target)
        
        elif self.loss_type == 'kl':
            # KL divergence loss
            sat_aligned_prob = F.softmax(sat_aligned, dim=1)
            drone_original_prob = F.softmax(drone_original, dim=1)
            drone_aligned_prob = F.softmax(drone_aligned, dim=1)
            sat_original_prob = F.softmax(sat_original, dim=1)
            
            loss = self.loss_fn(
                F.log_softmax(sat_aligned, dim=1), drone_original_prob
            ) + self.loss_fn(
                F.log_softmax(drone_aligned, dim=1), sat_original_prob
            )
        
        return loss


class MultiScaleLoss(nn.Module):
    """Multi-scale loss for hierarchical features."""
    
    def __init__(self, scale_weights: List[float], loss_type: str = 'cross_entropy'):
        """
        Initialize multi-scale loss.
        
        Args:
            scale_weights: Weights for different scales
            loss_type: Type of loss function
        """
        super(MultiScaleLoss, self).__init__()
        
        self.scale_weights = scale_weights
        
        if loss_type == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_type == 'focal':
            from .focal_loss import FocalLoss
            self.loss_fn = FocalLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, predictions: List[torch.Tensor], 
                labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute multi-scale loss.
        
        Args:
            predictions: List of predictions at different scales
            labels: Ground truth labels
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = 0.0
        
        for i, (pred, weight) in enumerate(zip(predictions, self.scale_weights)):
            scale_loss = self.loss_fn(pred, labels)
            losses[f'scale_{i}'] = scale_loss
            total_loss += weight * scale_loss
        
        losses['total'] = total_loss
        
        return losses


class ContrastiveLoss(nn.Module):
    """Contrastive loss for cross-view learning."""
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for contrastive loss
            temperature: Temperature for softmax
        """
        super(ContrastiveLoss, self).__init__()
        
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, features1: torch.Tensor, features2: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            features1: Features from first view
            features2: Features from second view
            labels: Labels indicating positive/negative pairs
            
        Returns:
            Contrastive loss value
        """
        # Normalize features
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.sum(features1 * features2, dim=1)
        
        # Compute contrastive loss
        positive_loss = (1 - similarity) ** 2
        negative_loss = torch.clamp(similarity - self.margin, min=0) ** 2
        
        # Create positive/negative mask
        positive_mask = (labels == 1).float()
        negative_mask = (labels == 0).float()
        
        loss = positive_mask * positive_loss + negative_mask * negative_loss
        
        return loss.mean()
