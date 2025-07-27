"""
Two-View FSRA Improved Model
Implements cross-view geo-localization with community clustering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .fsra_improved import FSRAImprovedModel


class CrossViewAlignment(nn.Module):
    """Cross-view feature alignment module."""
    
    def __init__(self, feature_dim: int = 256):
        super(CrossViewAlignment, self).__init__()
        self.feature_dim = feature_dim
        
        # Alignment networks
        self.satellite_align = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.drone_align = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, sat_features: torch.Tensor, drone_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align features between satellite and drone views.
        
        Args:
            sat_features: Satellite features (B, feature_dim)
            drone_features: Drone features (B, feature_dim)
            
        Returns:
            Tuple of aligned features
        """
        sat_aligned = self.satellite_align(sat_features)
        drone_aligned = self.drone_align(drone_features)
        
        # L2 normalization for better alignment
        sat_aligned = F.normalize(sat_aligned, p=2, dim=1)
        drone_aligned = F.normalize(drone_aligned, p=2, dim=1)
        
        return sat_aligned, drone_aligned


class TwoViewFSRAImproved(nn.Module):
    """
    Two-view FSRA Improved model for cross-view geo-localization.
    """
    
    def __init__(self, num_classes: int, num_clusters: int = 3,
                 use_pretrained: bool = True, feature_dim: int = 512,
                 share_weights: bool = True):
        super(TwoViewFSRAImproved, self).__init__()
        
        self.num_classes = num_classes
        self.num_clusters = num_clusters
        self.share_weights = share_weights
        self.feature_dim = feature_dim
        
        # Create models for both views
        self.satellite_model = FSRAImprovedModel(
            num_classes=num_classes,
            num_clusters=num_clusters,
            use_pretrained=use_pretrained,
            feature_dim=feature_dim
        )
        
        if share_weights:
            self.drone_model = self.satellite_model
        else:
            self.drone_model = FSRAImprovedModel(
                num_classes=num_classes,
                num_clusters=num_clusters,
                use_pretrained=use_pretrained,
                feature_dim=feature_dim
            )
        
        # Cross-view alignment
        self.cross_view_alignment = CrossViewAlignment(feature_dim=feature_dim)
        
    def forward(self, satellite_img: Optional[torch.Tensor], 
                drone_img: Optional[torch.Tensor]) -> Dict[str, any]:
        """
        Forward pass for two views.
        
        Args:
            satellite_img: Satellite image tensor (B, 3, H, W)
            drone_img: Drone image tensor (B, 3, H, W)
            
        Returns:
            Dictionary containing predictions and features for both views
        """
        outputs = {}
        
        # Process satellite view
        if satellite_img is not None:
            sat_predictions, sat_features = self.satellite_model(satellite_img)
            outputs['satellite'] = {
                'predictions': sat_predictions,
                'features': sat_features
            }
        else:
            outputs['satellite'] = None
        
        # Process drone view
        if drone_img is not None:
            drone_predictions, drone_features = self.drone_model(drone_img)
            outputs['drone'] = {
                'predictions': drone_predictions,
                'features': drone_features
            }
        else:
            outputs['drone'] = None
        
        # Cross-view alignment if both views are available
        if satellite_img is not None and drone_img is not None:
            # Use global features for alignment (first feature in the list)
            sat_global_feat = sat_features[0]
            drone_global_feat = drone_features[0]
            
            # Perform cross-view alignment
            sat_aligned, drone_aligned = self.cross_view_alignment(
                sat_global_feat, drone_global_feat
            )
            
            outputs['alignment'] = {
                'satellite_aligned': sat_aligned,
                'drone_aligned': drone_aligned,
                'satellite_original': sat_global_feat,
                'drone_original': drone_global_feat
            }
        else:
            outputs['alignment'] = None
        
        return outputs
    
    def extract_features(self, satellite_img: Optional[torch.Tensor], 
                        drone_img: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract features for retrieval/matching.
        
        Args:
            satellite_img: Satellite image tensor
            drone_img: Drone image tensor
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        with torch.no_grad():
            if satellite_img is not None:
                _, sat_features = self.satellite_model(satellite_img)
                features['satellite'] = sat_features[-1]  # Use final features
            
            if drone_img is not None:
                _, drone_features = self.drone_model(drone_img)
                features['drone'] = drone_features[-1]  # Use final features
        
        return features


def make_two_view_fsra_improved(num_classes: int, num_clusters: int = 3,
                              use_pretrained: bool = True, feature_dim: int = 512,
                              share_weights: bool = True) -> TwoViewFSRAImproved:
    """
    Create two-view FSRA Improved model.
    
    Args:
        num_classes: Number of classes
        num_clusters: Number of clusters for community detection
        use_pretrained: Whether to use pretrained ResNet
        feature_dim: Feature dimension
        share_weights: Whether to share weights between views
        
    Returns:
        Two-view FSRA Improved model
    """
    return TwoViewFSRAImproved(
        num_classes=num_classes,
        num_clusters=num_clusters,
        use_pretrained=use_pretrained,
        feature_dim=feature_dim,
        share_weights=share_weights
    )
