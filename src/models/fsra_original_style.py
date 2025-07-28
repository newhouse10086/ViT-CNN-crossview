"""
FSRA Original Style Model - Closer to the original FSRA design.
Focus on Feature Segmentation and Region Alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple
import math


class FeatureSegmentationModule(nn.Module):
    """
    Feature Segmentation Module - Core of FSRA.
    Segments feature maps into meaningful regions.
    """
    
    def __init__(self, input_dim=512, num_regions=6, region_dim=256):
        super(FeatureSegmentationModule, self).__init__()
        self.num_regions = num_regions
        self.region_dim = region_dim
        
        # Region attention mechanism
        self.region_attention = nn.Sequential(
            nn.Conv2d(input_dim, num_regions, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # Region feature extraction
        self.region_conv = nn.Conv2d(input_dim, region_dim, kernel_size=1)
        
    def forward(self, features):
        """
        Args:
            features: (B, C, H, W) feature maps
        Returns:
            region_features: (B, num_regions, region_dim) segmented features
            attention_maps: (B, num_regions, H, W) attention maps
        """
        B, C, H, W = features.shape
        
        # Generate attention maps for each region
        attention_maps = self.region_attention(features)  # (B, num_regions, H, W)
        
        # Extract region features
        region_conv_features = self.region_conv(features)  # (B, region_dim, H, W)
        
        # Apply attention to get region-specific features
        region_features = []
        for i in range(self.num_regions):
            attention = attention_maps[:, i:i+1, :, :]  # (B, 1, H, W)
            region_feat = (region_conv_features * attention).sum(dim=(2, 3))  # (B, region_dim)
            region_features.append(region_feat)
        
        region_features = torch.stack(region_features, dim=1)  # (B, num_regions, region_dim)
        
        return region_features, attention_maps


class CrossViewAlignmentModule(nn.Module):
    """
    Cross-View Alignment Module.
    Aligns features between satellite and drone views.
    """
    
    def __init__(self, feature_dim=256, num_heads=8):
        super(CrossViewAlignmentModule, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Feature refinement
        self.refinement = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, sat_features, drone_features):
        """
        Args:
            sat_features: (B, num_regions, feature_dim) satellite features
            drone_features: (B, num_regions, feature_dim) drone features
        Returns:
            aligned_sat: (B, num_regions, feature_dim) aligned satellite features
            aligned_drone: (B, num_regions, feature_dim) aligned drone features
        """
        # Cross-attention: satellite queries drone
        aligned_sat, _ = self.cross_attention(
            query=sat_features,
            key=drone_features,
            value=drone_features
        )
        
        # Cross-attention: drone queries satellite
        aligned_drone, _ = self.cross_attention(
            query=drone_features,
            key=sat_features,
            value=sat_features
        )
        
        # Refinement
        aligned_sat = self.refinement(aligned_sat)
        aligned_drone = self.refinement(aligned_drone)
        
        return aligned_sat, aligned_drone


class FSRAOriginalStyle(nn.Module):
    """
    FSRA Original Style Model.
    Implements Feature Segmentation and Region Alignment.
    """
    
    def __init__(self, num_classes=701, num_regions=6, feature_dim=256):
        super(FSRAOriginalStyle, self).__init__()
        self.num_classes = num_classes
        self.num_regions = num_regions
        self.feature_dim = feature_dim
        
        # Backbone networks
        self.sat_backbone = self._create_backbone()
        self.drone_backbone = self._create_backbone()
        
        # Feature segmentation modules
        self.sat_segmentation = FeatureSegmentationModule(
            input_dim=512, num_regions=num_regions, region_dim=feature_dim
        )
        self.drone_segmentation = FeatureSegmentationModule(
            input_dim=512, num_regions=num_regions, region_dim=feature_dim
        )
        
        # Cross-view alignment
        self.cross_alignment = CrossViewAlignmentModule(
            feature_dim=feature_dim, num_heads=8
        )
        
        # Global feature aggregation
        self.global_aggregation = nn.Sequential(
            nn.Linear(feature_dim * num_regions, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Classifiers
        self.global_classifier = self._create_classifier(feature_dim, num_classes)
        self.region_classifiers = nn.ModuleList([
            self._create_classifier(feature_dim, num_classes)
            for _ in range(num_regions)
        ])
        
    def _create_backbone(self):
        """Create ResNet18 backbone."""
        resnet = models.resnet18(pretrained=True)
        # Remove the last two layers (avgpool and fc)
        backbone = nn.Sequential(*list(resnet.children())[:-2])
        return backbone
    
    def _create_classifier(self, input_dim, num_classes):
        """Create classifier head."""
        return nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, sat_img: torch.Tensor, drone_img: torch.Tensor) -> Dict:
        """
        Forward pass implementing FSRA methodology.
        
        Args:
            sat_img: (B, 3, 256, 256) satellite images
            drone_img: (B, 3, 256, 256) drone images
            
        Returns:
            Dictionary containing predictions and features
        """
        B = sat_img.shape[0]
        
        # 1. Feature Extraction
        sat_backbone_features = self.sat_backbone(sat_img)      # (B, 512, H, W)
        drone_backbone_features = self.drone_backbone(drone_img) # (B, 512, H, W)
        
        # 2. Feature Segmentation
        sat_regions, sat_attention = self.sat_segmentation(sat_backbone_features)
        drone_regions, drone_attention = self.drone_segmentation(drone_backbone_features)
        # sat_regions: (B, num_regions, feature_dim)
        # drone_regions: (B, num_regions, feature_dim)
        
        # 3. Cross-View Alignment
        aligned_sat, aligned_drone = self.cross_alignment(sat_regions, drone_regions)
        
        # 4. Global Feature Aggregation
        sat_global = self.global_aggregation(aligned_sat.view(B, -1))  # (B, feature_dim)
        drone_global = self.global_aggregation(aligned_drone.view(B, -1))  # (B, feature_dim)
        
        # 5. Classification
        # Global predictions
        sat_global_pred = self.global_classifier(sat_global)
        drone_global_pred = self.global_classifier(drone_global)
        
        # Region predictions
        sat_region_preds = []
        drone_region_preds = []
        
        for i in range(self.num_regions):
            sat_region_pred = self.region_classifiers[i](aligned_sat[:, i, :])
            drone_region_pred = self.region_classifiers[i](aligned_drone[:, i, :])
            sat_region_preds.append(sat_region_pred)
            drone_region_preds.append(drone_region_pred)
        
        # Prepare outputs
        outputs = {
            'satellite': {
                'predictions': [sat_global_pred] + sat_region_preds,
                'features': {
                    'global': sat_global,
                    'regions': aligned_sat,
                    'backbone': sat_backbone_features
                },
                'attention_maps': sat_attention
            },
            'drone': {
                'predictions': [drone_global_pred] + drone_region_preds,
                'features': {
                    'global': drone_global,
                    'regions': aligned_drone,
                    'backbone': drone_backbone_features
                },
                'attention_maps': drone_attention
            },
            'alignment': {
                'sat_aligned': aligned_sat,
                'drone_aligned': aligned_drone
            }
        }
        
        return outputs


def create_fsra_original_style(num_classes=701, num_regions=6, feature_dim=256):
    """
    Create FSRA Original Style model.
    
    Args:
        num_classes: Number of classes
        num_regions: Number of regions for segmentation
        feature_dim: Feature dimension
        
    Returns:
        FSRAOriginalStyle model
    """
    return FSRAOriginalStyle(
        num_classes=num_classes,
        num_regions=num_regions,
        feature_dim=feature_dim
    )
