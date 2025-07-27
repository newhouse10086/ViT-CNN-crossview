"""
Simple FSRA Implementation
Based on the original FSRA paper: "A Transformer-Based Feature Segmentation and Region Alignment Method For UAV-View Geo-Localization"
Simplified to fix the 1D tensor issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional
import numpy as np

from .backbones.resnet import resnet18_backbone
from .components import weights_init_kaiming


class SimpleFSRAModel(nn.Module):
    """
    Simplified FSRA model that follows the original paper structure.
    """
    
    def __init__(self, num_classes: int, num_regions: int = 4, 
                 use_pretrained: bool = True):
        super(SimpleFSRAModel, self).__init__()
        
        self.num_classes = num_classes
        self.num_regions = num_regions
        
        # ResNet backbone (following FSRA paper)
        self.backbone = resnet18_backbone(pretrained=use_pretrained)
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256)
            backbone_output = self.backbone(dummy_input)
            self.backbone_dim = backbone_output.shape[1]
            self.feature_map_size = backbone_output.shape[2]  # Assuming square feature map
        
        print(f"Backbone output: {backbone_output.shape}")
        print(f"Backbone dim: {self.backbone_dim}, Feature map size: {self.feature_map_size}")
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Global classifier
        self.global_classifier = nn.Sequential(
            nn.Linear(self.backbone_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Regional classifiers (following FSRA paper)
        # Divide feature map into regions (e.g., 2x2 = 4 regions)
        self.region_pool = nn.AdaptiveAvgPool2d((2, 2))  # Creates 2x2 regions
        
        self.regional_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.backbone_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            ) for _ in range(4)  # 2x2 = 4 regions
        ])
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.backbone_dim * 5, 512),  # global + 4 regional features
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Final classifier
        self.final_classifier = nn.Linear(512, num_classes)
        
        # Initialize weights
        self.apply(weights_init_kaiming)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass following FSRA structure.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Tuple of (predictions, features)
        """
        batch_size = x.size(0)
        
        # Extract backbone features
        feature_map = self.backbone(x)  # (B, C, H', W')
        
        # Global features
        global_features = self.global_pool(feature_map).view(batch_size, -1)  # (B, C)
        global_pred = self.global_classifier(global_features)  # (B, num_classes)
        
        # Regional features
        regional_feature_map = self.region_pool(feature_map)  # (B, C, 2, 2)
        
        regional_preds = []
        regional_features = []
        
        # Process each region
        for i in range(2):
            for j in range(2):
                region_idx = i * 2 + j
                region_features = regional_feature_map[:, :, i, j]  # (B, C)
                region_pred = self.regional_classifiers[region_idx](region_features)  # (B, num_classes)
                
                regional_preds.append(region_pred)
                regional_features.append(region_features)
        
        # Feature fusion
        all_features = torch.cat([global_features] + regional_features, dim=1)  # (B, C*5)
        fused_features = self.feature_fusion(all_features)  # (B, 512)
        final_pred = self.final_classifier(fused_features)  # (B, num_classes)
        
        # Collect predictions and features
        predictions = [global_pred] + regional_preds + [final_pred]
        features = [global_features] + regional_features + [fused_features]
        
        return predictions, features


class TwoViewSimpleFSRA(nn.Module):
    """
    Two-view version of Simple FSRA model.
    """
    
    def __init__(self, num_classes: int, num_regions: int = 4,
                 use_pretrained: bool = True, share_weights: bool = True):
        super(TwoViewSimpleFSRA, self).__init__()
        
        self.num_classes = num_classes
        self.share_weights = share_weights
        
        # Create models for both views
        self.satellite_model = SimpleFSRAModel(
            num_classes=num_classes,
            num_regions=num_regions,
            use_pretrained=use_pretrained
        )
        
        if share_weights:
            self.drone_model = self.satellite_model
        else:
            self.drone_model = SimpleFSRAModel(
                num_classes=num_classes,
                num_regions=num_regions,
                use_pretrained=use_pretrained
            )
    
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
        
        # No alignment for now (keep it simple)
        outputs['alignment'] = None
        
        return outputs


def make_simple_fsra_model(num_classes: int, num_regions: int = 4,
                          use_pretrained: bool = True, views: int = 2,
                          share_weights: bool = True) -> nn.Module:
    """
    Create Simple FSRA model.
    
    Args:
        num_classes: Number of classes
        num_regions: Number of regions (not used in simple version)
        use_pretrained: Whether to use pretrained ResNet
        views: Number of views (1 or 2)
        share_weights: Whether to share weights between views
        
    Returns:
        Simple FSRA model
    """
    if views == 1:
        return SimpleFSRAModel(
            num_classes=num_classes,
            num_regions=num_regions,
            use_pretrained=use_pretrained
        )
    elif views == 2:
        return TwoViewSimpleFSRA(
            num_classes=num_classes,
            num_regions=num_regions,
            use_pretrained=use_pretrained,
            share_weights=share_weights
        )
    else:
        raise ValueError(f"Unsupported number of views: {views}")
