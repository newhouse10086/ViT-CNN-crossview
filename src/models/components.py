"""Common model components for ViT-CNN-crossview."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GeM(nn.Module):
    """Generalized Mean Pooling."""
    
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class ClassBlock(nn.Module):
    """Classification block with optional feature return."""
    
    def __init__(self, input_dim: int, class_num: int, dropout: float = 0.5, 
                 relu: bool = False, num_bottleneck: int = 512, return_f: bool = False):
        """
        Initialize classification block.
        
        Args:
            input_dim: Input feature dimension
            class_num: Number of classes
            dropout: Dropout rate
            relu: Whether to use ReLU activation
            num_bottleneck: Bottleneck dimension
            return_f: Whether to return features
        """
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        
        if dropout > 0:
            add_block += [nn.Dropout(p=dropout)]
        
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x, f]
        else:
            x = self.classifier(x)
            return x


class CrossViewAlignment(nn.Module):
    """Cross-view feature alignment module."""
    
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256):
        """
        Initialize cross-view alignment module.
        
        Args:
            feature_dim: Input feature dimension
            hidden_dim: Hidden dimension for alignment
        """
        super(CrossViewAlignment, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Feature projection layers
        self.satellite_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.drone_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention mechanism for alignment
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.apply(weights_init_kaiming)
    
    def forward(self, satellite_features: torch.Tensor, 
                drone_features: torch.Tensor) -> tuple:
        """
        Forward pass for cross-view alignment.
        
        Args:
            satellite_features: Satellite view features
            drone_features: Drone view features
            
        Returns:
            Tuple of aligned features (satellite, drone)
        """
        # Project features
        sat_proj = self.satellite_proj(satellite_features)
        drone_proj = self.drone_proj(drone_features)
        
        # Cross-attention alignment
        # Satellite attends to drone
        sat_aligned, _ = self.attention(sat_proj, drone_proj, drone_proj)
        
        # Drone attends to satellite
        drone_aligned, _ = self.attention(drone_proj, sat_proj, sat_proj)
        
        # Project back to original dimension
        sat_aligned = self.output_proj(sat_aligned)
        drone_aligned = self.output_proj(drone_aligned)
        
        return sat_aligned, drone_aligned


class FeatureFusion(nn.Module):
    """Feature fusion module for multi-scale features."""
    
    def __init__(self, input_dims: list, output_dim: int = 512):
        """
        Initialize feature fusion module.
        
        Args:
            input_dims: List of input feature dimensions
            output_dim: Output feature dimension
        """
        super(FeatureFusion, self).__init__()
        
        self.input_dims = input_dims
        self.output_dim = output_dim
        
        # Individual projection layers for each input
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(inplace=True)
            ) for dim in input_dims
        ])
        
        # Attention weights for fusion
        self.attention_weights = nn.Sequential(
            nn.Linear(output_dim * len(input_dims), len(input_dims)),
            nn.Softmax(dim=1)
        )
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )
        
        self.apply(weights_init_kaiming)
    
    def forward(self, features: list) -> torch.Tensor:
        """
        Forward pass for feature fusion.
        
        Args:
            features: List of feature tensors
            
        Returns:
            Fused feature tensor
        """
        # Project all features to same dimension
        projected_features = []
        for i, feat in enumerate(features):
            proj_feat = self.projections[i](feat)
            projected_features.append(proj_feat)
        
        # Concatenate for attention computation
        concat_features = torch.cat(projected_features, dim=1)
        
        # Compute attention weights
        weights = self.attention_weights(concat_features)
        
        # Weighted fusion
        fused_feature = torch.zeros_like(projected_features[0])
        for i, feat in enumerate(projected_features):
            fused_feature += weights[:, i:i+1] * feat
        
        # Final projection
        output = self.final_proj(fused_feature)
        
        return output


def weights_init_kaiming(m):
    """Initialize weights using Kaiming initialization."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    """Initialize classifier weights."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
