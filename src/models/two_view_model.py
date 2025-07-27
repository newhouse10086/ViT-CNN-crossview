"""Two-view model for cross-view geo-localization."""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from .vit_cnn_model import ViTCNNModel
from .components import CrossViewAlignment


class TwoViewViTCNN(nn.Module):
    """Two-view ViT-CNN model for satellite and drone images."""
    
    def __init__(self, num_classes: int, num_clusters: int = 3,
                 use_pretrained_resnet: bool = True, use_pretrained_vit: bool = False,
                 return_f: bool = False, share_weights: bool = True):
        """
        Initialize two-view ViT-CNN model.
        
        Args:
            num_classes: Number of classes
            num_clusters: Number of clusters
            use_pretrained_resnet: Whether to use pretrained ResNet
            use_pretrained_vit: Whether to use pretrained ViT
            return_f: Whether to return features
            share_weights: Whether to share weights between views
        """
        super(TwoViewViTCNN, self).__init__()
        
        self.share_weights = share_weights
        self.return_f = return_f
        self.num_clusters = num_clusters
        
        # First model for satellite images
        self.model_1 = ViTCNNModel(
            num_classes=num_classes,
            num_clusters=num_clusters,
            use_pretrained_resnet=use_pretrained_resnet,
            use_pretrained_vit=use_pretrained_vit,
            return_f=return_f
        )
        
        # Second model for drone images (shared or separate)
        if share_weights:
            self.model_2 = self.model_1
        else:
            self.model_2 = ViTCNNModel(
                num_classes=num_classes,
                num_clusters=num_clusters,
                use_pretrained_resnet=use_pretrained_resnet,
                use_pretrained_vit=use_pretrained_vit,
                return_f=return_f
            )
        
        # Cross-view alignment module
        self.cross_view_alignment = CrossViewAlignment(feature_dim=768)
        
        # Alignment loss weight
        self.alignment_weight = 1.0
    
    def forward(self, x1: Optional[torch.Tensor], x2: Optional[torch.Tensor]):
        """
        Forward pass for two views.
        
        Args:
            x1: Satellite image tensor
            x2: Drone image tensor
            
        Returns:
            Tuple of outputs from both models and alignment features
        """
        outputs = {}
        
        # Process satellite view
        if x1 is not None:
            sat_predictions, sat_features = self.model_1(x1)
            outputs['satellite'] = {
                'predictions': sat_predictions,
                'features': sat_features
            }
        else:
            outputs['satellite'] = None
        
        # Process drone view
        if x2 is not None:
            drone_predictions, drone_features = self.model_2(x2)
            outputs['drone'] = {
                'predictions': drone_predictions,
                'features': drone_features
            }
        else:
            outputs['drone'] = None
        
        # Cross-view alignment if both views are available
        if x1 is not None and x2 is not None:
            # Use global features for alignment
            sat_global_feat = sat_features[0] if self.return_f else sat_features[0]
            drone_global_feat = drone_features[0] if self.return_f else drone_features[0]
            
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
    
    def get_features(self, x1: Optional[torch.Tensor], x2: Optional[torch.Tensor]):
        """
        Extract features from both views.
        
        Args:
            x1: Satellite image tensor
            x2: Drone image tensor
            
        Returns:
            Dictionary of features from both views
        """
        features = {}
        
        if x1 is not None:
            _, sat_features = self.model_1(x1)
            features['satellite'] = sat_features
        
        if x2 is not None:
            _, drone_features = self.model_2(x2)
            features['drone'] = drone_features
        
        return features
    
    def compute_alignment_loss(self, outputs: dict) -> torch.Tensor:
        """
        Compute cross-view alignment loss.
        
        Args:
            outputs: Model outputs containing alignment information
            
        Returns:
            Alignment loss tensor
        """
        if outputs['alignment'] is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        alignment_info = outputs['alignment']
        
        # MSE loss between aligned and original features
        sat_loss = nn.functional.mse_loss(
            alignment_info['satellite_aligned'],
            alignment_info['drone_original']
        )
        
        drone_loss = nn.functional.mse_loss(
            alignment_info['drone_aligned'],
            alignment_info['satellite_original']
        )
        
        return (sat_loss + drone_loss) * self.alignment_weight
    
    def get_predictions_and_features(self, outputs: dict):
        """
        Extract predictions and features in the format expected by the training loop.
        
        Args:
            outputs: Model outputs
            
        Returns:
            Tuple of (sat_outputs, drone_outputs, sat_features, drone_features)
        """
        sat_outputs = outputs['satellite']['predictions'] if outputs['satellite'] else None
        drone_outputs = outputs['drone']['predictions'] if outputs['drone'] else None
        sat_features = outputs['satellite']['features'] if outputs['satellite'] else None
        drone_features = outputs['drone']['features'] if outputs['drone'] else None
        
        return sat_outputs, drone_outputs, sat_features, drone_features


class FSRAModel(nn.Module):
    """Original FSRA model for compatibility."""
    
    def __init__(self, num_classes: int, block_size: int = 3, return_f: bool = False):
        """
        Initialize FSRA model.
        
        Args:
            num_classes: Number of classes
            block_size: Number of local blocks
            return_f: Whether to return features
        """
        super(FSRAModel, self).__init__()
        
        self.num_classes = num_classes
        self.block_size = block_size
        self.return_f = return_f
        
        # Use ViT backbone
        from .backbones.vit_pytorch import vit_small_patch16_224
        self.backbone = vit_small_patch16_224(
            pretrained=True,
            num_classes=num_classes,
            local_feature=True
        )
        
        # Global classifier
        from .components import ClassBlock
        self.global_classifier = ClassBlock(
            input_dim=384,  # ViT-Small dimension
            class_num=num_classes,
            return_f=return_f
        )
        
        # Regional classifiers
        self.regional_classifiers = nn.ModuleList([
            ClassBlock(
                input_dim=384,
                class_num=num_classes,
                return_f=return_f
            ) for _ in range(block_size)
        ])
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Model outputs
        """
        # Extract features
        global_features, local_features, _ = self.backbone(x)
        
        # Global classification
        global_output = self.global_classifier(global_features)
        
        # Regional classification using K-means clustering
        B, N, D = local_features.shape
        
        # Simple clustering by spatial location
        regional_outputs = []
        patches_per_region = N // self.block_size
        
        for i in range(self.block_size):
            start_idx = i * patches_per_region
            end_idx = start_idx + patches_per_region if i < self.block_size - 1 else N
            
            regional_features = local_features[:, start_idx:end_idx].mean(dim=1)
            regional_output = self.regional_classifiers[i](regional_features)
            regional_outputs.append(regional_output)
        
        if self.return_f:
            predictions = [global_output[0]] + [out[0] if isinstance(out, list) else out 
                                              for out in regional_outputs]
            features = [global_output[1]] + [out[1] if isinstance(out, list) else out 
                                           for out in regional_outputs]
            return predictions, features
        else:
            return [global_output] + regional_outputs


def make_fsra_model(num_classes: int, block_size: int = 3, return_f: bool = False,
                    views: int = 2, share_weights: bool = True) -> nn.Module:
    """
    Create FSRA model.
    
    Args:
        num_classes: Number of classes
        block_size: Number of local blocks
        return_f: Whether to return features
        views: Number of views (1 or 2)
        share_weights: Whether to share weights between views
        
    Returns:
        FSRA model
    """
    if views == 1:
        return FSRAModel(
            num_classes=num_classes,
            block_size=block_size,
            return_f=return_f
        )
    elif views == 2:
        class TwoViewFSRA(nn.Module):
            def __init__(self):
                super().__init__()
                self.model_1 = FSRAModel(num_classes, block_size, return_f)
                if share_weights:
                    self.model_2 = self.model_1
                else:
                    self.model_2 = FSRAModel(num_classes, block_size, return_f)
            
            def forward(self, x1, x2):
                y1 = self.model_1(x1) if x1 is not None else None
                y2 = self.model_2(x2) if x2 is not None else None
                return y1, y2
        
        return TwoViewFSRA()
    else:
        raise ValueError(f"Unsupported number of views: {views}")
