"""
FSRA ViT Improved Model - True ViT+CNN Hybrid Architecture
Your Innovation: ViT (10x10 patches) + CNN (ResNet) + Community Clustering + PCA Alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
from sklearn.decomposition import PCA
import networkx as nx
from sklearn.cluster import KMeans

from .backbones.resnet import resnet18_backbone
from .components import ClassBlock, weights_init_kaiming
from .vit_module import VisionTransformer


class KMeansClusteringModule(nn.Module):
    """
    Simple K-means clustering module for feature segmentation.
    Replaces complex community detection with stable K-means clustering.
    """
    
    def __init__(self, num_clusters: int = 3, target_dim: int = 256):
        super().__init__()
        self.num_clusters = num_clusters
        self.target_dim = target_dim
        
        # PCA for dimensionality reduction
        self.pca = None
        
    def apply_pca(self, features: torch.Tensor) -> torch.Tensor:
        """Apply PCA for dimensionality alignment."""
        features_np = features.detach().cpu().numpy()
        
        # Get feature dimensions
        if len(features_np.shape) == 2:
            n_samples, n_features = features_np.shape
        else:
            return features  # Return as-is if unexpected shape
        
        # Skip PCA if we already have the target dimension or fewer features
        if n_features <= self.target_dim:
            return features
            
        # Initialize PCA if needed
        if self.pca is None:
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=min(n_features, self.target_dim))
            
        # Fit and transform
        try:
            if not hasattr(self.pca, 'components_'):  # Not fitted yet
                features_transformed = self.pca.fit_transform(features_np)
            else:
                features_transformed = self.pca.transform(features_np)
        except Exception:
            # Fallback: just truncate or pad
            if n_features > self.target_dim:
                features_transformed = features_np[:, :self.target_dim]
            else:
                features_transformed = features_np
        
        return torch.from_numpy(features_transformed).to(features.device).float()
    
    def kmeans_clustering(self, features: torch.Tensor) -> List[List[int]]:
        """Perform K-means clustering on features."""
        features_np = features.detach().cpu().numpy()
        
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_np)
            
            # Convert to list of communities
            communities = [[] for _ in range(self.num_clusters)]
            for idx, label in enumerate(cluster_labels):
                communities[label].append(idx)
            
            return communities
        except Exception:
            # Fallback: simple equal division
            N = features.shape[0]
            chunk_size = N // self.num_clusters
            communities = []
            for i in range(self.num_clusters):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < self.num_clusters - 1 else N
                communities.append(list(range(start_idx, end_idx)))
            return communities
    
    def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, List]:
        """
        Forward pass of K-means clustering.
        
        Args:
            feature_map: Feature tensor of shape (B, C, H, W)
            
        Returns:
            clustered_features: Tensor of shape (B, num_clusters, target_dim)
            communities: List of cluster assignments for each sample
        """
        B, C, H, W = feature_map.shape
        
        # Reshape to (B, H*W, C) for processing
        features = feature_map.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        
        batch_clustered_features = []
        batch_communities = []
        
        for b in range(B):
            batch_features = features[b]  # (H*W, C)
            
            # Perform K-means clustering
            communities = self.kmeans_clustering(batch_features)
            
            # Aggregate features for each cluster
            clustered_features = []
            for community in communities:
                if len(community) > 0:
                    community_features = batch_features[community].mean(dim=0)  # (C,)
                else:
                    community_features = torch.zeros(C, device=feature_map.device)
                clustered_features.append(community_features)
            
            # Stack and apply PCA
            clustered_features = torch.stack(clustered_features)  # (num_clusters, C)
            clustered_features = self.apply_pca(clustered_features)  # (num_clusters, target_dim)
            
            batch_clustered_features.append(clustered_features)
            batch_communities.append(communities)
        
        # Stack batch results
        clustered_features = torch.stack(batch_clustered_features)  # (B, num_clusters, target_dim)
        
        return clustered_features, batch_communities


class FSRAViTImproved(nn.Module):
    """
    FSRA ViT Improved Model - True Innovation Architecture
    
    Your Innovation:
    1. ViT Branch: 10x10 patches -> ViT Transformer -> (B, 100, 8, 8)
    2. CNN Branch: ResNet18 -> Dimension reduction -> (B, 100, 8, 8)  
    3. Fusion: Concat -> (B, 200, 8, 8)
    4. Community Clustering + PCA Alignment
    5. Multi-level Classification
    """
    
    def __init__(
        self,
        num_classes: int = 701,
        num_clusters: int = 3,
        patch_size: int = 10,
        cnn_output_dim: int = 100,
        vit_output_dim: int = 100,
        target_pca_dim: int = 256,
        use_pretrained: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_clusters = num_clusters
        self.patch_size = patch_size
        self.cnn_output_dim = cnn_output_dim
        self.vit_output_dim = vit_output_dim
        
        # CNN Branch: ResNet18 backbone
        self.cnn_backbone = resnet18_backbone(pretrained=use_pretrained)
        
        # CNN dimension reduction: 512 -> cnn_output_dim + spatial alignment
        self.cnn_dim_reduction = nn.Sequential(
            nn.Conv2d(512, cnn_output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(cnn_output_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))  # Ensure 8x8 output to match ViT
        )
        
        # ViT Branch: Vision Transformer optimized for 10x10 patches
        self.vit_branch = VisionTransformer(
            img_size=250,                # Changed from 256 to 250
            patch_size=patch_size,       # This will be 25 from config
            in_channels=3,
            embed_dim=768,
            depth=6,
            num_heads=12,
            output_dim=vit_output_dim    # 100
        )
        
        # Feature fusion dimension
        fusion_dim = cnn_output_dim + vit_output_dim  # 100 + 100 = 200
        
        # K-means clustering module (now working on fewer patches)
        self.kmeans_clustering = KMeansClusteringModule(
            num_clusters=num_clusters,
            target_dim=target_pca_dim
        )
        
        # Global classifier (on fused features)
        self.global_classifier = ClassBlock(
            input_dim=fusion_dim,
            class_num=num_classes,
            num_bottleneck=target_pca_dim,  # Use same bottleneck as regional
            return_f=True
        )
        
        # Regional classifiers for each cluster
        self.regional_classifiers = nn.ModuleList([
            ClassBlock(
                input_dim=target_pca_dim,
                class_num=num_classes,
                num_bottleneck=target_pca_dim,
                return_f=True
            ) for _ in range(num_clusters)
        ])
        
        # Feature fusion for final prediction
        # Now all features are target_pca_dim (256): global + 3 regional = 4 * 256 = 1024
        final_fusion_dim = target_pca_dim + num_clusters * target_pca_dim
        self.feature_fusion = nn.Sequential(
            nn.Linear(final_fusion_dim, target_pca_dim),  # Output consistent dimension
            nn.BatchNorm1d(target_pca_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Final classifier
        self.final_classifier = ClassBlock(
            input_dim=target_pca_dim,  # Input from feature_fusion
            class_num=num_classes,
            num_bottleneck=target_pca_dim,
            return_f=True
        )
        
        # Initialize weights
        self.apply(weights_init_kaiming)
        
    def forward(self, sat_img: torch.Tensor, drone_img: torch.Tensor) -> dict:
        """
        Forward pass of FSRA ViT Improved model.
        
        Args:
            sat_img: Satellite image tensor (B, 3, 250, 250)
            drone_img: Drone image tensor (B, 3, 250, 250)
            
        Returns:
            Dictionary containing predictions and features
        """
        B = sat_img.shape[0]
        
        # Ensure input size is correct (250x250)
        assert sat_img.shape[2:] == (250, 250), f"Satellite image size should be 250x250, got {sat_img.shape[2:]}"
        assert drone_img.shape[2:] == (250, 250), f"Drone image size should be 250x250, got {drone_img.shape[2:]}"
        
        # CNN Branch processing
        sat_cnn_features = self.cnn_backbone(sat_img)   # (B, 512, H, W)
        drone_cnn_features = self.cnn_backbone(drone_img)  # (B, 512, H, W)
        
        # CNN dimension reduction and spatial alignment
        sat_cnn_features = self.cnn_dim_reduction(sat_cnn_features)    # (B, 100, 8, 8)
        drone_cnn_features = self.cnn_dim_reduction(drone_cnn_features)  # (B, 100, 8, 8)
        
        # ViT Branch processing (now with 250x250 input and 100 patches)
        sat_vit_features = self.vit_branch(sat_img)     # (B, 100, 8, 8)
        drone_vit_features = self.vit_branch(drone_img)  # (B, 100, 8, 8)

        # Fuse satellite and drone features
        cnn_features = (sat_cnn_features + drone_cnn_features) / 2  # Average fusion
        vit_features = (sat_vit_features + drone_vit_features) / 2  # Average fusion
        
        # Feature Fusion: Concat CNN and ViT features
        fused_features = torch.cat([cnn_features, vit_features], dim=1)  # (B, 200, 8, 8)
        
        # Global average pooling for global classification
        global_feat = F.adaptive_avg_pool2d(fused_features, (1, 1)).view(B, -1)  # (B, 200)
        
        # Global classification
        global_output = self.global_classifier(global_feat)
        global_pred, global_f = global_output  # Unpack the list
        
        # K-means clustering on fused features (now with better performance)
        clustered_features, communities = self.kmeans_clustering(fused_features)  # (B, 3, 256)
        
        # Regional classification
        regional_preds = []
        regional_feats = []

        for i, regional_classifier in enumerate(self.regional_classifiers):
            regional_input = clustered_features[:, i, :]  # (B, 256)
            regional_output = regional_classifier(regional_input)
            regional_pred, regional_f = regional_output  # Unpack the list
            regional_preds.append(regional_pred)
            regional_feats.append(regional_f)
        
        # Feature fusion for final prediction
        all_features = torch.cat([global_f] + regional_feats, dim=1)  # (B, final_fusion_dim)
        fused_features_final = self.feature_fusion(all_features)  # (B, fusion_dim)
        
        # Final classification
        final_output = self.final_classifier(fused_features_final)
        final_pred, final_f = final_output  # Unpack the list
        
        # Prepare output
        predictions = [global_pred] + regional_preds + [final_pred]
        
        return {
            'satellite': {
                'predictions': predictions,
                'features': {
                    'global': global_f,
                    'regional': regional_feats,
                    'final': final_f,
                    'cnn_features': cnn_features,
                    'vit_features': vit_features,
                    'fused_features': fused_features
                }
            },
            'alignment': None  # No alignment loss for this model
        }


def create_fsra_vit_improved(num_classes=701,
                           num_clusters=3,
                           patch_size=25,                # Changed from 10 to 25
                           cnn_output_dim=100,
                           vit_output_dim=100,
                           target_pca_dim=256):
    """
    Create FSRA ViT Improved model optimized for 10x10 patches.

    Args:
        num_classes: Number of classes for classification
        num_clusters: Number of clusters for community clustering
        patch_size: Patch size for ViT (default: 25 for 250x250 images -> 10x10 patches)
        cnn_output_dim: CNN branch output dimension
        vit_output_dim: ViT branch output dimension
        target_pca_dim: Target dimension for PCA alignment

    Returns:
        FSRAViTImproved model instance
    """
    return FSRAViTImproved(
        num_classes=num_classes,
        num_clusters=num_clusters,
        patch_size=patch_size,
        cnn_output_dim=cnn_output_dim,
        vit_output_dim=vit_output_dim,
        target_pca_dim=target_pca_dim
    )
