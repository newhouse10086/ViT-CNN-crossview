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
    
    def __init__(self, num_clusters: int = 3, feature_dim: int = 200):
        super().__init__()
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim  # Use consistent feature dimension
        
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
            clustered_features: Tensor of shape (B, num_clusters, feature_dim)
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
            
            # Stack features (no PCA needed, keep original dimension)
            clustered_features = torch.stack(clustered_features)  # (num_clusters, C)
            
            batch_clustered_features.append(clustered_features)
            batch_communities.append(communities)
        
        # Stack batch results
        clustered_features = torch.stack(batch_clustered_features)  # (B, num_clusters, C)
        
        return clustered_features, batch_communities


class FSRAViTImproved(nn.Module):
    """
    FSRA ViT Improved: True ViT+CNN Hybrid with Community Clustering and PCA Alignment.
    Innovation: Combining Vision Transformer with CNN backbone for superior cross-view matching.
    """
    
    def __init__(
        self,
        num_classes: int = 701,
        num_clusters: int = 3,
        patch_size: int = 25,
        cnn_output_dim: int = 100,
        vit_output_dim: int = 100,
        use_pretrained: bool = True,
        use_kmeans_clustering: bool = False  # 新增参数控制聚类
    ):
        super(FSRAViTImproved, self).__init__()
        
        self.num_classes = num_classes
        self.num_clusters = num_clusters
        self.patch_size = patch_size
        self.cnn_output_dim = cnn_output_dim
        self.vit_output_dim = vit_output_dim
        self.use_kmeans_clustering = use_kmeans_clustering  # 存储聚类设置
        
        # CNN Backbone (ResNet18)
        from torchvision.models import resnet18
        self.cnn_backbone = resnet18(pretrained=use_pretrained)
        self.cnn_backbone.fc = nn.Identity()  # Remove final FC layer
        
        # CNN dimension reduction to match ViT output
        self.cnn_dim_reduction = nn.Sequential(
            nn.Conv2d(512, cnn_output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_output_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))  # Ensure 8x8 spatial size
        )
        
        # ViT Branch
        from .vit_module import VisionTransformer
        self.vit_branch = VisionTransformer(
            img_size=250,
            patch_size=patch_size,
            embed_dim=384,  # 简化的ViT配置
            depth=3,
            num_heads=6,
            mlp_ratio=2.0,
            output_dim=vit_output_dim
        )
        
        # Feature fusion
        fusion_dim = cnn_output_dim + vit_output_dim  # 200
        
        # Global classifier (always present)
        self.global_classifier = ClassBlock(
            input_dim=fusion_dim,            # 200
            class_num=num_classes,
            num_bottleneck=fusion_dim,       # 200
            return_f=True
        )
        
        # 条件性创建聚类和区域分类器
        if self.use_kmeans_clustering:
            # K-means clustering module (only if enabled)
            self.kmeans_clustering = KMeansClusteringModule(
                num_clusters=num_clusters,
                feature_dim=fusion_dim
            )
            
            # Regional classifiers (only if clustering enabled)
            self.regional_classifiers = nn.ModuleList([
                ClassBlock(
                    input_dim=fusion_dim,            # 200
                    class_num=num_classes,
                    num_bottleneck=fusion_dim,       # 200
                    return_f=True
                ) for _ in range(num_clusters)
            ])
            
            # Feature fusion for final prediction (with regional features)
            final_fusion_dim = fusion_dim + num_clusters * fusion_dim  # 200 + 3 * 200 = 800
            self.feature_fusion = nn.Sequential(
                nn.Linear(final_fusion_dim, fusion_dim),  # 800 -> 200
                nn.BatchNorm1d(fusion_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            )
            
            self.final_classifier = ClassBlock(
                input_dim=fusion_dim,            # 200
                class_num=num_classes,
                num_bottleneck=fusion_dim,       # 200
                return_f=True
            )
        else:
            # 无聚类模式：只使用全局分类器
            self.kmeans_clustering = None
            self.regional_classifiers = None
            self.feature_fusion = None
            self.final_classifier = None
        
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
        
        # Global classification (always present)
        global_output = self.global_classifier(global_feat)
        global_pred, global_f = global_output  # Unpack the list
        
        # 根据配置决定是否使用聚类
        if self.use_kmeans_clustering and self.kmeans_clustering is not None:
            # 完整模式：使用K-means聚类和区域分类器
            clustered_features, communities = self.kmeans_clustering(fused_features)  # (B, 3, 200)
            
            # Regional classification
            regional_preds = []
            regional_feats = []

            for i, regional_classifier in enumerate(self.regional_classifiers):
                regional_input = clustered_features[:, i, :]  # (B, 200)
                regional_output = regional_classifier(regional_input)
                regional_pred, regional_f = regional_output  # Unpack the list
                regional_preds.append(regional_pred)
                regional_feats.append(regional_f)
            
            # Feature fusion for final prediction
            all_features = torch.cat([global_f] + regional_feats, dim=1)  # (B, 800)
            fused_features_final = self.feature_fusion(all_features)  # (B, 200)
            
            # Final classification
            final_output = self.final_classifier(fused_features_final)
            final_pred, final_f = final_output  # Unpack the list
            
            # Prepare output with all predictions
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
                'alignment': None
            }
        else:
            # 简化模式：只使用全局分类器（大幅提升速度）
            predictions = [global_pred]  # 只有全局预测
            
            return {
                'satellite': {
                    'predictions': predictions,
                    'features': {
                        'global': global_f,
                        'cnn_features': cnn_features,
                        'vit_features': vit_features,
                        'fused_features': fused_features
                    }
                },
                'alignment': None
            }


def create_fsra_vit_improved(
    num_classes: int = 701,
    num_clusters: int = 3,
    patch_size: int = 25,
    cnn_output_dim: int = 100,
    vit_output_dim: int = 100,
    use_pretrained: bool = True,
    use_kmeans_clustering: bool = False  # 新增参数
) -> FSRAViTImproved:
    """
    Create FSRA ViT Improved model.
    """
    return FSRAViTImproved(
        num_classes=num_classes,
        num_clusters=num_clusters,
        patch_size=patch_size,
        cnn_output_dim=cnn_output_dim,
        vit_output_dim=vit_output_dim,
        use_pretrained=use_pretrained,
        use_kmeans_clustering=use_kmeans_clustering
    )
