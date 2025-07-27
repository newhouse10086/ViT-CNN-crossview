"""ViT-CNN hybrid model for cross-view geo-localization."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import networkx as nx
from sklearn.cluster import KMeans

from .backbones.resnet import resnet18_backbone
from .backbones.vit_pytorch import vit_small_patch16_224
from .components import ClassBlock, CrossViewAlignment, FeatureFusion, weights_init_kaiming


class CommunityClusteringModule(nn.Module):
    """Community clustering module using graph networks."""
    
    def __init__(self, feature_dim: int = 768, num_clusters: int = 3):
        """
        Initialize community clustering module.
        
        Args:
            feature_dim: Input feature dimension
            num_clusters: Number of final clusters
        """
        super(CommunityClusteringModule, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters
        
        # Graph construction layers
        self.edge_weight_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
        
        # Feature refinement after clustering
        self.cluster_refinement = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(num_clusters)
        ])
        
        self.apply(weights_init_kaiming)
    
    def construct_graph(self, features: torch.Tensor) -> torch.Tensor:
        """
        Construct graph from features.
        
        Args:
            features: Feature tensor of shape (B, N, D)
            
        Returns:
            Adjacency matrix of shape (B, N, N)
        """
        B, N, D = features.shape
        
        # Compute pairwise features
        features_i = features.unsqueeze(2).expand(B, N, N, D)
        features_j = features.unsqueeze(1).expand(B, N, N, D)
        pairwise_features = torch.cat([features_i, features_j], dim=-1)
        
        # Compute edge weights
        edge_weights = self.edge_weight_net(pairwise_features).squeeze(-1)
        
        return edge_weights
    
    def community_detection(self, adjacency_matrix: torch.Tensor) -> List[List[int]]:
        """
        Perform community detection on graph.
        
        Args:
            adjacency_matrix: Adjacency matrix (N, N)
            
        Returns:
            List of communities (each community is a list of node indices)
        """
        # Convert to NetworkX graph
        G = nx.from_numpy_array(adjacency_matrix.detach().cpu().numpy())
        
        # Perform community detection
        try:
            communities = nx.community.greedy_modularity_communities(G)
            communities = [list(community) for community in communities]
        except:
            # Fallback to simple clustering if community detection fails
            n_nodes = adjacency_matrix.shape[0]
            nodes_per_cluster = n_nodes // self.num_clusters
            communities = []
            for i in range(self.num_clusters):
                start_idx = i * nodes_per_cluster
                end_idx = start_idx + nodes_per_cluster if i < self.num_clusters - 1 else n_nodes
                communities.append(list(range(start_idx, end_idx)))
        
        return communities
    
    def forward(self, features: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass for community clustering.
        
        Args:
            features: Feature tensor of shape (B, N, D)
            
        Returns:
            Tuple of (clustered_features, adjacency_matrix)
        """
        B, N, D = features.shape
        
        # Construct graph
        adjacency_matrix = self.construct_graph(features)
        
        # Perform clustering for each sample in batch
        clustered_features = []
        
        for b in range(B):
            # Get communities for this sample
            communities = self.community_detection(adjacency_matrix[b])
            
            # Ensure we have exactly num_clusters communities
            while len(communities) < self.num_clusters:
                # Split largest community
                largest_community = max(communities, key=len)
                communities.remove(largest_community)
                mid = len(largest_community) // 2
                communities.extend([largest_community[:mid], largest_community[mid:]])
            
            while len(communities) > self.num_clusters:
                # Merge smallest communities
                communities.sort(key=len)
                communities[0].extend(communities[1])
                communities.pop(1)
            
            # Extract cluster features
            sample_clusters = []
            for i, community in enumerate(communities):
                if len(community) > 0:
                    cluster_features = features[b, community].mean(dim=0)
                    cluster_features = self.cluster_refinement[i](cluster_features)
                    sample_clusters.append(cluster_features)
                else:
                    # Empty cluster, use zero features
                    sample_clusters.append(torch.zeros(D, device=features.device))
            
            clustered_features.append(torch.stack(sample_clusters))
        
        clustered_features = torch.stack(clustered_features)
        
        return clustered_features, adjacency_matrix


class ViTCNNModel(nn.Module):
    """ViT-CNN hybrid model for geo-localization."""
    
    def __init__(self, num_classes: int, num_clusters: int = 3, 
                 use_pretrained_resnet: bool = True, use_pretrained_vit: bool = False,
                 return_f: bool = False):
        """
        Initialize ViT-CNN model.
        
        Args:
            num_classes: Number of classes
            num_clusters: Number of clusters for community clustering
            use_pretrained_resnet: Whether to use pretrained ResNet
            use_pretrained_vit: Whether to use pretrained ViT
            return_f: Whether to return features
        """
        super(ViTCNNModel, self).__init__()
        
        self.num_classes = num_classes
        self.num_clusters = num_clusters
        self.return_f = return_f
        
        # ResNet18 backbone for initial feature extraction
        self.resnet_backbone = resnet18_backbone(pretrained=use_pretrained_resnet)
        
        # Projection layer to match ViT input dimension
        self.feature_projection = nn.Sequential(
            nn.Conv2d(512, 768, kernel_size=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        
        # ViT for processing ResNet features
        self.vit = vit_small_patch16_224(
            pretrained=use_pretrained_vit,
            img_size=(16, 16),  # ResNet output is 16x16 for 256x256 input
            patch_size=1,  # 1x1 patches for fine-grained processing
            in_chans=768,
            embed_dim=768,
            local_feature=True
        )
        
        # Community clustering module
        self.community_clustering = CommunityClusteringModule(
            feature_dim=768,
            num_clusters=num_clusters
        )
        
        # K-means clustering for final region assignment
        self.kmeans_clusters = num_clusters
        
        # Cross-view alignment
        self.cross_view_alignment = CrossViewAlignment(feature_dim=768)
        
        # Global classifier
        self.global_classifier = ClassBlock(
            input_dim=768,
            class_num=num_classes,
            return_f=return_f
        )
        
        # Regional classifiers
        self.regional_classifiers = nn.ModuleList([
            ClassBlock(
                input_dim=768,
                class_num=num_classes,
                return_f=return_f
            ) for _ in range(num_clusters)
        ])
        
        # Feature fusion for final prediction
        self.feature_fusion = FeatureFusion(
            input_dims=[768] * (num_clusters + 1),  # global + regional features
            output_dim=768
        )
        
        # Final classifier
        self.final_classifier = ClassBlock(
            input_dim=768,
            class_num=num_classes,
            return_f=return_f
        )
        
        self.apply(weights_init_kaiming)
    
    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract features using ResNet + ViT pipeline.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Tuple of (global_features, local_features, attention_weights)
        """
        # ResNet feature extraction
        resnet_features = self.resnet_backbone(x)  # (B, 512, 16, 16)
        
        # Project to ViT dimension
        projected_features = self.feature_projection(resnet_features)  # (B, 768, 16, 16)
        
        # ViT processing
        global_features, local_features, _ = self.vit(projected_features)
        
        # Get attention weights from last layer
        attention_weights = self.vit.get_last_selfattention(projected_features)
        
        return global_features, local_features, attention_weights
    
    def cluster_features(self, local_features: torch.Tensor, 
                        attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Cluster local features using community detection and K-means.
        
        Args:
            local_features: Local feature tensor (B, N, D)
            attention_weights: Attention weights (B, H, N, N)
            
        Returns:
            Clustered features (B, num_clusters, D)
        """
        B, N, D = local_features.shape
        
        # Community clustering
        community_features, _ = self.community_clustering(local_features)
        
        # Additional K-means clustering for refinement
        refined_clusters = []
        for b in range(B):
            features_np = local_features[b].detach().cpu().numpy()
            
            try:
                kmeans = KMeans(n_clusters=self.kmeans_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features_np)
                
                # Compute cluster centers
                cluster_centers = []
                for k in range(self.kmeans_clusters):
                    mask = cluster_labels == k
                    if mask.sum() > 0:
                        center = local_features[b][mask].mean(dim=0)
                    else:
                        center = torch.zeros(D, device=local_features.device)
                    cluster_centers.append(center)
                
                refined_clusters.append(torch.stack(cluster_centers))
            except:
                # Fallback to community clustering result
                refined_clusters.append(community_features[b])
        
        return torch.stack(refined_clusters)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Tuple of (predictions, features)
        """
        # Extract features
        global_features, local_features, attention_weights = self.extract_features(x)
        
        # Cluster features
        clustered_features = self.cluster_features(local_features, attention_weights)
        
        # Global classification
        global_pred = self.global_classifier(global_features)
        
        # Regional classification
        regional_preds = []
        regional_features = []
        
        for i in range(self.num_clusters):
            regional_feat = clustered_features[:, i, :]
            regional_pred = self.regional_classifiers[i](regional_feat)
            regional_preds.append(regional_pred)
            regional_features.append(regional_feat)
        
        # Feature fusion
        if self.return_f:
            all_features = [global_pred[1]] + [pred[1] if isinstance(pred, list) else pred 
                                             for pred in regional_preds]
            fused_features = self.feature_fusion(all_features)
            final_pred = self.final_classifier(fused_features)
            
            predictions = [global_pred[0]] + [pred[0] if isinstance(pred, list) else pred 
                                            for pred in regional_preds] + [final_pred[0]]
            features = [global_pred[1]] + [pred[1] if isinstance(pred, list) else pred 
                                         for pred in regional_preds] + [final_pred[1]]
        else:
            all_features = [global_features] + regional_features
            fused_features = self.feature_fusion(all_features)
            final_pred = self.final_classifier(fused_features)
            
            predictions = [global_pred] + regional_preds + [final_pred]
            features = [global_features] + regional_features + [fused_features]
        
        return predictions, features


def make_vit_cnn_model(num_classes: int, num_clusters: int = 3,
                       use_pretrained_resnet: bool = True, use_pretrained_vit: bool = False,
                       return_f: bool = False, views: int = 2, 
                       share_weights: bool = True) -> nn.Module:
    """
    Create ViT-CNN model.
    
    Args:
        num_classes: Number of classes
        num_clusters: Number of clusters
        use_pretrained_resnet: Whether to use pretrained ResNet
        use_pretrained_vit: Whether to use pretrained ViT
        return_f: Whether to return features
        views: Number of views (1 or 2)
        share_weights: Whether to share weights between views
        
    Returns:
        ViT-CNN model
    """
    if views == 1:
        return ViTCNNModel(
            num_classes=num_classes,
            num_clusters=num_clusters,
            use_pretrained_resnet=use_pretrained_resnet,
            use_pretrained_vit=use_pretrained_vit,
            return_f=return_f
        )
    elif views == 2:
        from .two_view_model import TwoViewViTCNN
        return TwoViewViTCNN(
            num_classes=num_classes,
            num_clusters=num_clusters,
            use_pretrained_resnet=use_pretrained_resnet,
            use_pretrained_vit=use_pretrained_vit,
            return_f=return_f,
            share_weights=share_weights
        )
    else:
        raise ValueError(f"Unsupported number of views: {views}")
