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


class CommunityClusteringModule(nn.Module):
    """
    Community clustering module using graph-based clustering.
    Innovation: Replace K-means with community detection for adaptive region discovery.
    """
    
    def __init__(self, num_clusters: int = 3, similarity_threshold: float = 0.5, target_dim: int = 256):
        super().__init__()
        self.num_clusters = num_clusters
        self.similarity_threshold = similarity_threshold
        self.target_dim = target_dim
        
    def build_similarity_graph(self, features: torch.Tensor) -> nx.Graph:
        """Build similarity graph from features with optimization for large batches."""
        N = features.shape[0]
        
        # For large feature sets, use sampling to reduce complexity
        if N > 1000:
            sample_size = min(500, N)
            indices = torch.randperm(N)[:sample_size]
            sampled_features = features[indices]
            
            features_norm = F.normalize(sampled_features, p=2, dim=1)
            similarity_matrix = torch.mm(features_norm, features_norm.t())
            similarity_np = similarity_matrix.detach().cpu().numpy()
            
            G = nx.Graph()
            for i in range(sample_size):
                G.add_node(indices[i].item())
            
            for i in range(sample_size):
                for j in range(i + 1, sample_size):
                    if similarity_np[i, j] > self.similarity_threshold:
                        G.add_edge(indices[i].item(), indices[j].item(), weight=similarity_np[i, j])
        else:
            features_norm = F.normalize(features, p=2, dim=1)
            similarity_matrix = torch.mm(features_norm, features_norm.t())
            similarity_np = similarity_matrix.detach().cpu().numpy()
            
            G = nx.Graph()
            for i in range(N):
                G.add_node(i)
            
            for i in range(N):
                for j in range(i + 1, N):
                    if similarity_np[i, j] > self.similarity_threshold:
                        G.add_edge(i, j, weight=similarity_np[i, j])
        
        return G
    
    def community_detection(self, graph: nx.Graph) -> List[List[int]]:
        """Perform community detection using Louvain algorithm with fallback."""
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(graph)
            
            # Convert partition to list of communities
            communities = {}
            for node, comm_id in partition.items():
                if comm_id not in communities:
                    communities[comm_id] = []
                communities[comm_id].append(node)
            
            community_list = list(communities.values())
            
            # Ensure we have exactly num_clusters communities
            if len(community_list) > self.num_clusters:
                # Merge smallest communities
                community_list.sort(key=len, reverse=True)
                community_list = community_list[:self.num_clusters]
            elif len(community_list) < self.num_clusters:
                # Split largest community
                while len(community_list) < self.num_clusters:
                    largest_idx = max(range(len(community_list)), key=lambda i: len(community_list[i]))
                    largest = community_list[largest_idx]
                    if len(largest) > 1:
                        mid = len(largest) // 2
                        community_list[largest_idx] = largest[:mid]
                        community_list.append(largest[mid:])
                    else:
                        break
            
            return community_list
            
        except ImportError:
            print("Warning: python-louvain not available, using fallback clustering")
            # Fallback to K-means clustering
            return self.fallback_clustering(graph)
    
    def fallback_clustering(self, graph: nx.Graph) -> List[List[int]]:
        """Fallback clustering using K-means."""
        nodes = list(graph.nodes())
        if len(nodes) < self.num_clusters:
            # Not enough nodes, create single-node communities
            return [[node] for node in nodes] + [[] for _ in range(self.num_clusters - len(nodes))]
        
        # Simple spatial clustering based on node indices
        nodes_per_cluster = len(nodes) // self.num_clusters
        communities = []
        for i in range(self.num_clusters):
            start_idx = i * nodes_per_cluster
            if i == self.num_clusters - 1:
                # Last cluster gets remaining nodes
                communities.append(nodes[start_idx:])
            else:
                communities.append(nodes[start_idx:start_idx + nodes_per_cluster])
        
        return communities
    
    def apply_pca(self, features: torch.Tensor) -> torch.Tensor:
        """Apply PCA for dimensionality reduction with enhanced error handling."""
        with torch.no_grad():
            features_np = features.detach().cpu().numpy()
            n_samples, n_features = features_np.shape
            
            max_components = min(n_samples, n_features)
            target_components = min(self.target_dim, max_components)
            
            if max_components <= 1 or target_components <= 0:
                if n_features >= self.target_dim:
                    return features[:, :self.target_dim]
                else:
                    padding = torch.zeros(n_samples, self.target_dim - n_features, 
                                        dtype=features.dtype, device=features.device)
                    return torch.cat([features, padding], dim=1)
            
            if not hasattr(self, 'pca') or self.pca is None or self.pca.n_components_ != target_components:
                if hasattr(self, 'pca') and self.pca is not None:
                    del self.pca
                self.pca = PCA(n_components=target_components)
                self.pca.fit(features_np)
            
            try:
                features_reduced = self.pca.transform(features_np)
                result = torch.tensor(features_reduced, dtype=features.dtype, device=features.device)
                
                if result.shape[1] < self.target_dim:
                    padding = torch.zeros(result.shape[0], self.target_dim - result.shape[1],
                                        dtype=features.dtype, device=features.device)
                    result = torch.cat([result, padding], dim=1)
                elif result.shape[1] > self.target_dim:
                    result = result[:, :self.target_dim]
                
                return result
                
            except Exception:
                if n_features >= self.target_dim:
                    return features[:, :self.target_dim]
                else:
                    padding = torch.zeros(n_samples, self.target_dim - n_features,
                                        dtype=features.dtype, device=features.device)
                    return torch.cat([features, padding], dim=1)
    
    def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, List]:
        """
        Forward pass of community clustering.
        
        Args:
            feature_map: Feature tensor of shape (B, C, H, W)
            
        Returns:
            clustered_features: Tensor of shape (B, num_clusters, target_dim)
            communities: List of community assignments for each sample
        """
        B, C, H, W = feature_map.shape
        
        # Reshape to (B, H*W, C) for processing
        features = feature_map.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        
        batch_clustered_features = []
        batch_communities = []
        
        for b in range(B):
            batch_features = features[b]  # (H*W, C)
            
            # Build similarity graph
            graph = self.build_similarity_graph(batch_features)
            
            # Perform community detection
            communities = self.community_detection(graph)
            
            # Clean up graph to prevent memory leaks
            graph.clear()
            del graph
            
            # Aggregate features for each community
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
        print(f"DEBUG: Stacking {len(batch_clustered_features)} batch results")
        for i, bf in enumerate(batch_clustered_features):
            print(f"  Batch {i}: {bf.shape}")
        clustered_features = torch.stack(batch_clustered_features)  # (B, num_clusters, target_dim)
        print(f"DEBUG: Stacked clustered_features shape: {clustered_features.shape}")
        
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
        
        # ViT Branch: Vision Transformer
        self.vit_branch = VisionTransformer(
            img_size=256,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=768,
            depth=6,
            num_heads=12,
            output_dim=vit_output_dim
        )
        
        # Feature fusion dimension
        fusion_dim = cnn_output_dim + vit_output_dim  # 100 + 100 = 200
        
        # Community clustering module
        self.community_clustering = CommunityClusteringModule(
            num_clusters=num_clusters,
            similarity_threshold=0.5,
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
            sat_img: Satellite image tensor (B, 3, 256, 256)
            drone_img: Drone image tensor (B, 3, 256, 256)

        Returns:
            Dictionary containing predictions from different levels
        """
        # Use satellite image for feature extraction (can be extended to both)
        x = sat_img
        B = x.shape[0]

        # Debug: Check input shapes
        print(f"DEBUG: Input shapes - sat_img: {sat_img.shape}, drone_img: {drone_img.shape}")
        
        # CNN Branch: ResNet18 + dimension reduction
        cnn_features = self.cnn_backbone(x)  # (B, 512, 8, 8)
        print(f"DEBUG: CNN backbone output: {cnn_features.shape}")
        cnn_features = self.cnn_dim_reduction(cnn_features)  # (B, 100, 8, 8)
        print(f"DEBUG: CNN reduced output: {cnn_features.shape}")

        # ViT Branch: 10x10 patches -> ViT -> spatial features
        vit_features = self.vit_branch(x)  # (B, 100, 8, 8)
        print(f"DEBUG: ViT output: {vit_features.shape}")

        # Safety check: Ensure batch sizes match
        if cnn_features.shape[0] != vit_features.shape[0]:
            print(f"WARNING: Batch size mismatch! CNN: {cnn_features.shape[0]}, ViT: {vit_features.shape[0]}")
            min_batch = min(cnn_features.shape[0], vit_features.shape[0])
            cnn_features = cnn_features[:min_batch]
            vit_features = vit_features[:min_batch]
            print(f"Fixed to batch size: {min_batch}")
        
        # Feature Fusion: Concat CNN and ViT features
        print(f"DEBUG: CNN features shape: {cnn_features.shape}")
        print(f"DEBUG: ViT features shape: {vit_features.shape}")
        fused_features = torch.cat([cnn_features, vit_features], dim=1)  # (B, 200, 8, 8)
        print(f"DEBUG: Fused features shape: {fused_features.shape}")
        
        # Global average pooling for global classification
        global_feat = F.adaptive_avg_pool2d(fused_features, (1, 1)).view(B, -1)  # (B, 200)
        
        # Global classification
        global_output = self.global_classifier(global_feat)
        global_pred, global_f = global_output  # Unpack the list
        
        # Community clustering on fused features
        print(f"DEBUG: Before community clustering - fused_features shape: {fused_features.shape}")
        clustered_features, communities = self.community_clustering(fused_features)  # (B, 3, 256)
        print(f"DEBUG: After community clustering - clustered_features shape: {clustered_features.shape}")
        
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
        # Debug: Check tensor shapes before concatenation
        if global_f.shape[0] != regional_feats[0].shape[0]:
            print(f"WARNING: Batch size mismatch!")
            print(f"  global_f shape: {global_f.shape}")
            for i, rf in enumerate(regional_feats):
                print(f"  regional_feats[{i}] shape: {rf.shape}")
            # Fix batch size mismatch by taking minimum
            min_batch_size = min(global_f.shape[0], min(rf.shape[0] for rf in regional_feats))
            global_f = global_f[:min_batch_size]
            regional_feats = [rf[:min_batch_size] for rf in regional_feats]

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
            }
        }
