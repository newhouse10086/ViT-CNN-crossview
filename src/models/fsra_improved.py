"""
FSRA Improved Model with Community Clustering
Based on: "A Transformer-Based Feature Segmentation and Region Alignment Method For UAV-View Geo-Localization"
Innovation: Replace K-means clustering with community clustering + PCA for feature alignment
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


class CommunityClusteringModule(nn.Module):
    """Community clustering module for feature segmentation."""
    
    def __init__(self, feature_dim: int = 512, num_clusters: int = 3):
        super(CommunityClusteringModule, self).__init__()
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters
        
        # Graph construction parameters
        self.similarity_threshold = 0.5
        
        # PCA for dimensionality reduction (will be fitted during training)
        self.pca = None
        self.target_dim = 256  # Target dimension after PCA
        
    def build_similarity_graph(self, features: torch.Tensor) -> nx.Graph:
        """
        Build similarity graph from features.
        
        Args:
            features: Feature tensor of shape (H*W, D)
            
        Returns:
            NetworkX graph
        """
        # Compute pairwise similarities
        features_norm = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.mm(features_norm, features_norm.t())
        
        # Convert to numpy for NetworkX (detach to avoid gradient issues)
        similarity_np = similarity_matrix.detach().cpu().numpy()
        
        # Build graph
        G = nx.Graph()
        num_nodes = similarity_np.shape[0]
        
        # Add nodes
        for i in range(num_nodes):
            G.add_node(i)
        
        # Add edges based on similarity threshold
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if similarity_np[i, j] > self.similarity_threshold:
                    G.add_edge(i, j, weight=similarity_np[i, j])
        
        return G
    
    def community_detection(self, graph: nx.Graph) -> List[List[int]]:
        """
        Perform community detection using Louvain algorithm.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            List of communities (each community is a list of node indices)
        """
        try:
            # Use Louvain algorithm for community detection
            import community as community_louvain
            partition = community_louvain.best_partition(graph)
        except ImportError:
            # Fallback to simple clustering if community package not available
            print("Warning: python-louvain not available, using fallback clustering")
            return self._fallback_clustering(graph)
        
        # Convert partition to community list
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
        
        # Ensure we have exactly num_clusters communities
        community_list = list(communities.values())
        
        if len(community_list) > self.num_clusters:
            # Merge smallest communities
            community_list.sort(key=len, reverse=True)
            community_list = community_list[:self.num_clusters]
        elif len(community_list) < self.num_clusters:
            # Split largest community
            while len(community_list) < self.num_clusters:
                largest_comm = max(community_list, key=len)
                if len(largest_comm) <= 1:
                    break
                community_list.remove(largest_comm)
                mid = len(largest_comm) // 2
                community_list.extend([largest_comm[:mid], largest_comm[mid:]])
        
        return community_list[:self.num_clusters]
    
    def _fallback_clustering(self, graph: nx.Graph) -> List[List[int]]:
        """Fallback clustering method using K-means."""
        nodes = list(graph.nodes())
        if len(nodes) <= self.num_clusters:
            return [[node] for node in nodes]
        
        # Simple spatial clustering based on node indices
        nodes_per_cluster = len(nodes) // self.num_clusters
        communities = []
        for i in range(self.num_clusters):
            start_idx = i * nodes_per_cluster
            end_idx = start_idx + nodes_per_cluster if i < self.num_clusters - 1 else len(nodes)
            communities.append(nodes[start_idx:end_idx])
        
        return communities
    
    def apply_pca(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply PCA for dimensionality reduction.

        Args:
            features: Feature tensor of shape (N, D)

        Returns:
            Reduced features of shape (N, target_dim)
        """
        with torch.no_grad():
            features_np = features.detach().cpu().numpy()
            n_samples, n_features = features_np.shape

            # Determine appropriate PCA components
            max_components = min(n_samples, n_features)
            target_components = min(self.target_dim, max_components)

            # Skip PCA if not enough samples or features
            if max_components <= 1 or target_components <= 0:
                # Return original features or zero-padded/truncated features
                if n_features >= self.target_dim:
                    return features[:, :self.target_dim]
                else:
                    # Pad with zeros if needed
                    padding = torch.zeros(n_samples, self.target_dim - n_features,
                                        dtype=features.dtype, device=features.device)
                    return torch.cat([features, padding], dim=1)

            # Initialize PCA if needed
            if self.pca is None or self.pca.n_components_ != target_components:
                self.pca = PCA(n_components=target_components)
                self.pca.fit(features_np)

            # Transform features
            try:
                features_reduced = self.pca.transform(features_np)
                result = torch.tensor(features_reduced, dtype=features.dtype, device=features.device)

                # Pad or truncate to target dimension
                if result.shape[1] < self.target_dim:
                    padding = torch.zeros(result.shape[0], self.target_dim - result.shape[1],
                                        dtype=features.dtype, device=features.device)
                    result = torch.cat([result, padding], dim=1)
                elif result.shape[1] > self.target_dim:
                    result = result[:, :self.target_dim]

                return result

            except Exception as e:
                # Fallback: return truncated or padded original features
                if n_features >= self.target_dim:
                    return features[:, :self.target_dim]
                else:
                    padding = torch.zeros(n_samples, self.target_dim - n_features,
                                        dtype=features.dtype, device=features.device)
                    return torch.cat([features, padding], dim=1)
    
    def forward(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, List[List[int]]]:
        """
        Forward pass for community clustering.
        
        Args:
            feature_map: Feature map of shape (B, C, H, W)
            
        Returns:
            Tuple of (clustered_features, communities)
            clustered_features: (B, num_clusters, target_dim)
            communities: List of community assignments for each batch
        """
        B, C, H, W = feature_map.shape
        
        # Reshape to (B, H*W, C)
        features = feature_map.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        
        batch_clustered_features = []
        batch_communities = []
        
        for b in range(B):
            batch_features = features[b]  # (H*W, C)
            
            # Build similarity graph
            graph = self.build_similarity_graph(batch_features)
            
            # Perform community detection
            communities = self.community_detection(graph)
            
            # Aggregate features for each community
            clustered_features = []
            for community in communities:
                if len(community) > 0:
                    community_features = batch_features[community].mean(dim=0)  # (C,)
                else:
                    # Empty community, use zero features
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


class FSRAImprovedModel(nn.Module):
    """
    FSRA Improved Model with Community Clustering.
    
    Based on FSRA paper with community clustering innovation.
    """
    
    def __init__(self, num_classes: int, num_clusters: int = 3, 
                 use_pretrained: bool = True, feature_dim: int = 512):
        super(FSRAImprovedModel, self).__init__()
        
        self.num_classes = num_classes
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        
        # ResNet backbone (following FSRA)
        self.backbone = resnet18_backbone(pretrained=use_pretrained)
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256)
            backbone_output = self.backbone(dummy_input)
            backbone_dim = backbone_output.shape[1]
        
        # Feature projection to standard dimension
        self.feature_projection = nn.Sequential(
            nn.Conv2d(backbone_dim, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Community clustering module
        self.community_clustering = CommunityClusteringModule(
            feature_dim=feature_dim, 
            num_clusters=num_clusters
        )
        
        # Global feature classifier
        self.global_classifier = ClassBlock(
            input_dim=feature_dim,
            class_num=num_classes,
            return_f=True
        )
        
        # Regional classifiers for each cluster
        # Use target_dim as bottleneck to maintain consistent feature dimensions
        self.regional_classifiers = nn.ModuleList([
            ClassBlock(
                input_dim=self.community_clustering.target_dim,
                class_num=num_classes,
                num_bottleneck=self.community_clustering.target_dim,  # Keep same dimension
                return_f=True
            ) for _ in range(num_clusters)
        ])

        print(f"Regional classifier input dim: {self.community_clustering.target_dim}")
        print(f"Expected regional feature output dim: {self.community_clustering.target_dim}")
        
        # Feature fusion for final prediction
        # Calculate fusion dimension dynamically
        # global_feat: feature_dim, regional_feats: num_clusters * target_dim
        fusion_dim = feature_dim + num_clusters * self.community_clustering.target_dim

        print(f"Fusion dim calculation: {feature_dim} + {num_clusters} * {self.community_clustering.target_dim} = {fusion_dim}")

        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Final classifier
        self.final_classifier = ClassBlock(
            input_dim=feature_dim,
            class_num=num_classes,
            return_f=True
        )
        
        # Initialize weights
        self.apply(weights_init_kaiming)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Tuple of (predictions, features)
        """
        # Extract backbone features
        backbone_features = self.backbone(x)  # (B, backbone_dim, H', W')
        
        # Project to standard feature dimension
        feature_map = self.feature_projection(backbone_features)  # (B, feature_dim, H', W')
        
        # Global average pooling for global features
        global_features = F.adaptive_avg_pool2d(feature_map, (1, 1)).squeeze(-1).squeeze(-1)  # (B, feature_dim)
        
        # Global classification
        global_pred, global_feat = self.global_classifier(global_features)
        
        # Community clustering
        clustered_features, communities = self.community_clustering(feature_map)  # (B, num_clusters, target_dim)
        print(f"Clustered features shape: {clustered_features.shape}")
        
        # Regional classification
        regional_preds = []
        regional_feats = []
        
        for i in range(self.num_clusters):
            regional_feat = clustered_features[:, i, :]  # (B, target_dim)
            regional_pred, regional_f = self.regional_classifiers[i](regional_feat)
            regional_preds.append(regional_pred)
            regional_feats.append(regional_f)
        
        # Feature fusion
        print(f"Global feat shape: {global_feat.shape}")
        for i, rf in enumerate(regional_feats):
            print(f"Regional feat {i} shape: {rf.shape}")

        all_features = torch.cat([global_feat] + regional_feats, dim=1)  # (B, fusion_dim)
        print(f"All features shape: {all_features.shape}")
        print(f"Expected fusion input dim: {self.feature_fusion[0].in_features}")

        fused_features = self.feature_fusion(all_features)  # (B, feature_dim)
        
        # Final classification
        final_pred, final_feat = self.final_classifier(fused_features)
        
        # Collect all predictions and features
        predictions = [global_pred] + regional_preds + [final_pred]
        features = [global_feat] + regional_feats + [final_feat]
        
        return predictions, features


def make_fsra_improved_model(num_classes: int, num_clusters: int = 3,
                           use_pretrained: bool = True, feature_dim: int = 512) -> FSRAImprovedModel:
    """
    Create FSRA Improved model.
    
    Args:
        num_classes: Number of classes
        num_clusters: Number of clusters for community detection
        use_pretrained: Whether to use pretrained ResNet
        feature_dim: Feature dimension
        
    Returns:
        FSRA Improved model
    """
    return FSRAImprovedModel(
        num_classes=num_classes,
        num_clusters=num_clusters,
        use_pretrained=use_pretrained,
        feature_dim=feature_dim
    )
