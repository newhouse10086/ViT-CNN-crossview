"""Metrics calculation utilities for ViT-CNN-crossview."""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score, top_k_accuracy_score
)
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = "", fmt: str = ":f"):
        """
        Initialize average meter.
        
        Args:
            name: Name of the meter
            fmt: Format string for display
        """
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update statistics.
        
        Args:
            val: New value
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class MetricsCalculator:
    """Calculate various metrics for model evaluation."""
    
    def __init__(self, num_classes: int, average: str = 'macro'):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes
            average: Averaging strategy for multi-class metrics
        """
        self.num_classes = num_classes
        self.average = average
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and labels."""
        self.predictions = []
        self.labels = []
        self.probabilities = []
    
    def update(self, predictions: Union[torch.Tensor, np.ndarray],
               labels: Union[torch.Tensor, np.ndarray],
               probabilities: Optional[Union[torch.Tensor, np.ndarray]] = None):
        """
        Update with new predictions and labels.
        
        Args:
            predictions: Predicted class indices
            labels: Ground truth labels
            probabilities: Predicted probabilities (optional)
        """
        # Convert to numpy arrays
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.detach().cpu().numpy()
        
        self.predictions.extend(predictions.flatten())
        self.labels.extend(labels.flatten())
        
        if probabilities is not None:
            if probabilities.ndim == 1:
                # Binary classification
                self.probabilities.extend(probabilities.flatten())
            else:
                # Multi-class classification
                self.probabilities.extend(probabilities)
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of computed metrics
        """
        if not self.predictions or not self.labels:
            logger.warning("No predictions or labels available for metric computation")
            return {}
        
        y_true = np.array(self.labels)
        y_pred = np.array(self.predictions)
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Handle different averaging strategies
        for avg_type in ['macro', 'micro', 'weighted']:
            try:
                metrics[f'precision_{avg_type}'] = precision_score(
                    y_true, y_pred, average=avg_type, zero_division=0
                )
                metrics[f'recall_{avg_type}'] = recall_score(
                    y_true, y_pred, average=avg_type, zero_division=0
                )
                metrics[f'f1_score_{avg_type}'] = f1_score(
                    y_true, y_pred, average=avg_type, zero_division=0
                )
            except Exception as e:
                logger.warning(f"Error computing {avg_type} metrics: {e}")
        
        # AUC metrics (if probabilities available)
        if self.probabilities:
            try:
                y_prob = np.array(self.probabilities)
                
                if self.num_classes == 2:
                    # Binary classification
                    if y_prob.ndim == 1:
                        metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
                        metrics['auc_pr'] = average_precision_score(y_true, y_prob)
                else:
                    # Multi-class classification
                    if y_prob.ndim == 2 and y_prob.shape[1] == self.num_classes:
                        # One-vs-rest AUC
                        metrics['auc_roc_ovr'] = roc_auc_score(
                            y_true, y_prob, multi_class='ovr', average=self.average
                        )
                        metrics['auc_roc_ovo'] = roc_auc_score(
                            y_true, y_prob, multi_class='ovo', average=self.average
                        )
                        
                        # Top-k accuracy
                        for k in [1, 3, 5]:
                            if k <= self.num_classes:
                                metrics[f'top_{k}_accuracy'] = top_k_accuracy_score(
                                    y_true, y_prob, k=k
                                )
            except Exception as e:
                logger.warning(f"Error computing AUC metrics: {e}")
        
        # Per-class metrics
        try:
            per_class_precision = precision_score(
                y_true, y_pred, average=None, zero_division=0
            )
            per_class_recall = recall_score(
                y_true, y_pred, average=None, zero_division=0
            )
            per_class_f1 = f1_score(
                y_true, y_pred, average=None, zero_division=0
            )
            
            for i in range(len(per_class_precision)):
                metrics[f'precision_class_{i}'] = per_class_precision[i]
                metrics[f'recall_class_{i}'] = per_class_recall[i]
                metrics[f'f1_score_class_{i}'] = per_class_f1[i]
        except Exception as e:
            logger.warning(f"Error computing per-class metrics: {e}")
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix.
        
        Returns:
            Confusion matrix as numpy array
        """
        if not self.predictions or not self.labels:
            return np.array([])
        
        y_true = np.array(self.labels)
        y_pred = np.array(self.predictions)
        
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(self) -> str:
        """
        Get detailed classification report.
        
        Returns:
            Classification report as string
        """
        if not self.predictions or not self.labels:
            return ""
        
        y_true = np.array(self.labels)
        y_pred = np.array(self.predictions)
        
        return classification_report(y_true, y_pred, zero_division=0)


class MultiViewMetricsCalculator:
    """Metrics calculator for multi-view models."""
    
    def __init__(self, num_classes: int, views: List[str] = ['satellite', 'drone']):
        """
        Initialize multi-view metrics calculator.
        
        Args:
            num_classes: Number of classes
            views: List of view names
        """
        self.num_classes = num_classes
        self.views = views
        self.calculators = {view: MetricsCalculator(num_classes) for view in views}
        self.global_calculator = MetricsCalculator(num_classes)
    
    def reset(self):
        """Reset all calculators."""
        for calculator in self.calculators.values():
            calculator.reset()
        self.global_calculator.reset()
    
    def update(self, predictions: Dict[str, torch.Tensor],
               labels: torch.Tensor,
               probabilities: Optional[Dict[str, torch.Tensor]] = None):
        """
        Update with predictions from multiple views.
        
        Args:
            predictions: Dictionary of predictions for each view
            labels: Ground truth labels
            probabilities: Dictionary of probabilities for each view (optional)
        """
        for view in self.views:
            if view in predictions:
                view_probs = probabilities.get(view) if probabilities else None
                self.calculators[view].update(
                    predictions[view], labels, view_probs
                )
        
        # Update global calculator with ensemble predictions
        if len(predictions) > 1:
            # Simple ensemble: average predictions
            ensemble_pred = torch.stack(list(predictions.values())).mean(dim=0)
            ensemble_pred = torch.argmax(ensemble_pred, dim=1)
            
            ensemble_probs = None
            if probabilities and len(probabilities) > 1:
                ensemble_probs = torch.stack(list(probabilities.values())).mean(dim=0)
            
            self.global_calculator.update(ensemble_pred, labels, ensemble_probs)
    
    def compute_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for all views and global ensemble.
        
        Returns:
            Dictionary of metrics for each view and global
        """
        all_metrics = {}
        
        # Compute metrics for each view
        for view in self.views:
            all_metrics[view] = self.calculators[view].compute_metrics()
        
        # Compute global ensemble metrics
        all_metrics['ensemble'] = self.global_calculator.compute_metrics()
        
        return all_metrics


class RankingMetricsCalculator:
    """Calculate ranking metrics for retrieval tasks."""
    
    def __init__(self, k_values: List[int] = [1, 5, 10]):
        """
        Initialize ranking metrics calculator.
        
        Args:
            k_values: List of k values for top-k metrics
        """
        self.k_values = k_values
        self.reset()
    
    def reset(self):
        """Reset stored data."""
        self.query_features = []
        self.gallery_features = []
        self.query_labels = []
        self.gallery_labels = []
    
    def update(self, query_features: torch.Tensor, gallery_features: torch.Tensor,
               query_labels: torch.Tensor, gallery_labels: torch.Tensor):
        """
        Update with new features and labels.
        
        Args:
            query_features: Query features
            gallery_features: Gallery features
            query_labels: Query labels
            gallery_labels: Gallery labels
        """
        self.query_features.append(query_features.detach().cpu())
        self.gallery_features.append(gallery_features.detach().cpu())
        self.query_labels.append(query_labels.detach().cpu())
        self.gallery_labels.append(gallery_labels.detach().cpu())
    
    def compute_ranking_metrics(self) -> Dict[str, float]:
        """
        Compute ranking metrics.
        
        Returns:
            Dictionary of ranking metrics
        """
        if not self.query_features or not self.gallery_features:
            return {}
        
        # Concatenate all features and labels
        query_feats = torch.cat(self.query_features, dim=0)
        gallery_feats = torch.cat(self.gallery_features, dim=0)
        query_lbls = torch.cat(self.query_labels, dim=0)
        gallery_lbls = torch.cat(self.gallery_labels, dim=0)
        
        # Compute distance matrix
        dist_matrix = torch.cdist(query_feats, gallery_feats, p=2)
        
        # Sort by distance
        sorted_indices = torch.argsort(dist_matrix, dim=1)
        
        metrics = {}
        
        # Compute Rank-k accuracy
        for k in self.k_values:
            correct = 0
            for i in range(len(query_lbls)):
                top_k_indices = sorted_indices[i, :k]
                top_k_labels = gallery_lbls[top_k_indices]
                if query_lbls[i] in top_k_labels:
                    correct += 1
            
            metrics[f'rank_{k}_accuracy'] = correct / len(query_lbls)
        
        # Compute mAP (mean Average Precision)
        ap_scores = []
        for i in range(len(query_lbls)):
            query_label = query_lbls[i]
            sorted_gallery_labels = gallery_lbls[sorted_indices[i]]
            
            # Find relevant items
            relevant_mask = (sorted_gallery_labels == query_label)
            
            if relevant_mask.sum() == 0:
                continue
            
            # Compute average precision
            relevant_indices = torch.where(relevant_mask)[0]
            precisions = []
            
            for j, rel_idx in enumerate(relevant_indices):
                precision_at_k = (j + 1) / (rel_idx + 1)
                precisions.append(precision_at_k)
            
            ap_scores.append(np.mean(precisions))
        
        metrics['mAP'] = np.mean(ap_scores) if ap_scores else 0.0
        
        return metrics
