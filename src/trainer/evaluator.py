"""Evaluator module for ViT-CNN-crossview."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from ..utils import MetricsCalculator
from ..utils import plot_confusion_matrix, plot_roc_curves

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator class for model evaluation."""
    
    def __init__(self, model: nn.Module, device: torch.device = torch.device('cpu'),
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to use for evaluation
            config: Configuration dictionary
        """
        self.model = model
        self.device = device
        self.config = config or {}
        
        # Metrics calculators
        self.metrics_calculator = MetricsCalculator(
            num_classes=self.config.get('model', {}).get('num_classes', 10)
        )
        # Import RankingMetricsCalculator dynamically to avoid circular imports
        try:
            from ..utils.metrics import RankingMetricsCalculator
            self.ranking_calculator = RankingMetricsCalculator()
        except ImportError:
            logger.warning("RankingMetricsCalculator not available, ranking metrics will be skipped")
            self.ranking_calculator = None
        
        # Results storage
        self.predictions = []
        self.labels = []
        self.features = []
        self.probabilities = []
    
    def evaluate(self, dataloader: DataLoader, 
                save_results: bool = True) -> Dict[str, Any]:
        """
        Evaluate model on dataset.
        
        Args:
            dataloader: Evaluation dataloader
            save_results: Whether to save evaluation results
            
        Returns:
            Evaluation results
        """
        logger.info("Starting model evaluation...")
        
        self.model.eval()
        self.metrics_calculator.reset()
        if self.ranking_calculator:
            self.ranking_calculator.reset()
        
        # Clear previous results
        self.predictions.clear()
        self.labels.clear()
        self.features.clear()
        self.probabilities.clear()
        
        all_query_features = []
        all_gallery_features = []
        all_query_labels = []
        all_gallery_labels = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                try:
                    # Parse batch data
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                        (sat_images, sat_labels), (drone_images, drone_labels) = batch_data
                        
                        # Move to device
                        sat_images = sat_images.to(self.device)
                        drone_images = drone_images.to(self.device)
                        sat_labels = sat_labels.to(self.device)
                        
                        # Forward pass
                        if hasattr(self.model, 'module'):
                            outputs = self.model.module(sat_images, drone_images)
                        else:
                            outputs = self.model(sat_images, drone_images)
                        
                        # Extract predictions and features
                        if 'satellite' in outputs and outputs['satellite'] is not None:
                            sat_preds = outputs['satellite']['predictions']
                            sat_feats = outputs['satellite']['features']
                            
                            if isinstance(sat_preds, list):
                                predictions = sat_preds[0]  # Use first prediction
                                features = sat_feats[0] if isinstance(sat_feats, list) else sat_feats
                            else:
                                predictions = sat_preds
                                features = sat_feats
                            
                            # Convert to probabilities
                            probabilities = torch.softmax(predictions, dim=1)
                            pred_classes = torch.argmax(predictions, dim=1)
                            
                            # Store results
                            self.predictions.extend(pred_classes.cpu().numpy())
                            self.labels.extend(sat_labels.cpu().numpy())
                            self.probabilities.extend(probabilities.cpu().numpy())
                            self.features.extend(features.cpu().numpy())
                            
                            # Update metrics calculators
                            self.metrics_calculator.update(
                                pred_classes.cpu().numpy(),
                                sat_labels.cpu().numpy(),
                                probabilities.cpu().numpy()
                            )
                            
                            # For ranking metrics (satellite as query, drone as gallery)
                            all_query_features.append(features.cpu())
                            all_query_labels.append(sat_labels.cpu())
                        
                        # Process drone features for ranking
                        if 'drone' in outputs and outputs['drone'] is not None:
                            drone_feats = outputs['drone']['features']
                            
                            if isinstance(drone_feats, list):
                                features = drone_feats[0]
                            else:
                                features = drone_feats
                            
                            all_gallery_features.append(features.cpu())
                            all_gallery_labels.append(drone_labels.cpu())
                
                except Exception as e:
                    logger.error(f"Error in evaluation batch {batch_idx}: {e}")
                    continue
                
                if batch_idx % 50 == 0:
                    logger.info(f"Processed {batch_idx}/{len(dataloader)} batches")
        
        # Compute classification metrics
        classification_metrics = self.metrics_calculator.compute_metrics()
        
        # Compute ranking metrics if we have both query and gallery features
        ranking_metrics = {}
        if all_query_features and all_gallery_features and self.ranking_calculator:
            try:
                query_features = torch.cat(all_query_features, dim=0)
                gallery_features = torch.cat(all_gallery_features, dim=0)
                query_labels = torch.cat(all_query_labels, dim=0)
                gallery_labels = torch.cat(all_gallery_labels, dim=0)

                self.ranking_calculator.update(
                    query_features, gallery_features,
                    query_labels, gallery_labels
                )
                ranking_metrics = self.ranking_calculator.compute_ranking_metrics()
            except Exception as e:
                logger.warning(f"Could not compute ranking metrics: {e}")
        
        # Combine all metrics
        results = {
            'classification_metrics': classification_metrics,
            'ranking_metrics': ranking_metrics,
            'num_samples': len(self.predictions),
            'predictions': np.array(self.predictions),
            'labels': np.array(self.labels),
            'probabilities': np.array(self.probabilities),
            'features': np.array(self.features)
        }
        
        # Save results if requested
        if save_results:
            self.save_evaluation_results(results)
        
        # Log summary
        self.log_evaluation_summary(results)
        
        logger.info("Model evaluation completed!")
        
        return results
    
    def save_evaluation_results(self, results: Dict[str, Any]):
        """
        Save evaluation results to files.
        
        Args:
            results: Evaluation results dictionary
        """
        save_dir = Path(self.config.get('system', {}).get('log_dir', 'logs')) / 'evaluation'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics to text file
        metrics_file = save_dir / 'evaluation_metrics.txt'
        with open(metrics_file, 'w') as f:
            f.write("Classification Metrics:\n")
            f.write("=" * 40 + "\n")
            for key, value in results['classification_metrics'].items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.6f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            f.write("\nRanking Metrics:\n")
            f.write("=" * 40 + "\n")
            for key, value in results['ranking_metrics'].items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.6f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        logger.info(f"Saved evaluation metrics to {metrics_file}")
        
        # Save confusion matrix
        try:
            cm = self.metrics_calculator.get_confusion_matrix()
            if cm.size > 0:
                cm_path = save_dir / 'confusion_matrix.png'
                plot_confusion_matrix(
                    cm, 
                    title="Confusion Matrix",
                    save_path=str(cm_path),
                    show=False
                )
                logger.info(f"Saved confusion matrix to {cm_path}")
        except Exception as e:
            logger.warning(f"Could not save confusion matrix: {e}")
        
        # Save ROC curves
        try:
            if len(results['probabilities']) > 0 and results['probabilities'].ndim == 2:
                roc_path = save_dir / 'roc_curves.png'
                plot_roc_curves(
                    results['labels'],
                    results['probabilities'],
                    title="ROC Curves",
                    save_path=str(roc_path),
                    show=False
                )
                logger.info(f"Saved ROC curves to {roc_path}")
        except Exception as e:
            logger.warning(f"Could not save ROC curves: {e}")
        
        # Save detailed classification report
        try:
            report = self.metrics_calculator.get_classification_report()
            if report:
                report_file = save_dir / 'classification_report.txt'
                with open(report_file, 'w') as f:
                    f.write(report)
                logger.info(f"Saved classification report to {report_file}")
        except Exception as e:
            logger.warning(f"Could not save classification report: {e}")
    
    def log_evaluation_summary(self, results: Dict[str, Any]):
        """
        Log evaluation summary.
        
        Args:
            results: Evaluation results dictionary
        """
        logger.info("Evaluation Summary:")
        logger.info("=" * 50)
        
        # Classification metrics
        cls_metrics = results['classification_metrics']
        logger.info("Classification Metrics:")
        logger.info(f"  Accuracy: {cls_metrics.get('accuracy', 0.0):.4f}")
        logger.info(f"  Precision (macro): {cls_metrics.get('precision_macro', 0.0):.4f}")
        logger.info(f"  Recall (macro): {cls_metrics.get('recall_macro', 0.0):.4f}")
        logger.info(f"  F1-Score (macro): {cls_metrics.get('f1_score_macro', 0.0):.4f}")
        
        if 'auc_roc' in cls_metrics:
            logger.info(f"  AUC-ROC: {cls_metrics['auc_roc']:.4f}")
        
        # Ranking metrics
        rank_metrics = results['ranking_metrics']
        if rank_metrics:
            logger.info("Ranking Metrics:")
            for k in [1, 5, 10]:
                if f'rank_{k}_accuracy' in rank_metrics:
                    logger.info(f"  Rank-{k} Accuracy: {rank_metrics[f'rank_{k}_accuracy']:.4f}")
            
            if 'mAP' in rank_metrics:
                logger.info(f"  mAP: {rank_metrics['mAP']:.4f}")
        
        logger.info(f"Total samples evaluated: {results['num_samples']}")
        logger.info("=" * 50)
    
    def evaluate_single_batch(self, batch_data: Any) -> Dict[str, Any]:
        """
        Evaluate a single batch.
        
        Args:
            batch_data: Single batch of data
            
        Returns:
            Batch evaluation results
        """
        self.model.eval()
        
        with torch.no_grad():
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                (sat_images, sat_labels), (drone_images, drone_labels) = batch_data
                
                # Move to device
                sat_images = sat_images.to(self.device)
                drone_images = drone_images.to(self.device)
                sat_labels = sat_labels.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'module'):
                    outputs = self.model.module(sat_images, drone_images)
                else:
                    outputs = self.model(sat_images, drone_images)
                
                # Extract predictions
                if 'satellite' in outputs and outputs['satellite'] is not None:
                    sat_preds = outputs['satellite']['predictions']
                    
                    if isinstance(sat_preds, list):
                        predictions = sat_preds[0]
                    else:
                        predictions = sat_preds
                    
                    probabilities = torch.softmax(predictions, dim=1)
                    pred_classes = torch.argmax(predictions, dim=1)
                    
                    # Compute accuracy for this batch
                    correct = (pred_classes == sat_labels).float().sum()
                    accuracy = correct / sat_labels.size(0)
                    
                    return {
                        'predictions': pred_classes.cpu().numpy(),
                        'probabilities': probabilities.cpu().numpy(),
                        'labels': sat_labels.cpu().numpy(),
                        'accuracy': accuracy.item(),
                        'batch_size': sat_labels.size(0)
                    }
        
        return {}
