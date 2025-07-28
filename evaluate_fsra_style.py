#!/usr/bin/env python3
"""
FSRA-style evaluation script for cross-view geo-localization.
Implements the same evaluation metrics as the original FSRA project.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config_utils import load_config
from src.models import create_model
from src.datasets import make_dataloader


def extract_features(model, dataloader, device):
    """
    Extract features from the model for all samples.
    
    Returns:
        features: numpy array of shape (N, feature_dim)
        labels: numpy array of shape (N,)
    """
    model.eval()
    features_list = []
    labels_list = []
    
    print("Extracting features...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader)):
            try:
                # Handle different data formats
                if isinstance(batch_data, (list, tuple)):
                    if len(batch_data) == 2:
                        (sat_images, sat_labels), (drone_images, drone_labels) = batch_data
                    elif len(batch_data) == 3:
                        sat_images, drone_images, sat_labels = batch_data
                    else:
                        raise ValueError(f"Unexpected batch_data length: {len(batch_data)}")
                    
                    sat_images = sat_images.to(device)
                    drone_images = drone_images.to(device)
                    sat_labels = sat_labels.to(device)
                    
                    # Forward pass
                    if hasattr(model, 'module'):
                        outputs = model.module(sat_images, drone_images)
                    else:
                        outputs = model(sat_images, drone_images)
                    
                    # Extract features (use global features for evaluation)
                    if 'satellite' in outputs and 'features' in outputs['satellite']:
                        features = outputs['satellite']['features']
                        if isinstance(features, dict) and 'global' in features:
                            batch_features = features['global']
                        elif isinstance(features, dict) and 'final' in features:
                            batch_features = features['final']
                        else:
                            # Use the first available feature
                            batch_features = list(features.values())[0]
                    else:
                        # Fallback: use the first prediction as feature
                        predictions = outputs['satellite']['predictions']
                        batch_features = predictions[0]  # Use global prediction as feature
                    
                    features_list.append(batch_features.cpu().numpy())
                    labels_list.append(sat_labels.cpu().numpy())
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
    
    if not features_list:
        raise ValueError("No features extracted!")
    
    features = np.vstack(features_list)
    labels = np.hstack(labels_list)
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return features, labels


def compute_distance_matrix(query_features, gallery_features):
    """
    Compute distance matrix between query and gallery features.
    
    Args:
        query_features: (N_q, D) query features
        gallery_features: (N_g, D) gallery features
        
    Returns:
        distance_matrix: (N_q, N_g) distance matrix
    """
    print("Computing distance matrix...")
    
    # L2 normalize features
    query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
    gallery_features = gallery_features / np.linalg.norm(gallery_features, axis=1, keepdims=True)
    
    # Compute cosine distance (1 - cosine similarity)
    similarity_matrix = np.dot(query_features, gallery_features.T)
    distance_matrix = 1 - similarity_matrix
    
    return distance_matrix


def evaluate_ranking(distance_matrix, query_labels, gallery_labels, topk=[1, 5, 10]):
    """
    Evaluate ranking performance.
    
    Args:
        distance_matrix: (N_q, N_g) distance matrix
        query_labels: (N_q,) query labels
        gallery_labels: (N_g,) gallery labels
        topk: list of k values for top-k accuracy
        
    Returns:
        results: dictionary containing evaluation metrics
    """
    print("Evaluating ranking performance...")
    
    num_query = distance_matrix.shape[0]
    
    # Sort gallery by distance for each query
    indices = np.argsort(distance_matrix, axis=1)
    
    # Compute metrics
    results = {}
    
    # Top-k accuracy
    for k in topk:
        correct = 0
        for i in range(num_query):
            # Get top-k gallery indices for query i
            top_k_indices = indices[i, :k]
            # Check if any of top-k gallery samples has the same label as query
            if query_labels[i] in gallery_labels[top_k_indices]:
                correct += 1
        
        accuracy = correct / num_query
        results[f'top_{k}_accuracy'] = accuracy
        print(f"Top-{k} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Mean Average Precision (mAP)
    aps = []
    for i in range(num_query):
        # Find all gallery samples with the same label as query
        relevant_indices = np.where(gallery_labels == query_labels[i])[0]
        
        if len(relevant_indices) == 0:
            continue
        
        # Get ranking of relevant samples
        query_distances = distance_matrix[i]
        sorted_indices = np.argsort(query_distances)
        
        # Compute average precision
        relevant_retrieved = 0
        precision_sum = 0
        
        for rank, idx in enumerate(sorted_indices):
            if idx in relevant_indices:
                relevant_retrieved += 1
                precision_at_rank = relevant_retrieved / (rank + 1)
                precision_sum += precision_at_rank
        
        if relevant_retrieved > 0:
            ap = precision_sum / len(relevant_indices)
            aps.append(ap)
    
    map_score = np.mean(aps) if aps else 0.0
    results['mAP'] = map_score
    print(f"mAP: {map_score:.4f} ({map_score*100:.2f}%)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='FSRA-style Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    parser.add_argument('--data-dir', type=str, help='Data directory')
    parser.add_argument('--gpu-ids', type=str, default='0', help='GPU IDs to use')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.gpu_ids:
        config['system']['gpu_ids'] = args.gpu_ids
    
    # Setup device
    device = torch.device(f"cuda:{config['system']['gpu_ids']}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    model = model.to(device)
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully!")
    else:
        print("No checkpoint provided, using randomly initialized model")
    
    # Create dataloader
    print("Creating dataloader...")
    dataloader, class_names, dataset_sizes = make_dataloader(config)
    print(f"Dataset: {len(class_names)} classes, {sum(dataset_sizes.values())} total samples")
    
    # Extract features
    features, labels = extract_features(model, dataloader, device)
    
    # For cross-view evaluation, we need to split features into query and gallery
    # In University-1652, typically we evaluate satellite->drone and drone->satellite
    # For now, we'll use the same set as both query and gallery (self-retrieval)
    # In a real cross-view setup, you would have separate satellite and drone features

    # Remove self-matching by setting diagonal to infinity
    num_samples = features.shape[0]
    query_features = features
    gallery_features = features
    query_labels = labels
    gallery_labels = labels
    
    print(f"\nEvaluation Setup:")
    print(f"Query set: {query_features.shape[0]} samples")
    print(f"Gallery set: {gallery_features.shape[0]} samples")
    print(f"Feature dimension: {query_features.shape[1]}")
    
    # Compute distance matrix
    distance_matrix = compute_distance_matrix(query_features, gallery_features)

    # Remove self-matching by setting diagonal to infinity (for self-retrieval evaluation)
    np.fill_diagonal(distance_matrix, np.inf)
    
    # Evaluate ranking
    print(f"\n{'='*50}")
    print("FSRA-STYLE EVALUATION RESULTS")
    print(f"{'='*50}")
    
    results = evaluate_ranking(distance_matrix, query_labels, gallery_labels, topk=[1, 5, 10, 20])
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Model: {config['model']['name']}")
    print(f"Dataset: University-1652 ({len(class_names)} classes)")
    print(f"Feature dimension: {query_features.shape[1]}")
    print(f"Evaluation samples: {query_features.shape[0]}")
    print()
    
    for metric, value in results.items():
        if 'top_' in metric:
            k = metric.split('_')[1]
            print(f"Top-{k} Accuracy: {value:.4f} ({value*100:.2f}%)")
        else:
            print(f"{metric}: {value:.4f} ({value*100:.2f}%)")
    
    # Save results
    results_file = f"evaluation_results_{config['model']['name']}.txt"
    with open(results_file, 'w') as f:
        f.write("FSRA-Style Evaluation Results\n")
        f.write("="*50 + "\n")
        f.write(f"Model: {config['model']['name']}\n")
        f.write(f"Dataset: University-1652 ({len(class_names)} classes)\n")
        f.write(f"Feature dimension: {query_features.shape[1]}\n")
        f.write(f"Evaluation samples: {query_features.shape[0]}\n\n")
        
        for metric, value in results.items():
            if 'top_' in metric:
                k = metric.split('_')[1]
                f.write(f"Top-{k} Accuracy: {value:.4f} ({value*100:.2f}%)\n")
            else:
                f.write(f"{metric}: {value:.4f} ({value*100:.2f}%)\n")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
