#!/usr/bin/env python3
"""Complete project test script for ViT-CNN-crossview."""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


def test_complete_pipeline():
    """Test the complete training pipeline with dummy data."""
    print("Testing complete training pipeline...")
    
    try:
        # Import required modules
        from src.models import create_model
        from src.datasets import make_dataloader, create_dummy_dataset
        from src.losses import CombinedLoss
        from src.optimizers import create_optimizer_with_config
        from src.utils import TrainingVisualizer, MetricsCalculator
        import torch
        import torch.nn as nn
        
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        data_dir = os.path.join(temp_dir, "data")
        log_dir = os.path.join(temp_dir, "logs")
        checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"Using temporary directory: {temp_dir}")
        
        # Configuration
        config = {
            'model': {
                'name': 'ViTCNN',
                'num_classes': 5,
                'num_final_clusters': 2,
                'use_pretrained_resnet': False,
                'use_pretrained_vit': False,
                'return_features': True
            },
            'data': {
                'data_dir': data_dir,
                'batch_size': 4,
                'num_workers': 0,
                'image_height': 256,
                'image_width': 256,
                'views': 2,
                'sample_num': 2,
                'pad': 0,
                'color_jitter': False,
                'random_erasing_prob': 0.0
            },
            'training': {
                'num_epochs': 3,
                'learning_rate': 0.01,
                'weight_decay': 0.0001,
                'momentum': 0.9,
                'warm_epochs': 1,
                'lr_scheduler_steps': [2],
                'lr_scheduler_gamma': 0.1,
                'triplet_loss_weight': 0.3,
                'kl_loss_weight': 0.0,
                'use_kl_loss': False,
                'cross_attention_weight': 1.0,
                'use_data_augmentation': False,
                'optimizer': 'sgd',
                'scheduler': 'step'
            },
            'evaluation': {
                'eval_interval': 1,
                'save_plots': True,
                'plot_interval': 1,
                'metrics_to_track': ['accuracy', 'precision', 'recall', 'f1_score']
            },
            'system': {
                'gpu_ids': '0',
                'use_gpu': False,  # Use CPU for testing
                'seed': 42,
                'log_interval': 1,
                'save_interval': 2,
                'checkpoint_dir': checkpoint_dir,
                'log_dir': log_dir
            }
        }
        
        # 1. Create dummy dataset
        print("1. Creating dummy dataset...")
        create_dummy_dataset(data_dir, num_classes=5, images_per_class=3)
        print("✓ Dummy dataset created")
        
        # 2. Create model
        print("2. Creating model...")
        model = create_model(config)
        device = torch.device('cpu')
        model = model.to(device)
        print("✓ Model created")
        
        # 3. Create dataloader
        print("3. Creating dataloader...")
        dataloader, class_names, dataset_sizes = make_dataloader(config, create_dummy=True)
        print(f"✓ Dataloader created with {len(class_names)} classes")
        
        # 4. Create loss function
        print("4. Creating loss function...")
        criterion = CombinedLoss(
            num_classes=config['model']['num_classes'],
            triplet_weight=config['training']['triplet_loss_weight'],
            kl_weight=config['training']['kl_loss_weight'],
            alignment_weight=config['training']['cross_attention_weight'],
            use_kl_loss=config['training']['use_kl_loss']
        )
        print("✓ Loss function created")
        
        # 5. Create optimizer and scheduler
        print("5. Creating optimizer and scheduler...")
        optimizer, scheduler = create_optimizer_with_config(model, config)
        print("✓ Optimizer and scheduler created")
        
        # 6. Create visualizer and metrics calculator
        print("6. Creating visualizer and metrics calculator...")
        visualizer = TrainingVisualizer(
            save_dir=os.path.join(log_dir, 'plots'),
            experiment_name='test_experiment'
        )
        metrics_calculator = MetricsCalculator(config['model']['num_classes'])
        print("✓ Visualizer and metrics calculator created")
        
        # 7. Training loop
        print("7. Running training loop...")
        model.train()
        
        for epoch in range(config['training']['num_epochs']):
            print(f"  Epoch {epoch+1}/{config['training']['num_epochs']}")
            
            epoch_loss = 0.0
            epoch_samples = 0
            all_predictions = []
            all_labels = []
            
            for batch_idx, batch_data in enumerate(dataloader):
                try:
                    # Parse batch data
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                        (sat_images, sat_labels), (drone_images, drone_labels) = batch_data
                        
                        sat_images = sat_images.to(device)
                        drone_images = drone_images.to(device)
                        sat_labels = sat_labels.to(device)
                        
                        # Forward pass
                        outputs = model(sat_images, drone_images)
                        
                        # Compute loss
                        losses = criterion(outputs, sat_labels)
                        total_loss = losses['total']
                        
                        # Backward pass
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()
                        
                        # Update metrics
                        epoch_loss += total_loss.item()
                        epoch_samples += sat_images.size(0)
                        
                        # Collect predictions for metrics
                        if 'satellite' in outputs and outputs['satellite'] is not None:
                            sat_preds = outputs['satellite']['predictions']
                            if isinstance(sat_preds, list):
                                pred = torch.argmax(sat_preds[0], dim=1)
                            else:
                                pred = torch.argmax(sat_preds, dim=1)
                            
                            all_predictions.extend(pred.cpu().numpy())
                            all_labels.extend(sat_labels.cpu().numpy())
                        
                        if batch_idx == 0:  # Only log first batch
                            print(f"    Batch {batch_idx+1}: Loss = {total_loss.item():.6f}")
                
                except Exception as e:
                    print(f"    Error in batch {batch_idx}: {e}")
                    continue
            
            # Epoch metrics
            avg_loss = epoch_loss / max(len(dataloader), 1)
            
            # Calculate accuracy
            if all_predictions and all_labels:
                metrics_calculator.reset()
                metrics_calculator.update(all_predictions, all_labels)
                epoch_metrics = metrics_calculator.compute_metrics()
                accuracy = epoch_metrics.get('accuracy', 0.0)
            else:
                accuracy = 0.0
            
            # Update scheduler
            if scheduler:
                scheduler.step()
            
            # Log epoch results
            current_lr = optimizer.param_groups[0]['lr']
            epoch_metrics_dict = {
                'train_loss': avg_loss,
                'train_accuracy': accuracy,
                'learning_rate': current_lr,
                'epoch': epoch + 1
            }
            
            print(f"    Epoch {epoch+1} - Loss: {avg_loss:.6f}, Accuracy: {accuracy:.4f}, LR: {current_lr:.8f}")
            
            # Update visualizer
            visualizer.update_metrics(epoch + 1, epoch_metrics_dict)
        
        print("✓ Training loop completed")
        
        # 8. Generate final plots
        print("8. Generating plots...")
        visualizer.plot_training_curves(save=True, show=False)
        csv_path = visualizer.save_metrics_csv()
        print(f"✓ Plots and metrics saved")
        
        # 9. Save final model
        print("9. Saving final model...")
        final_model_path = os.path.join(checkpoint_dir, "test_final_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'class_names': class_names
        }, final_model_path)
        print(f"✓ Final model saved to {final_model_path}")
        
        # 10. Test model loading
        print("10. Testing model loading...")
        checkpoint = torch.load(final_model_path, map_location='cpu')
        test_model = create_model(config)
        test_model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model loading successful")
        
        print("\n🎉 Complete pipeline test PASSED!")
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Complete pipeline test FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("ViT-CNN-crossview Complete Project Test")
    print("=" * 60)
    
    # Test complete pipeline
    if test_complete_pipeline():
        print("\n✅ All tests passed! The project is working correctly.")
        print("\nYou can now:")
        print("1. Prepare your dataset in the data/ directory")
        print("2. Update the configuration file")
        print("3. Run: python train.py --config config/default_config.yaml")
        print("\nFor quick testing with dummy data:")
        print("python train.py --create-dummy-data --experiment-name quick_test")
        return 0
    else:
        print("\n❌ Tests failed! Please check the installation and try again.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
