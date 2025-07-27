#!/usr/bin/env python3
"""Test dataloader with your dataset structure."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_dataset_structure(data_dir):
    """Check the dataset structure."""
    print(f"Checking dataset structure in: {data_dir}")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Data directory {data_dir} does not exist")
        return False
    
    # Check for direct structure (data/satellite, data/drone)
    direct_satellite = data_path / "satellite"
    direct_drone = data_path / "drone"
    
    # Check for train structure (data/train/satellite, data/train/drone)
    train_satellite = data_path / "train" / "satellite"
    train_drone = data_path / "train" / "drone"
    
    print("\nDataset structure analysis:")
    print(f"Direct satellite: {direct_satellite.exists()} ({direct_satellite})")
    print(f"Direct drone: {direct_drone.exists()} ({direct_drone})")
    print(f"Train satellite: {train_satellite.exists()} ({train_satellite})")
    print(f"Train drone: {train_drone.exists()} ({train_drone})")
    
    if direct_satellite.exists() and direct_drone.exists():
        print("✓ Using direct structure (data/satellite, data/drone)")
        return True, str(data_path)
    elif train_satellite.exists() and train_drone.exists():
        print("✓ Using train structure (data/train/satellite, data/train/drone)")
        return True, str(data_path / "train")
    else:
        print("❌ No valid structure found")
        return False, None

def count_images_in_directory(directory):
    """Count images in a directory."""
    if not Path(directory).exists():
        return 0
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    count = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                count += 1
    
    return count

def analyze_dataset(data_dir):
    """Analyze the dataset."""
    print(f"\nAnalyzing dataset in: {data_dir}")
    
    structure_valid, actual_data_dir = check_dataset_structure(data_dir)
    if not structure_valid:
        return False
    
    # Count images
    satellite_dir = Path(actual_data_dir) / "satellite"
    drone_dir = Path(actual_data_dir) / "drone"
    
    satellite_count = count_images_in_directory(satellite_dir)
    drone_count = count_images_in_directory(drone_dir)
    
    print(f"\nImage counts:")
    print(f"Satellite images: {satellite_count}")
    print(f"Drone images: {drone_count}")
    
    # Count classes
    satellite_classes = len([d for d in satellite_dir.iterdir() if d.is_dir()]) if satellite_dir.exists() else 0
    drone_classes = len([d for d in drone_dir.iterdir() if d.is_dir()]) if drone_dir.exists() else 0
    
    print(f"\nClass counts:")
    print(f"Satellite classes: {satellite_classes}")
    print(f"Drone classes: {drone_classes}")
    
    # Show some example classes
    if satellite_dir.exists():
        example_classes = sorted([d.name for d in satellite_dir.iterdir() if d.is_dir()])[:5]
        print(f"Example satellite classes: {example_classes}")
    
    if drone_dir.exists():
        example_classes = sorted([d.name for d in drone_dir.iterdir() if d.is_dir()])[:5]
        print(f"Example drone classes: {example_classes}")
    
    return True

def test_dataloader_creation(data_dir):
    """Test dataloader creation."""
    print(f"\nTesting dataloader creation with: {data_dir}")
    
    try:
        from src.datasets import make_dataloader
        from src.utils import load_config
        
        # Create test config
        config = {
            'data': {
                'data_dir': data_dir,
                'batch_size': 4,
                'num_workers': 2,
                'sample_num': 4,
                'views': 2,
                'image_height': 256,
                'image_width': 256
            }
        }
        
        print("Creating dataloader...")
        dataloader, class_names, dataset_sizes = make_dataloader(config, create_dummy=False)
        
        print(f"✓ Dataloader created successfully!")
        print(f"Number of classes: {len(class_names)}")
        print(f"Dataset sizes: {dataset_sizes}")
        print(f"Batch size: {dataloader.batch_size}")
        print(f"Number of batches: {len(dataloader)}")
        
        # Test loading one batch
        print("\nTesting batch loading...")
        for batch_idx, batch_data in enumerate(dataloader):
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                (sat_images, sat_labels), (drone_images, drone_labels) = batch_data
                
                print(f"✓ Batch {batch_idx} loaded successfully!")
                print(f"Satellite images shape: {sat_images.shape}")
                print(f"Drone images shape: {drone_images.shape}")
                print(f"Satellite labels shape: {sat_labels.shape}")
                print(f"Drone labels shape: {drone_labels.shape}")
                print(f"Label range: {sat_labels.min().item()} - {sat_labels.max().item()}")
                
                break  # Only test first batch
            else:
                print(f"❌ Unexpected batch format: {type(batch_data)}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Dataloader creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("Dataset and Dataloader Test")
    print("=" * 60)
    
    # Test with the data directory
    data_dir = "data"
    
    success = True
    
    # Analyze dataset
    if not analyze_dataset(data_dir):
        success = False
    
    # Test dataloader
    if not test_dataloader_creation(data_dir):
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("Your dataset is compatible with the training script.")
        print("\nYou can now run:")
        print(f"python train.py --config config/default_config.yaml --data-dir {data_dir} --batch-size 32 --learning-rate 0.001 --num-epochs 10 --gpu-ids \"0\"")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the dataset structure and try again.")
    print("=" * 60)

if __name__ == "__main__":
    main()
