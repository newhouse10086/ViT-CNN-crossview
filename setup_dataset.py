#!/usr/bin/env python3
"""Setup dataset structure and create dummy data for ViT-CNN-crossview."""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import argparse

def create_directory_structure(base_dir="data", num_classes=10):
    """Create the required directory structure."""
    base_path = Path(base_dir)
    
    # Create main directories
    splits = ['train', 'test']
    views = ['satellite', 'drone']
    
    print(f"Creating dataset structure in {base_path}...")
    
    for split in splits:
        for view in views:
            for class_id in range(num_classes):
                class_name = f"class_{class_id:03d}"
                dir_path = base_path / split / view / class_name
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"Created: {dir_path}")
    
    print(f"✓ Dataset structure created with {num_classes} classes")
    return base_path

def generate_dummy_images(base_dir="data", num_classes=10, images_per_class=20, image_size=(256, 256)):
    """Generate dummy images for testing."""
    base_path = Path(base_dir)
    
    print(f"Generating dummy images ({image_size[0]}x{image_size[1]})...")
    
    splits = ['train', 'test']
    views = ['satellite', 'drone']
    
    total_images = 0
    
    for split in splits:
        split_images = images_per_class if split == 'train' else max(1, images_per_class // 4)
        
        for view in views:
            for class_id in range(num_classes):
                class_name = f"class_{class_id:03d}"
                class_dir = base_path / split / view / class_name
                
                for img_id in range(split_images):
                    # Generate random image
                    if view == 'satellite':
                        # Satellite images: more green/brown (terrain-like)
                        img_array = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
                        img_array[:, :, 1] = np.clip(img_array[:, :, 1] + 30, 0, 255)  # More green
                        img_array[:, :, 2] = np.clip(img_array[:, :, 2] - 20, 0, 255)  # Less blue
                    else:
                        # Drone images: more varied colors
                        img_array = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
                    
                    # Add some structure (simple patterns)
                    center_x, center_y = image_size[0] // 2, image_size[1] // 2
                    radius = min(image_size) // 4
                    
                    # Add a circular pattern based on class
                    y, x = np.ogrid[:image_size[0], :image_size[1]]
                    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
                    
                    # Different pattern for each class
                    pattern_color = [
                        (class_id * 25) % 255,
                        (class_id * 50) % 255,
                        (class_id * 75) % 255
                    ]
                    
                    img_array[mask] = pattern_color
                    
                    # Save image
                    img = Image.fromarray(img_array)
                    img_path = class_dir / f"{view}_{class_name}_{img_id:03d}.jpg"
                    img.save(img_path, quality=85)
                    total_images += 1
                
                print(f"Generated {split_images} images for {split}/{view}/{class_name}")
    
    print(f"✓ Generated {total_images} dummy images")

def create_dataset_info(base_dir="data", num_classes=10):
    """Create dataset information file."""
    base_path = Path(base_dir)
    
    info_content = f"""# ViT-CNN-crossview Dataset Information

## Dataset Structure
- Classes: {num_classes}
- Views: satellite, drone
- Splits: train, test

## Directory Structure
```
{base_dir}/
├── train/
│   ├── satellite/
│   │   ├── class_001/
│   │   ├── class_002/
│   │   └── ... (class_{num_classes:03d})
│   └── drone/
│       ├── class_001/
│       ├── class_002/
│       └── ... (class_{num_classes:03d})
└── test/
    ├── satellite/
    └── drone/
```

## Usage
```bash
# Train with this dataset
python train.py --config config/default_config.yaml --data-dir {base_dir} --batch-size 32 --learning-rate 0.001 --num-epochs 100 --gpu-ids "0"

# Train with smaller batch size for testing
python train.py --config config/pytorch110_config.yaml --data-dir {base_dir} --batch-size 16 --learning-rate 0.001 --num-epochs 10 --gpu-ids "0"
```

## Notes
- This is dummy data for testing purposes
- Replace with real satellite and drone image pairs for actual training
- Each class should contain corresponding satellite and drone views of the same locations
"""
    
    info_path = base_path / "README.md"
    with open(info_path, 'w') as f:
        f.write(info_content)
    
    print(f"✓ Created dataset info: {info_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Setup dataset for ViT-CNN-crossview')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Base directory for dataset')
    parser.add_argument('--num-classes', type=int, default=10,
                       help='Number of classes to create')
    parser.add_argument('--images-per-class', type=int, default=20,
                       help='Number of images per class (train split)')
    parser.add_argument('--image-size', type=int, nargs=2, default=[256, 256],
                       help='Image size (height width)')
    parser.add_argument('--structure-only', action='store_true',
                       help='Create directory structure only (no dummy images)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ViT-CNN-crossview Dataset Setup")
    print("=" * 60)
    
    # Create directory structure
    base_path = create_directory_structure(args.data_dir, args.num_classes)
    
    if not args.structure_only:
        # Generate dummy images
        generate_dummy_images(
            args.data_dir, 
            args.num_classes, 
            args.images_per_class,
            tuple(args.image_size)
        )
    
    # Create dataset info
    create_dataset_info(args.data_dir, args.num_classes)
    
    print("\n" + "=" * 60)
    print("✅ Dataset setup completed!")
    print(f"Dataset location: {base_path.absolute()}")
    print("\nNext steps:")
    print("1. Check the generated dataset structure")
    print("2. Replace dummy images with real data if needed")
    print("3. Run training:")
    print(f"   python train.py --config config/default_config.yaml --data-dir {args.data_dir} --batch-size 32 --learning-rate 0.001 --num-epochs 10 --gpu-ids \"0\"")
    print("=" * 60)

if __name__ == "__main__":
    main()
