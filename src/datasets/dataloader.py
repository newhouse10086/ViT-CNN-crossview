"""Data loading utilities for ViT-CNN-crossview."""

import torch
from torch.utils.data import DataLoader
import os
import logging
from typing import Dict, Any, Tuple, Optional

from .university_dataset import UniversityDataset, UniversitySampler, train_collate_fn
from .transforms import get_transforms

logger = logging.getLogger(__name__)


def create_dummy_dataset(data_dir: str, num_classes: int = 10, images_per_class: int = 3):
    """
    Create a dummy dataset for testing when real data is not available.
    
    Args:
        data_dir: Directory to create dummy data
        num_classes: Number of dummy classes to create
        images_per_class: Number of images per class
    """
    from PIL import Image
    import numpy as np
    
    os.makedirs(data_dir, exist_ok=True)
    
    views = ['satellite', 'drone']
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'cyan']
    
    for view in views:
        view_dir = os.path.join(data_dir, view)
        os.makedirs(view_dir, exist_ok=True)
        
        for i in range(num_classes):
            cls_name = f"{i:04d}"
            cls_dir = os.path.join(view_dir, cls_name)
            os.makedirs(cls_dir, exist_ok=True)
            
            for j in range(images_per_class):
                img_path = os.path.join(cls_dir, f"{cls_name}_{j:02d}.jpg")
                if not os.path.exists(img_path):
                    # Create a dummy image with random noise and class-specific color
                    color = colors[i % len(colors)]
                    img = Image.new('RGB', (256, 256), color=color)
                    
                    # Add some random noise
                    img_array = np.array(img)
                    noise = np.random.randint(0, 50, img_array.shape, dtype=np.uint8)
                    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    img = Image.fromarray(img_array)
                    
                    img.save(img_path)
    
    logger.info(f"Created dummy dataset with {num_classes} classes in {data_dir}")


def make_dataloader(config: Dict[str, Any], create_dummy: bool = True) -> Tuple[DataLoader, list, Dict[str, int]]:
    """
    Create data loader for training.
    
    Args:
        config: Configuration dictionary
        create_dummy: Whether to create dummy data if real data is missing
        
    Returns:
        Tuple of (dataloader, class_names, dataset_sizes)
    """
    data_dir = config['data']['data_dir']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    sample_num = config['data']['sample_num']
    views = config['data']['views']
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        if create_dummy:
            logger.warning(f"Data directory {data_dir} not found. Creating dummy dataset.")
            create_dummy_dataset(data_dir)
        else:
            logger.error(f"Data directory {data_dir} not found.")
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Check if required view directories exist
    view_names = ['satellite', 'drone'] if views == 2 else ['satellite', 'drone', 'street']
    missing_views = []

    # First check if views are directly in data_dir
    direct_views_exist = True
    for view in view_names:
        view_path = os.path.join(data_dir, view)
        if not os.path.exists(view_path):
            direct_views_exist = False
            break

    # If direct views don't exist, check in train subdirectory
    if not direct_views_exist:
        train_dir = os.path.join(data_dir, 'train')
        if os.path.exists(train_dir):
            logger.info(f"Views not found in {data_dir}, checking {train_dir}")
            data_dir = train_dir  # Update data_dir to point to train directory

            for view in view_names:
                view_path = os.path.join(data_dir, view)
                if not os.path.exists(view_path):
                    missing_views.append(view)
        else:
            # Neither direct views nor train directory exist
            for view in view_names:
                view_path = os.path.join(data_dir, view)
                if not os.path.exists(view_path):
                    missing_views.append(view)
    
    if missing_views:
        if create_dummy:
            logger.warning(f"Missing view directories: {missing_views}. Creating dummy dataset.")
            create_dummy_dataset(data_dir)
        else:
            logger.error(f"Missing view directories: {missing_views}")
            raise FileNotFoundError(f"Missing view directories: {missing_views}")
    
    # Get transforms
    transforms = get_transforms(config)
    
    try:
        # Create dataset
        dataset = UniversityDataset(
            root=data_dir,
            transforms=transforms,
            views=view_names,
            create_dummy=create_dummy
        )
        
        # Create sampler
        sampler = UniversitySampler(
            data_source=dataset,
            batch_size=batch_size,
            sample_num=sample_num
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=train_collate_fn,
            drop_last=True  # Drop last incomplete batch
        )
        
        # Calculate dataset sizes
        dataset_sizes = {
            'satellite': len(dataset) * sample_num,
            'drone': len(dataset) * sample_num
        }
        
        class_names = dataset.cls_names
        
        logger.info(f"Created dataloader with {len(dataset)} classes, batch_size={batch_size}")
        logger.info(f"Dataset sizes: {dataset_sizes}")
        
        return dataloader, class_names, dataset_sizes
        
    except Exception as e:
        logger.error(f"Error creating dataloader: {str(e)}")
        
        if create_dummy:
            logger.info("Attempting to create minimal dummy dataset...")
            create_dummy_dataset(data_dir, num_classes=5, images_per_class=2)
            
            # Retry with dummy data
            dataset = UniversityDataset(
                root=data_dir,
                transforms=transforms,
                views=view_names,
                create_dummy=True
            )
            
            sampler = UniversitySampler(
                data_source=dataset,
                batch_size=min(batch_size, len(dataset)),
                sample_num=sample_num
            )
            
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=min(batch_size, len(dataset)),
                sampler=sampler,
                num_workers=0,  # Use 0 workers for dummy data
                pin_memory=False,
                collate_fn=train_collate_fn,
                drop_last=False
            )
            
            dataset_sizes = {
                'satellite': len(dataset) * sample_num,
                'drone': len(dataset) * sample_num
            }
            
            class_names = dataset.cls_names
            
            logger.info(f"Created dummy dataloader with {len(dataset)} classes")
            return dataloader, class_names, dataset_sizes
        else:
            raise


def make_test_dataloader(config: Dict[str, Any]) -> Optional[DataLoader]:
    """
    Create test data loader.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Test dataloader or None if test data not available
    """
    test_dir = config['data'].get('test_dir')
    if not test_dir or not os.path.exists(test_dir):
        logger.warning("Test directory not found. Skipping test dataloader creation.")
        return None
    
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    
    # Get transforms
    transforms = get_transforms(config)
    
    try:
        # Create test dataset
        test_dataset = UniversityDataset(
            root=test_dir,
            transforms={'train': transforms['test'], 'satellite': transforms['test']},
            views=['satellite', 'drone'],
            create_dummy=False
        )
        
        # Create test dataloader (no custom sampler needed)
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=train_collate_fn
        )
        
        logger.info(f"Created test dataloader with {len(test_dataset)} classes")
        return test_dataloader
        
    except Exception as e:
        logger.warning(f"Could not create test dataloader: {str(e)}")
        return None
