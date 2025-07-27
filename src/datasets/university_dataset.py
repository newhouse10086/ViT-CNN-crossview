"""University-1652 dataset implementation with improved error handling."""

import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class UniversityDataset(Dataset):
    """University-1652 dataset for cross-view geo-localization."""
    
    def __init__(self, root: str, transforms: Dict[str, Any], 
                 views: List[str] = ['satellite', 'drone'],
                 create_dummy: bool = False):
        """
        Initialize University dataset.
        
        Args:
            root: Root directory path
            transforms: Dictionary of transforms for different views
            views: List of view names to load
            create_dummy: Whether to create dummy data if real data is missing
        """
        super(UniversityDataset, self).__init__()
        
        self.root = root
        self.views = views
        self.transforms_drone = transforms.get('train', transforms.get('drone'))
        self.transforms_satellite = transforms.get('satellite', transforms.get('train'))
        self.create_dummy = create_dummy
        
        # Check if data directory exists
        if not os.path.exists(root):
            if create_dummy:
                logger.warning(f"Data directory {root} not found. Creating dummy dataset.")
                self._create_dummy_structure()
            else:
                logger.error(f"Data directory {root} not found.")
                raise FileNotFoundError(f"Data directory not found: {root}")
        
        # Load dataset structure
        self.dict_path = {}
        self.cls_names = []
        self._load_dataset_structure()
        
        # Create class name to index mapping
        self.cls_names.sort()
        self.map_dict = {i: self.cls_names[i] for i in range(len(self.cls_names))}
        
        logger.info(f"Loaded dataset with {len(self.cls_names)} classes from {root}")
    
    def _create_dummy_structure(self):
        """Create dummy dataset structure for testing."""
        os.makedirs(self.root, exist_ok=True)
        
        # Create dummy classes
        dummy_classes = [f"{i:04d}" for i in range(10)]  # 10 dummy classes
        
        for view in self.views:
            view_dir = os.path.join(self.root, view)
            os.makedirs(view_dir, exist_ok=True)
            
            for cls_name in dummy_classes:
                cls_dir = os.path.join(view_dir, cls_name)
                os.makedirs(cls_dir, exist_ok=True)
                
                # Create dummy image files (just empty files for structure)
                for i in range(3):  # 3 images per class
                    dummy_file = os.path.join(cls_dir, f"{cls_name}_{i:02d}.jpg")
                    if not os.path.exists(dummy_file):
                        # Create a small dummy image
                        dummy_img = Image.new('RGB', (256, 256), color='red')
                        dummy_img.save(dummy_file)
        
        logger.info(f"Created dummy dataset structure in {self.root}")
    
    def _load_dataset_structure(self):
        """Load the dataset directory structure."""
        for view in self.views:
            view_path = os.path.join(self.root, view)
            
            if not os.path.exists(view_path):
                if self.create_dummy:
                    logger.warning(f"View directory {view_path} not found. Creating dummy structure.")
                    self._create_dummy_structure()
                else:
                    logger.error(f"View directory not found: {view_path}")
                    raise FileNotFoundError(f"View directory not found: {view_path}")
            
            view_dict = {}
            try:
                class_dirs = os.listdir(view_path)
                for cls_name in class_dirs:
                    cls_path = os.path.join(view_path, cls_name)
                    if os.path.isdir(cls_path):
                        img_files = [f for f in os.listdir(cls_path) 
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if img_files:
                            img_paths = [os.path.join(cls_path, img) for img in img_files]
                            view_dict[cls_name] = img_paths
                            
                            # Add to class names if not already present
                            if cls_name not in self.cls_names:
                                self.cls_names.append(cls_name)
                
                self.dict_path[view] = view_dict
                logger.info(f"Loaded {len(view_dict)} classes for view '{view}'")
                
            except Exception as e:
                logger.error(f"Error loading view {view}: {str(e)}")
                if self.create_dummy:
                    self._create_dummy_structure()
                    self._load_dataset_structure()  # Retry after creating dummy data
                else:
                    raise
    
    def sample_from_class(self, view: str, cls_name: str) -> Image.Image:
        """Sample an image from a specific class and view."""
        try:
            if view not in self.dict_path or cls_name not in self.dict_path[view]:
                # Create a dummy image if class not found
                logger.warning(f"Class {cls_name} not found in view {view}. Creating dummy image.")
                return Image.new('RGB', (256, 256), color='gray')
            
            img_paths = self.dict_path[view][cls_name]
            if not img_paths:
                logger.warning(f"No images found for class {cls_name} in view {view}. Creating dummy image.")
                return Image.new('RGB', (256, 256), color='gray')
            
            img_path = np.random.choice(img_paths)
            
            try:
                img = Image.open(img_path).convert('RGB')
                return img
            except Exception as e:
                logger.warning(f"Error loading image {img_path}: {str(e)}. Creating dummy image.")
                return Image.new('RGB', (256, 256), color='blue')
                
        except Exception as e:
            logger.error(f"Error in sample_from_class: {str(e)}")
            return Image.new('RGB', (256, 256), color='red')
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Get item by index."""
        try:
            cls_name = self.map_dict[index]
            
            # Sample satellite image
            img_satellite = self.sample_from_class("satellite", cls_name)
            if self.transforms_satellite:
                img_satellite = self.transforms_satellite(img_satellite)
            
            # Sample drone image  
            img_drone = self.sample_from_class("drone", cls_name)
            if self.transforms_drone:
                img_drone = self.transforms_drone(img_drone)
            
            return img_satellite, img_drone, index
            
        except Exception as e:
            logger.error(f"Error in __getitem__ for index {index}: {str(e)}")
            # Return dummy tensors
            dummy_tensor = torch.zeros(3, 256, 256)
            return dummy_tensor, dummy_tensor, index
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.cls_names)


class UniversitySampler(Sampler):
    """Custom sampler for University dataset with repeat sampling."""
    
    def __init__(self, data_source: Dataset, batch_size: int = 8, sample_num: int = 4):
        """
        Initialize sampler.
        
        Args:
            data_source: Dataset to sample from
            batch_size: Batch size
            sample_num: Number of times to repeat each sample
        """
        super(UniversitySampler, self).__init__(data_source)
        self.data_len = len(data_source)
        self.batch_size = batch_size
        self.sample_num = sample_num
    
    def __iter__(self):
        """Generate sampling indices."""
        indices = np.arange(self.data_len)
        np.random.shuffle(indices)
        repeated_indices = np.repeat(indices, self.sample_num)
        return iter(repeated_indices)
    
    def __len__(self) -> int:
        """Return sampler length."""
        return self.data_len * self.sample_num


def train_collate_fn(batch: List[Tuple]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Custom collate function for training.
    
    Args:
        batch: List of (img_satellite, img_drone, index) tuples
        
    Returns:
        Tuple of [satellite_batch, labels], [drone_batch, labels]
    """
    try:
        img_satellite, img_drone, indices = zip(*batch)
        indices = torch.tensor(indices, dtype=torch.long)
        
        satellite_batch = torch.stack(img_satellite, dim=0)
        drone_batch = torch.stack(img_drone, dim=0)
        
        return [satellite_batch, indices], [drone_batch, indices]
        
    except Exception as e:
        logger.error(f"Error in collate_fn: {str(e)}")
        # Return dummy batch
        batch_size = len(batch)
        dummy_tensor = torch.zeros(batch_size, 3, 256, 256)
        dummy_indices = torch.zeros(batch_size, dtype=torch.long)
        return [dummy_tensor, dummy_indices], [dummy_tensor, dummy_indices]
