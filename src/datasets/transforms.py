"""Data transforms for ViT-CNN-crossview."""

from torchvision import transforms
from typing import Dict, Any
import torch


class RandomErasing:
    """Random erasing augmentation."""
    
    def __init__(self, probability: float = 0.5, sl: float = 0.02, sh: float = 0.4,
                 r1: float = 0.3, mean: list = [0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
    
    def __call__(self, img):
        if torch.rand(1) >= self.probability:
            return img
        
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            
            target_area = torch.rand(1) * (self.sh - self.sl) + self.sl
            target_area = target_area * area
            
            aspect_ratio = torch.rand(1) * (1 / self.r1 - self.r1) + self.r1
            aspect_ratio = aspect_ratio.item()
            
            h = int(round((target_area * aspect_ratio) ** 0.5))
            w = int(round((target_area / aspect_ratio) ** 0.5))
            
            if w < img.size()[2] and h < img.size()[1]:
                x1 = torch.randint(0, img.size()[1] - h + 1, (1,)).item()
                y1 = torch.randint(0, img.size()[2] - w + 1, (1,)).item()
                
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img
        
        return img


def get_transforms(config: Dict[str, Any]) -> Dict[str, transforms.Compose]:
    """
    Get data transforms based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of transforms for different data splits
    """
    h, w = config['data']['image_height'], config['data']['image_width']
    pad = config['data']['pad']
    erasing_p = config['data']['random_erasing_prob']
    color_jitter = config['data']['color_jitter']
    use_da = config['training']['use_data_augmentation']
    
    # Base transforms for drone/street view
    transform_drone_list = [
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Pad(pad, padding_mode='edge'),
        transforms.RandomCrop((h, w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    
    # Base transforms for satellite view
    transform_satellite_list = [
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Pad(pad, padding_mode='edge'),
        transforms.RandomAffine(90),  # Random rotation for satellite images
        transforms.RandomCrop((h, w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    
    # Validation/test transforms
    transform_val_list = [
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    
    # Add random erasing if specified
    if erasing_p > 0:
        transform_drone_list.append(RandomErasing(probability=erasing_p))
        transform_satellite_list.append(RandomErasing(probability=erasing_p))
    
    # Add color jitter if specified
    if color_jitter:
        color_jitter_transform = transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0
        )
        transform_drone_list.insert(-3, color_jitter_transform)
        transform_satellite_list.insert(-3, color_jitter_transform)
    
    # Add data augmentation if specified
    if use_da:
        # Add more aggressive augmentations
        additional_augs = [
            transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        ]
        transform_drone_list = additional_augs + transform_drone_list
        transform_satellite_list = additional_augs + transform_satellite_list
    
    return {
        'train': transforms.Compose(transform_drone_list),
        'satellite': transforms.Compose(transform_satellite_list),
        'val': transforms.Compose(transform_val_list),
        'test': transforms.Compose(transform_val_list)
    }
