"""Model factory for creating different types of models."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

from .vit_cnn_model import make_vit_cnn_model
from .two_view_model import make_fsra_model
from .fsra_improved import make_fsra_improved_model
from .two_view_fsra import make_two_view_fsra_improved
from .simple_fsra import make_simple_fsra_model
from .cross_attention import CrossAttentionModel

logger = logging.getLogger(__name__)


def create_model(config: Dict[str, Any]) -> nn.Module:
    """
    Create model based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Created model
    """
    model_config = config['model']
    data_config = config['data']
    
    model_name = model_config['name']
    num_classes = model_config['num_classes']
    views = data_config['views']
    
    logger.info(f"Creating model: {model_name}")
    
    if model_name.lower() == 'vitcnn':
        return create_vit_cnn_model(config)
    elif model_name.lower() == 'fsra':
        return create_fsra_model(config)
    elif model_name.lower() == 'fsra_improved':
        return create_fsra_improved_model(config)
    elif model_name.lower() == 'simple_fsra':
        return create_simple_fsra_model(config)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")


def create_vit_cnn_model(config: Dict[str, Any]) -> nn.Module:
    """
    Create ViT-CNN model based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ViT-CNN model
    """
    model_config = config['model']
    data_config = config['data']
    
    model = make_vit_cnn_model(
        num_classes=model_config['num_classes'],
        num_clusters=model_config.get('num_final_clusters', 3),
        use_pretrained_resnet=model_config.get('use_pretrained_resnet', True),
        use_pretrained_vit=model_config.get('use_pretrained_vit', False),
        return_f=model_config.get('return_features', True),
        views=data_config['views'],
        share_weights=model_config.get('share_weights', True)
    )
    
    logger.info(f"Created ViT-CNN model with {model_config['num_classes']} classes")
    return model


def create_fsra_model(config: Dict[str, Any]) -> nn.Module:
    """
    Create FSRA model based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        FSRA model
    """
    model_config = config['model']
    data_config = config['data']
    
    model = make_fsra_model(
        num_classes=model_config['num_classes'],
        block_size=model_config.get('block_size', 3),
        return_f=model_config.get('return_features', True),
        views=data_config['views'],
        share_weights=model_config.get('share_weights', True)
    )
    
    logger.info(f"Created FSRA model with {model_config['num_classes']} classes")
    return model


def create_cross_attention_model(config: Dict[str, Any]) -> CrossAttentionModel:
    """
    Create cross-attention model based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Cross-attention model
    """
    model_config = config['model']
    
    model = CrossAttentionModel(
        d_model=512,
        block_size=model_config.get('block_size', 3),
        num_heads=8
    )
    
    logger.info("Created cross-attention model")
    return model


def load_pretrained_weights(model: nn.Module, pretrained_path: str, 
                          strict: bool = False) -> nn.Module:
    """
    Load pretrained weights into model.
    
    Args:
        model: Model to load weights into
        pretrained_path: Path to pretrained weights
        strict: Whether to strictly enforce weight loading
        
    Returns:
        Model with loaded weights
    """
    try:
        if torch.cuda.is_available():
            checkpoint = torch.load(pretrained_path)
        else:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=strict)
        
        if missing_keys:
            logger.warning(f"Missing keys when loading pretrained weights: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
        
        logger.info(f"Successfully loaded pretrained weights from {pretrained_path}")
        
    except Exception as e:
        logger.error(f"Error loading pretrained weights from {pretrained_path}: {str(e)}")
        if strict:
            raise
    
    return model


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get model information including parameter counts and size.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # Estimate model size in MB
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size_mb = (param_size + buffer_size) / 1024 / 1024
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params,
        'model_size_mb': model_size_mb
    }
    
    return info


def print_model_info(model: nn.Module, model_name: str = "Model"):
    """
    Print model information.
    
    Args:
        model: Model to analyze
        model_name: Name of the model for display
    """
    info = get_model_info(model)
    
    print(f"\n{model_name} Information:")
    print(f"  Total parameters: {info['total_parameters']:,}")
    print(f"  Trainable parameters: {info['trainable_parameters']:,}")
    print(f"  Non-trainable parameters: {info['non_trainable_parameters']:,}")
    print(f"  Model size: {info['model_size_mb']:.2f} MB")


def freeze_model_layers(model: nn.Module, freeze_backbone: bool = False,
                       freeze_classifier: bool = False) -> nn.Module:
    """
    Freeze specific layers of the model.
    
    Args:
        model: Model to freeze layers
        freeze_backbone: Whether to freeze backbone layers
        freeze_classifier: Whether to freeze classifier layers
        
    Returns:
        Model with frozen layers
    """
    if freeze_backbone:
        # Freeze backbone layers
        for name, param in model.named_parameters():
            if 'backbone' in name or 'resnet' in name or 'vit' in name:
                param.requires_grad = False
                logger.info(f"Frozen layer: {name}")
    
    if freeze_classifier:
        # Freeze classifier layers
        for name, param in model.named_parameters():
            if 'classifier' in name or 'head' in name:
                param.requires_grad = False
                logger.info(f"Frozen layer: {name}")
    
    return model


def setup_model_for_training(model: nn.Module, config: Dict[str, Any]) -> nn.Module:
    """
    Setup model for training based on configuration.
    
    Args:
        model: Model to setup
        config: Configuration dictionary
        
    Returns:
        Configured model
    """
    model_config = config['model']
    
    # Load pretrained weights if specified
    if model_config.get('use_pretrained', False):
        pretrained_path = model_config.get('pretrained_path')
        if pretrained_path:
            model = load_pretrained_weights(model, pretrained_path, strict=False)
    
    # Freeze layers if specified
    freeze_backbone = model_config.get('freeze_backbone', False)
    freeze_classifier = model_config.get('freeze_classifier', False)
    
    if freeze_backbone or freeze_classifier:
        model = freeze_model_layers(model, freeze_backbone, freeze_classifier)
    
    # Print model information
    print_model_info(model, model_config['name'])
    
    return model


def create_fsra_improved_model(config: Dict[str, Any]) -> nn.Module:
    """
    Create FSRA Improved model based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        FSRA Improved model
    """
    model_config = config['model']
    data_config = config['data']

    num_classes = model_config['num_classes']
    num_clusters = model_config.get('num_final_clusters', 3)
    use_pretrained = model_config.get('use_pretrained_resnet', True)
    feature_dim = model_config.get('feature_dim', 512)
    share_weights = model_config.get('share_weights', True)
    views = data_config['views']

    logger.info(f"Creating FSRA Improved model with {num_classes} classes, {num_clusters} clusters")

    if views == 1:
        model = make_fsra_improved_model(
            num_classes=num_classes,
            num_clusters=num_clusters,
            use_pretrained=use_pretrained,
            feature_dim=feature_dim
        )
    elif views == 2:
        model = make_two_view_fsra_improved(
            num_classes=num_classes,
            num_clusters=num_clusters,
            use_pretrained=use_pretrained,
            feature_dim=feature_dim,
            share_weights=share_weights
        )
    else:
        raise ValueError(f"Unsupported number of views: {views}")

    return model


def create_simple_fsra_model(config: Dict[str, Any]) -> nn.Module:
    """
    Create Simple FSRA model based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Simple FSRA model
    """
    model_config = config['model']
    data_config = config['data']

    num_classes = model_config['num_classes']
    num_regions = model_config.get('num_regions', 4)
    use_pretrained = model_config.get('use_pretrained_resnet', True)
    share_weights = model_config.get('share_weights', True)
    views = data_config['views']

    logger.info(f"Creating Simple FSRA model with {num_classes} classes, {num_regions} regions")

    model = make_simple_fsra_model(
        num_classes=num_classes,
        num_regions=num_regions,
        use_pretrained=use_pretrained,
        views=views,
        share_weights=share_weights
    )

    return model
