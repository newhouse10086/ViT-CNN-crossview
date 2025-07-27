"""Optimizer factory for ViT-CNN-crossview."""

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from typing import Dict, Any, List
import logging

from .lr_schedulers import WarmupCosineScheduler, WarmupLinearScheduler

logger = logging.getLogger(__name__)


def create_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
        
    Returns:
        Optimizer
    """
    training_config = config['training']
    
    lr = training_config['learning_rate']
    weight_decay = training_config['weight_decay']
    momentum = training_config.get('momentum', 0.9)
    
    # Separate backbone and other parameters for different learning rates
    backbone_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if any(backbone_name in name.lower() for backbone_name in 
               ['backbone', 'resnet', 'vit', 'transformer']):
            backbone_params.append(param)
        else:
            other_params.append(param)
    
    # Create parameter groups with different learning rates
    param_groups = []
    
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': lr * 0.1,  # Lower learning rate for backbone
            'weight_decay': weight_decay
        })
        logger.info(f"Backbone parameters: {len(backbone_params)} with lr={lr * 0.1}")
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': lr,
            'weight_decay': weight_decay
        })
        logger.info(f"Other parameters: {len(other_params)} with lr={lr}")
    
    # Create optimizer
    optimizer_type = training_config.get('optimizer', 'sgd').lower()
    
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            momentum=momentum,
            nesterov=True
        )
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(
            param_groups,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(
            param_groups,
            momentum=momentum,
            alpha=0.99
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    logger.info(f"Created {optimizer_type.upper()} optimizer with {len(param_groups)} parameter groups")
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]) -> lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer to schedule
        config: Configuration dictionary
        
    Returns:
        Learning rate scheduler
    """
    training_config = config['training']
    
    scheduler_type = training_config.get('scheduler', 'step').lower()
    num_epochs = training_config['num_epochs']
    warm_epochs = training_config.get('warm_epochs', 0)
    
    if scheduler_type == 'step':
        steps = training_config.get('lr_scheduler_steps', [70, 110])
        gamma = training_config.get('lr_scheduler_gamma', 0.1)
        
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=steps,
            gamma=gamma
        )
        
    elif scheduler_type == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )
        
    elif scheduler_type == 'warmup_cosine':
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warm_epochs,
            max_epochs=num_epochs,
            eta_min=1e-6
        )
        
    elif scheduler_type == 'warmup_linear':
        scheduler = WarmupLinearScheduler(
            optimizer,
            warmup_epochs=warm_epochs,
            max_epochs=num_epochs
        )
        
    elif scheduler_type == 'exponential':
        gamma = training_config.get('lr_scheduler_gamma', 0.95)
        scheduler = lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma
        )
        
    elif scheduler_type == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    logger.info(f"Created {scheduler_type} scheduler")
    
    return scheduler


def get_learning_rates(optimizer: optim.Optimizer) -> List[float]:
    """
    Get current learning rates from optimizer.
    
    Args:
        optimizer: Optimizer
        
    Returns:
        List of learning rates for each parameter group
    """
    return [group['lr'] for group in optimizer.param_groups]


def set_learning_rates(optimizer: optim.Optimizer, learning_rates: List[float]):
    """
    Set learning rates for optimizer.
    
    Args:
        optimizer: Optimizer
        learning_rates: List of learning rates for each parameter group
    """
    assert len(learning_rates) == len(optimizer.param_groups), \
        "Number of learning rates must match number of parameter groups"
    
    for group, lr in zip(optimizer.param_groups, learning_rates):
        group['lr'] = lr


def warmup_learning_rate(optimizer: optim.Optimizer, epoch: int, warmup_epochs: int,
                        base_lrs: List[float], warmup_factor: float = 0.1):
    """
    Apply warmup to learning rate.
    
    Args:
        optimizer: Optimizer
        epoch: Current epoch
        warmup_epochs: Number of warmup epochs
        base_lrs: Base learning rates
        warmup_factor: Warmup factor
    """
    if epoch < warmup_epochs:
        # Linear warmup
        alpha = epoch / warmup_epochs
        warmup_lrs = [warmup_factor * lr + alpha * (lr - warmup_factor * lr) 
                     for lr in base_lrs]
        set_learning_rates(optimizer, warmup_lrs)


def freeze_parameters(model: torch.nn.Module, freeze_patterns: List[str]):
    """
    Freeze parameters matching given patterns.
    
    Args:
        model: Model to freeze parameters
        freeze_patterns: List of patterns to match parameter names
    """
    frozen_count = 0
    
    for name, param in model.named_parameters():
        if any(pattern in name.lower() for pattern in freeze_patterns):
            param.requires_grad = False
            frozen_count += 1
            logger.info(f"Frozen parameter: {name}")
    
    logger.info(f"Frozen {frozen_count} parameters")


def unfreeze_parameters(model: torch.nn.Module, unfreeze_patterns: List[str]):
    """
    Unfreeze parameters matching given patterns.
    
    Args:
        model: Model to unfreeze parameters
        unfreeze_patterns: List of patterns to match parameter names
    """
    unfrozen_count = 0
    
    for name, param in model.named_parameters():
        if any(pattern in name.lower() for pattern in unfreeze_patterns):
            param.requires_grad = True
            unfrozen_count += 1
            logger.info(f"Unfrozen parameter: {name}")
    
    logger.info(f"Unfrozen {unfrozen_count} parameters")


def get_parameter_count(model: torch.nn.Module) -> Dict[str, int]:
    """
    Get parameter count statistics.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }


def print_optimizer_info(optimizer: optim.Optimizer):
    """
    Print optimizer information.
    
    Args:
        optimizer: Optimizer to analyze
    """
    print(f"\nOptimizer Information:")
    print(f"  Type: {type(optimizer).__name__}")
    print(f"  Parameter groups: {len(optimizer.param_groups)}")
    
    for i, group in enumerate(optimizer.param_groups):
        param_count = sum(p.numel() for p in group['params'])
        print(f"  Group {i}: {param_count:,} parameters, lr={group['lr']:.6f}")


def create_optimizer_with_config(model: torch.nn.Module, config: Dict[str, Any]) -> tuple:
    """
    Create optimizer and scheduler with configuration.
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Print information
    print_optimizer_info(optimizer)
    param_stats = get_parameter_count(model)
    print(f"\nModel Parameters:")
    print(f"  Total: {param_stats['total']:,}")
    print(f"  Trainable: {param_stats['trainable']:,}")
    print(f"  Frozen: {param_stats['frozen']:,}")
    
    return optimizer, scheduler
