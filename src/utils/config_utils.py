"""Configuration utilities for ViT-CNN-crossview."""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_config(config: Dict[str, Any], save_path: Union[str, Path]):
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        if save_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif save_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {save_path.suffix}")
    
    logger.info(f"Saved configuration to {save_path}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_sections = ['model', 'data', 'training', 'system']
    
    for section in required_sections:
        if section not in config:
            logger.error(f"Missing required configuration section: {section}")
            return False
    
    # Validate model section
    model_config = config['model']
    required_model_keys = ['name', 'num_classes']
    for key in required_model_keys:
        if key not in model_config:
            logger.error(f"Missing required model configuration: {key}")
            return False
    
    # Validate data section
    data_config = config['data']
    required_data_keys = ['data_dir', 'batch_size', 'views']
    for key in required_data_keys:
        if key not in data_config:
            logger.error(f"Missing required data configuration: {key}")
            return False
    
    # Validate training section
    training_config = config['training']
    required_training_keys = ['num_epochs', 'learning_rate']
    for key in required_training_keys:
        if key not in training_config:
            logger.error(f"Missing required training configuration: {key}")
            return False
    
    # Validate system section
    system_config = config['system']
    required_system_keys = ['gpu_ids']
    for key in required_system_keys:
        if key not in system_config:
            logger.error(f"Missing required system configuration: {key}")
            return False
    
    logger.info("Configuration validation passed")
    return True


def update_config_from_args(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """
    Update configuration with command line arguments.
    
    Args:
        config: Base configuration
        args: Command line arguments
        
    Returns:
        Updated configuration
    """
    updated_config = config.copy()
    
    # Map command line arguments to configuration keys
    arg_mapping = {
        'data_dir': ('data', 'data_dir'),
        'batch_size': ('data', 'batch_size'),
        'learning_rate': ('training', 'learning_rate'),
        'num_epochs': ('training', 'num_epochs'),
        'gpu_ids': ('system', 'gpu_ids'),
        'model_name': ('model', 'name'),
        'num_classes': ('model', 'num_classes'),
    }
    
    for arg_name, (section, key) in arg_mapping.items():
        if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
            if section not in updated_config:
                updated_config[section] = {}
            updated_config[section][key] = getattr(args, arg_name)
            logger.info(f"Updated config: {section}.{key} = {getattr(args, arg_name)}")
    
    return updated_config


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'model.num_classes')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def set_config_value(config: Dict[str, Any], key_path: str, value: Any):
    """
    Set configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'model.num_classes')
        value: Value to set
    """
    keys = key_path.split('.')
    current = config
    
    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the final value
    current[keys[-1]] = value


def create_experiment_config(base_config_path: Union[str, Path],
                           experiment_name: str,
                           overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create experiment configuration with overrides.
    
    Args:
        base_config_path: Path to base configuration
        experiment_name: Name of the experiment
        overrides: Configuration overrides
        
    Returns:
        Experiment configuration
    """
    # Load base configuration
    config = load_config(base_config_path)
    
    # Apply overrides
    if overrides:
        config = merge_configs(config, overrides)
    
    # Set experiment-specific values
    config['experiment_name'] = experiment_name
    config['system']['log_dir'] = f"logs/{experiment_name}"
    config['system']['checkpoint_dir'] = f"checkpoints/{experiment_name}"
    
    return config


def print_config(config: Dict[str, Any], title: str = "Configuration"):
    """
    Print configuration in a formatted way.
    
    Args:
        config: Configuration dictionary
        title: Title for the configuration
    """
    print(f"\n{title}:")
    print("=" * (len(title) + 1))
    _print_config_recursive(config, indent=0)
    print()


def _print_config_recursive(config: Dict[str, Any], indent: int = 0):
    """
    Recursively print configuration.
    
    Args:
        config: Configuration dictionary
        indent: Current indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            _print_config_recursive(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


def backup_config(config: Dict[str, Any], backup_dir: Union[str, Path],
                 experiment_name: str):
    """
    Backup configuration for experiment reproducibility.
    
    Args:
        config: Configuration dictionary
        backup_dir: Directory to save backup
        experiment_name: Name of the experiment
    """
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp to config
    import datetime
    config_with_timestamp = config.copy()
    config_with_timestamp['backup_timestamp'] = datetime.datetime.now().isoformat()
    
    # Save backup
    backup_path = backup_dir / f"{experiment_name}_config_backup.yaml"
    save_config(config_with_timestamp, backup_path)
    
    logger.info(f"Configuration backed up to {backup_path}")


def load_config_with_fallback(config_paths: List[Union[str, Path]]) -> Dict[str, Any]:
    """
    Load configuration with fallback options.
    
    Args:
        config_paths: List of configuration paths to try
        
    Returns:
        Loaded configuration
    """
    for config_path in config_paths:
        try:
            return load_config(config_path)
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {config_path}")
            continue
        except Exception as e:
            logger.warning(f"Error loading configuration from {config_path}: {e}")
            continue
    
    raise FileNotFoundError(f"Could not load configuration from any of: {config_paths}")
