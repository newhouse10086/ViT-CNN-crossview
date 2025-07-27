"""Logging utilities for ViT-CNN-crossview."""

import logging
import sys
from pathlib import Path
from typing import Optional
import time


def setup_logger(name: str = "ViT-CNN-crossview", 
                log_file: Optional[str] = None,
                log_level: str = "INFO",
                console_output: bool = True) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file
        log_level: Logging level
        console_output: Whether to output to console
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "ViT-CNN-crossview") -> logging.Logger:
    """
    Get existing logger.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TrainingLogger:
    """Logger for training progress."""
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = "experiment"):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # Setup logger
        log_file = self.log_dir / f"{experiment_name}_train.log"
        self.logger = setup_logger(
            name=f"training_{experiment_name}",
            log_file=str(log_file),
            log_level="INFO"
        )
        
        # Training state
        self.start_time = None
        self.epoch_start_time = None
    
    def log_training_start(self, config: dict):
        """
        Log training start information.
        
        Args:
            config: Training configuration
        """
        self.start_time = time.time()
        self.logger.info("=" * 80)
        self.logger.info(f"Starting training: {self.experiment_name}")
        self.logger.info("=" * 80)
        
        # Log configuration
        self.logger.info("Training Configuration:")
        for key, value in config.items():
            if isinstance(value, dict):
                self.logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    self.logger.info(f"    {sub_key}: {sub_value}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """
        Log epoch start.
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
        """
        self.epoch_start_time = time.time()
        self.logger.info("-" * 60)
        self.logger.info(f"Epoch {epoch}/{total_epochs}")
        self.logger.info("-" * 60)
    
    def log_epoch_end(self, epoch: int, metrics: dict):
        """
        Log epoch end with metrics.
        
        Args:
            epoch: Current epoch
            metrics: Epoch metrics
        """
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        
        # Log metrics
        self.logger.info("Epoch Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.6f}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_training_end(self, final_metrics: dict):
        """
        Log training completion.
        
        Args:
            final_metrics: Final training metrics
        """
        if self.start_time:
            total_time = time.time() - self.start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            
            self.logger.info("=" * 80)
            self.logger.info(f"Training completed in {hours:02d}:{minutes:02d}:{seconds:02d}")
            self.logger.info("=" * 80)
        
        # Log final metrics
        self.logger.info("Final Metrics:")
        for key, value in final_metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.6f}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_model_info(self, model_info: dict):
        """
        Log model information.
        
        Args:
            model_info: Model information dictionary
        """
        self.logger.info("Model Information:")
        for key, value in model_info.items():
            if isinstance(value, int):
                self.logger.info(f"  {key}: {value:,}")
            elif isinstance(value, float):
                self.logger.info(f"  {key}: {value:.2f}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_error(self, error_msg: str, exception: Optional[Exception] = None):
        """
        Log error message.
        
        Args:
            error_msg: Error message
            exception: Exception object (optional)
        """
        self.logger.error(error_msg)
        if exception:
            self.logger.error(f"Exception: {str(exception)}")
    
    def log_warning(self, warning_msg: str):
        """
        Log warning message.
        
        Args:
            warning_msg: Warning message
        """
        self.logger.warning(warning_msg)
    
    def log_info(self, info_msg: str):
        """
        Log info message.
        
        Args:
            info_msg: Info message
        """
        self.logger.info(info_msg)


class MetricsLogger:
    """Logger for metrics tracking."""
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = "experiment"):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # Metrics storage
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.log"
        
        # Setup logger
        self.logger = setup_logger(
            name=f"metrics_{experiment_name}",
            log_file=str(self.metrics_file),
            log_level="INFO",
            console_output=False
        )
    
    def log_metrics(self, epoch: int, phase: str, metrics: dict):
        """
        Log metrics for an epoch and phase.
        
        Args:
            epoch: Current epoch
            phase: Training phase (train/val/test)
            metrics: Metrics dictionary
        """
        # Create metrics string
        metrics_str = ", ".join([f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" 
                                for k, v in metrics.items()])
        
        self.logger.info(f"Epoch: {epoch}, Phase: {phase}, {metrics_str}")
    
    def log_best_metrics(self, metrics: dict):
        """
        Log best metrics achieved.
        
        Args:
            metrics: Best metrics dictionary
        """
        self.logger.info("BEST METRICS:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.6f}")
            else:
                self.logger.info(f"  {key}: {value}")


def log_system_info():
    """Log system information."""
    import torch
    import platform
    import psutil
    
    logger = get_logger()
    
    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"  CUDA Version: {torch.version.cuda}")
        logger.info(f"  GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Memory info
    memory = psutil.virtual_memory()
    logger.info(f"  RAM: {memory.total / 1024**3:.1f} GB total, {memory.available / 1024**3:.1f} GB available")
    
    # CPU info
    logger.info(f"  CPU: {platform.processor()}")
    logger.info(f"  CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
