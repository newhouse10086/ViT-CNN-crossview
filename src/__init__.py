"""ViT-CNN-crossview: Advanced Deep Learning Framework for UAV Geo-Localization."""

__version__ = "1.0.0"
__author__ = "Research Team"
__email__ = "1914906669@qq.com"

from . import models, datasets, losses, optimizers, utils, trainer

__all__ = [
    "models",
    "datasets", 
    "losses",
    "optimizers",
    "utils",
    "trainer"
]
