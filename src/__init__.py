"""ViT-CNN-crossview: Advanced Deep Learning Framework for UAV Geo-Localization."""

__version__ = "1.0.0"
__author__ = "Research Team"
__email__ = "1914906669@qq.com"

# Import modules with error handling
try:
    from . import utils
except ImportError as e:
    print(f"Warning: Could not import utils: {e}")
    utils = None

try:
    from . import models
except ImportError as e:
    print(f"Warning: Could not import models: {e}")
    models = None

try:
    from . import datasets
except ImportError as e:
    print(f"Warning: Could not import datasets: {e}")
    datasets = None

try:
    from . import losses
except ImportError as e:
    print(f"Warning: Could not import losses: {e}")
    losses = None

try:
    from . import optimizers
except ImportError as e:
    print(f"Warning: Could not import optimizers: {e}")
    optimizers = None

try:
    from . import trainer
except ImportError as e:
    print(f"Warning: Could not import trainer: {e}")
    trainer = None

__all__ = [
    "models",
    "datasets",
    "losses",
    "optimizers",
    "utils",
    "trainer"
]
