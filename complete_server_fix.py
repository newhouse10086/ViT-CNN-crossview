#!/usr/bin/env python3
"""Complete server-side fix for all issues."""

import os
import shutil
from pathlib import Path

def fix_utils_init():
    """Fix src/utils/__init__.py"""
    content = '''"""Utilities for ViT-CNN-crossview."""

from .metrics import MetricsCalculator, AverageMeter, RankingMetricsCalculator
from .visualization import TrainingVisualizer, plot_training_curves, plot_confusion_matrix, plot_roc_curves
from .logger import setup_logger, get_logger, log_system_info
from .config_utils import load_config, save_config, merge_configs, validate_config

__all__ = [
    'MetricsCalculator',
    'AverageMeter',
    'RankingMetricsCalculator',
    'TrainingVisualizer',
    'plot_training_curves',
    'plot_confusion_matrix',
    'plot_roc_curves',
    'setup_logger',
    'get_logger',
    'log_system_info',
    'load_config',
    'save_config',
    'merge_configs',
    'validate_config'
]'''
    
    with open('src/utils/__init__.py', 'w') as f:
        f.write(content)
    print("✓ Fixed src/utils/__init__.py")

def fix_optimizers_init():
    """Fix src/optimizers/__init__.py"""
    content = '''"""Optimizers for ViT-CNN-crossview."""

from .optimizer_factory import create_optimizer, create_scheduler, create_optimizer_with_config
from .lr_schedulers import WarmupCosineScheduler, WarmupLinearScheduler

__all__ = [
    'create_optimizer',
    'create_scheduler',
    'create_optimizer_with_config',
    'WarmupCosineScheduler',
    'WarmupLinearScheduler'
]'''
    
    with open('src/optimizers/__init__.py', 'w') as f:
        f.write(content)
    print("✓ Fixed src/optimizers/__init__.py")

def fix_vit_pytorch():
    """Fix patch_size conflict in vit_pytorch.py"""
    vit_file = 'src/models/backbones/vit_pytorch.py'

    if not os.path.exists(vit_file):
        print(f"Warning: {vit_file} not found")
        return

    with open(vit_file, 'r') as f:
        content = f.read()

    # Fix the vit_small_patch16_224 function
    old_code = '''    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)'''

    new_code = '''    # Set default values, but allow kwargs to override
    default_kwargs = {
        'patch_size': 16,
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'mlp_ratio': 4,
        'qkv_bias': True,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6)
    }

    # Update defaults with provided kwargs
    default_kwargs.update(kwargs)

    model = VisionTransformer(**default_kwargs)'''

    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(vit_file, 'w') as f:
            f.write(content)
        print("✓ Fixed patch_size conflict in vit_pytorch.py")
    else:
        print("⚠ vit_pytorch.py patch_size fix not needed or already applied")

def fix_weights_init():
    """Fix weights initialization functions in components.py"""
    components_file = 'src/models/components.py'

    if not os.path.exists(components_file):
        print(f"Warning: {components_file} not found")
        return

    with open(components_file, 'r') as f:
        content = f.read()

    # Fix weights_init_classifier
    old_classifier = '''        if m.bias:
            nn.init.constant_(m.bias, 0.0)'''
    new_classifier = '''        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)'''

    # Fix weights_init_kaiming for Linear layers
    old_kaiming = '''    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)'''
    new_kaiming = '''    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)'''

    fixed = False
    if old_classifier in content:
        content = content.replace(old_classifier, new_classifier)
        fixed = True

    if old_kaiming in content:
        content = content.replace(old_kaiming, new_kaiming)
        fixed = True

    if fixed:
        with open(components_file, 'w') as f:
            f.write(content)
        print("✓ Fixed weights initialization functions in components.py")
    else:
        print("⚠ components.py weights init fix not needed or already applied")

def fix_sklearn_compatibility():
    """Fix sklearn compatibility issues in metrics.py"""
    metrics_file = 'src/utils/metrics.py'

    if not os.path.exists(metrics_file):
        print(f"Warning: {metrics_file} not found")
        return

    with open(metrics_file, 'r') as f:
        content = f.read()

    # Fix sklearn imports
    old_import = '''from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score, top_k_accuracy_score
)'''

    new_import = '''from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score
)

# Try to import top_k_accuracy_score, fallback if not available
try:
    from sklearn.metrics import top_k_accuracy_score
    HAS_TOP_K_ACCURACY = True
except ImportError:
    HAS_TOP_K_ACCURACY = False

    def top_k_accuracy_score(y_true, y_prob, k=1):
        """Fallback implementation for top-k accuracy."""
        if y_prob.ndim == 1:
            # Binary classification
            return accuracy_score(y_true, (y_prob > 0.5).astype(int))

        # Multi-class classification
        top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
        correct = np.any(top_k_preds == y_true.reshape(-1, 1), axis=1)
        return np.mean(correct)'''

    fixed = False
    if old_import in content:
        content = content.replace(old_import, new_import)
        fixed = True

    if fixed:
        with open(metrics_file, 'w') as f:
            f.write(content)
        print("✓ Fixed sklearn compatibility in metrics.py")
    else:
        print("⚠ metrics.py sklearn fix not needed or already applied")

def fix_matplotlib_compatibility():
    """Fix matplotlib/seaborn compatibility issues in visualization.py"""
    viz_file = 'src/utils/visualization.py'

    if not os.path.exists(viz_file):
        print(f"Warning: {viz_file} not found")
        return

    with open(viz_file, 'r') as f:
        content = f.read()

    # Fix matplotlib style
    old_style = '''# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")'''

    new_style = '''# Set style with compatibility fallback
try:
    # Try modern seaborn style first
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        # Fallback to older seaborn style
        plt.style.use('seaborn')
    except OSError:
        try:
            # Fallback to seaborn-whitegrid
            plt.style.use('seaborn-whitegrid')
        except OSError:
            # Final fallback to default
            plt.style.use('default')
            print("Warning: Using default matplotlib style (seaborn styles not available)")

# Set seaborn palette with error handling
try:
    sns.set_palette("husl")
except Exception:
    print("Warning: Could not set seaborn palette")'''

    fixed = False
    if old_style in content:
        content = content.replace(old_style, new_style)
        fixed = True

    if fixed:
        with open(viz_file, 'w') as f:
            f.write(content)
        print("✓ Fixed matplotlib compatibility in visualization.py")
    else:
        print("⚠ visualization.py matplotlib fix not needed or already applied")

def fix_torchvision_compatibility():
    """Fix torchvision compatibility issues in resnet.py"""
    resnet_file = 'src/models/backbones/resnet.py'

    if not os.path.exists(resnet_file):
        print(f"Warning: {resnet_file} not found")
        return

    with open(resnet_file, 'r') as f:
        content = f.read()

    # Fix torchvision ResNet API
    old_resnet = '''        # Load ResNet18
        if pretrained:
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=None)'''

    new_resnet = '''        # Load ResNet18 with version compatibility
        try:
            # Try new torchvision API (v0.13+)
            if pretrained:
                resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                resnet = models.resnet18(weights=None)
        except AttributeError:
            # Fallback to old torchvision API (< v0.13)
            resnet = models.resnet18(pretrained=pretrained)'''

    fixed = False
    if old_resnet in content:
        content = content.replace(old_resnet, new_resnet)
        fixed = True

    if fixed:
        with open(resnet_file, 'w') as f:
            f.write(content)
        print("✓ Fixed torchvision compatibility in resnet.py")
    else:
        print("⚠ resnet.py torchvision fix not needed or already applied")

def test_imports():
    """Test if imports work after fixes"""
    import sys
    sys.path.insert(0, 'src')
    
    # Test critical imports
    critical_imports = [
        ("src.utils", "RankingMetricsCalculator"),
        ("src.utils", "plot_roc_curves"),
        ("src.utils", "log_system_info"),
        ("src.utils", "validate_config"),
        ("src.optimizers", "create_optimizer_with_config"),
    ]
    
    all_good = True
    
    for module, item in critical_imports:
        try:
            exec(f"from {module} import {item}")
            print(f"✓ {item}")
        except Exception as e:
            print(f"✗ {item}: {e}")
            all_good = False
    
    return all_good

def test_model_creation():
    """Test model creation"""
    import sys
    sys.path.insert(0, 'src')
    
    try:
        from src.models import create_model
        
        config = {
            'model': {
                'name': 'ViTCNN',
                'num_classes': 10,
                'use_pretrained_resnet': False,
                'use_pretrained_vit': False
            },
            'data': {'views': 2}
        }
        
        model = create_model(config)
        print("✓ Model creation successful")
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def main():
    """Main fix function"""
    print("=" * 60)
    print("Complete ViT-CNN-crossview Server Fix")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("Error: Please run this script from the ViT-CNN-crossview directory")
        return 1
    
    # Apply all fixes
    print("Applying fixes...")
    fix_utils_init()
    fix_optimizers_init()
    fix_vit_pytorch()
    fix_weights_init()
    fix_sklearn_compatibility()
    fix_matplotlib_compatibility()
    fix_torchvision_compatibility()
    
    # Test imports
    print("\nTesting imports...")
    imports_ok = test_imports()
    
    # Test model creation
    print("\nTesting model creation...")
    model_ok = test_model_creation()
    
    # Final result
    print("\n" + "=" * 60)
    if imports_ok and model_ok:
        print("🎉 ALL FIXES APPLIED SUCCESSFULLY!")
        print("\nYou can now run:")
        print("  python train.py --config config/default_config.yaml")
        print("  python train.py --create-dummy-data --experiment-name test")
        print("\nThe following issues have been resolved:")
        print("  ✓ Missing import: RankingMetricsCalculator")
        print("  ✓ Missing import: create_optimizer_with_config")
        print("  ✓ Missing import: plot_roc_curves")
        print("  ✓ Missing import: log_system_info")
        print("  ✓ Missing import: validate_config")
        print("  ✓ patch_size conflict in VisionTransformer")
        print("  ✓ weights initialization bias check issue")
    else:
        print("❌ SOME ISSUES REMAIN")
        if not imports_ok:
            print("  - Import issues detected")
        if not model_ok:
            print("  - Model creation issues detected")
        return 1
    
    print("=" * 60)
    return 0

if __name__ == "__main__":
    exit(main())
