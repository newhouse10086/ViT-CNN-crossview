#!/usr/bin/env python3
"""
Debug ClassBlock behavior.
"""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_classblock():
    """Test ClassBlock behavior."""
    print("🔍 Testing ClassBlock Behavior")
    print("="*40)
    
    try:
        from src.models.components import ClassBlock
        
        # Create ClassBlock instances like in the model
        global_classifier = ClassBlock(
            input_dim=200,
            class_num=701,
            num_bottleneck=256,
            return_f=True
        )
        
        regional_classifier = ClassBlock(
            input_dim=256,
            class_num=701,
            num_bottleneck=256,
            return_f=True
        )
        
        final_classifier = ClassBlock(
            input_dim=256,
            class_num=701,
            num_bottleneck=256,
            return_f=True
        )
        
        print(f"✅ ClassBlock instances created successfully!")
        
        # Test inputs
        global_input = torch.randn(4, 200)
        regional_input = torch.randn(4, 256)
        final_input = torch.randn(4, 256)
        
        print(f"\n🧪 Testing Forward Passes:")
        
        # Test global classifier
        global_output = global_classifier(global_input)
        print(f"   Global classifier:")
        print(f"     Input: {global_input.shape}")
        print(f"     Output type: {type(global_output)}")
        if isinstance(global_output, list):
            print(f"     Output length: {len(global_output)}")
            for i, item in enumerate(global_output):
                print(f"       Item {i}: type={type(item)}, shape={item.shape if hasattr(item, 'shape') else 'no shape'}")
        elif isinstance(global_output, dict):
            print(f"     Output keys: {list(global_output.keys())}")
        else:
            print(f"     Output shape: {global_output.shape if hasattr(global_output, 'shape') else 'no shape'}")
        
        # Test regional classifier
        regional_output = regional_classifier(regional_input)
        print(f"   Regional classifier:")
        print(f"     Input: {regional_input.shape}")
        print(f"     Output type: {type(regional_output)}")
        if isinstance(regional_output, list):
            print(f"     Output length: {len(regional_output)}")
            for i, item in enumerate(regional_output):
                print(f"       Item {i}: type={type(item)}, shape={item.shape if hasattr(item, 'shape') else 'no shape'}")
        elif isinstance(regional_output, dict):
            print(f"     Output keys: {list(regional_output.keys())}")
        else:
            print(f"     Output shape: {regional_output.shape if hasattr(regional_output, 'shape') else 'no shape'}")
        
        # Test final classifier
        final_output = final_classifier(final_input)
        print(f"   Final classifier:")
        print(f"     Input: {final_input.shape}")
        print(f"     Output type: {type(final_output)}")
        if isinstance(final_output, list):
            print(f"     Output length: {len(final_output)}")
            for i, item in enumerate(final_output):
                print(f"       Item {i}: type={type(item)}, shape={item.shape if hasattr(item, 'shape') else 'no shape'}")
        elif isinstance(final_output, dict):
            print(f"     Output keys: {list(final_output.keys())}")
        else:
            print(f"     Output shape: {final_output.shape if hasattr(final_output, 'shape') else 'no shape'}")
        
        return True
        
    except Exception as e:
        print(f"❌ ClassBlock test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation to see where the dict comes from."""
    print("\n🔍 Testing Model Creation")
    print("="*40)
    
    try:
        from src.utils import load_config
        from src.models import create_model
        
        # Load config
        config = load_config("config/fsra_vit_improved_config.yaml")
        
        # Create model
        model = create_model(config)
        
        print(f"✅ Model created successfully!")
        
        # Check classifier types
        print(f"\n🔍 Checking Classifier Types:")
        print(f"   Global classifier type: {type(model.global_classifier)}")
        print(f"   Regional classifiers type: {type(model.regional_classifiers)}")
        print(f"   Final classifier type: {type(model.final_classifier)}")
        
        # Test a simple forward pass
        sat_img = torch.randn(2, 3, 256, 256)
        drone_img = torch.randn(2, 3, 256, 256)
        
        print(f"\n🧪 Testing Simple Forward Pass:")
        try:
            with torch.no_grad():
                outputs = model(sat_img, drone_img)
            print(f"   ✅ Forward pass successful!")
            print(f"   Output keys: {list(outputs.keys())}")
        except Exception as e:
            print(f"   ❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🎯 ClassBlock Debug Session")
    print("="*50)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: ClassBlock behavior
    if test_classblock():
        success_count += 1
    
    # Test 2: Model creation
    if test_model_creation():
        success_count += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"🎊 DEBUG SUMMARY")
    print(f"{'='*50}")
    print(f"Tests passed: {success_count}/{total_tests}")
    
    return success_count == total_tests


if __name__ == "__main__":
    main()
